// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Utility kernels for Tensor operations:
//   - fill_kernel: fills a contiguous buffer with a constant value
//   - strided_copy_kernel: copies non-contiguous tensor to contiguous layout
//
// These are NOT hot-path kernels. They are only invoked for:
//   - Tensor::zeros(), ones(), fill_()
//   - Tensor::contiguous() on non-contiguous tensors
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "forge/tensor.h"
#include "forge/error.h"
#include "internal/cuda_utils.cuh"

namespace forge {
namespace detail {

// ---------------------------------------------------------------------------
// fill_kernel
// ---------------------------------------------------------------------------

template <typename T>
__global__ void fill_kernel(T* data, T value, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] = value;
    }
}

void fill_kernel_dispatch(void* data, float value, int64_t numel,
                          DType dtype, cudaStream_t stream) {
    if (numel == 0) return;

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(numel), block);

    switch (dtype) {
        case DType::Float32:
            fill_kernel<float><<<grid, block, 0, stream>>>(
                static_cast<float*>(data), value, numel);
            break;
        case DType::Float16:
            fill_kernel<__half><<<grid, block, 0, stream>>>(
                static_cast<__half*>(data), __float2half(value), numel);
            break;
        case DType::Int32:
            fill_kernel<int32_t><<<grid, block, 0, stream>>>(
                static_cast<int32_t*>(data), static_cast<int32_t>(value), numel);
            break;
        case DType::Int64:
            fill_kernel<int64_t><<<grid, block, 0, stream>>>(
                static_cast<int64_t*>(data), static_cast<int64_t>(value), numel);
            break;
        case DType::Int8:
            fill_kernel<int8_t><<<grid, block, 0, stream>>>(
                static_cast<int8_t*>(data), static_cast<int8_t>(value), numel);
            break;
        case DType::Bool:
            fill_kernel<uint8_t><<<grid, block, 0, stream>>>(
                static_cast<uint8_t*>(data), value != 0.0f ? 1 : 0, numel);
            break;
        default:
            break;
    }
}

// ---------------------------------------------------------------------------
// strided_copy_kernel
// ---------------------------------------------------------------------------

// General N-dimensional strided copy.
// Each thread copies one element, computing the source offset from the
// multi-dimensional index using the source strides.
//
// Handles:
//   - Negative strides (from flip())
//   - Zero strides (from expand() / broadcasting)
//   - Non-aligned offsets
//   - Multi-dimensional transposes

__global__ void strided_copy_kernel_general(
    const char* __restrict__ src,
    char* __restrict__ dst,
    int ndim,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ src_strides,
    const int64_t* __restrict__ dst_strides,
    size_t elem_size,
    int64_t numel)
{
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= numel) return;

    // Decompose linear index into multi-dimensional coordinates
    int64_t remaining = linear;
    int64_t src_offset = 0;
    int64_t dst_offset = 0;

    for (int d = ndim - 1; d >= 0; --d) {
        int64_t coord = remaining % sizes[d];
        remaining /= sizes[d];
        src_offset += coord * src_strides[d];
        dst_offset += coord * dst_strides[d];
    }

    // Byte-level copy for type genericity
    const char* src_ptr = src + src_offset * elem_size;
    char* dst_ptr = dst + dst_offset * elem_size;

    for (size_t b = 0; b < elem_size; ++b) {
        dst_ptr[b] = src_ptr[b];
    }
}

// Fast path: last dimension is contiguous in source.
// This covers the most common non-contiguous pattern (transpose of last two dims)
// and enables coalesced memory reads for the inner loop.
__global__ void strided_copy_kernel_lastdim_contig(
    const char* __restrict__ src,
    char* __restrict__ dst,
    int ndim,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ src_strides,
    const int64_t* __restrict__ dst_strides,
    size_t elem_size,
    int64_t numel,
    int64_t inner_size)
{
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= numel) return;

    // Split into outer index and inner index
    int64_t inner_idx = linear % inner_size;
    int64_t outer_linear = linear / inner_size;

    // Compute outer coordinates
    int64_t remaining = outer_linear;
    int64_t src_offset = 0;
    int64_t dst_offset = 0;

    for (int d = ndim - 2; d >= 0; --d) {
        int64_t coord = remaining % sizes[d];
        remaining /= sizes[d];
        src_offset += coord * src_strides[d];
        dst_offset += coord * dst_strides[d];
    }

    // Last dim is contiguous in source: stride == 1
    src_offset += inner_idx * src_strides[ndim - 1];
    dst_offset += inner_idx * dst_strides[ndim - 1];

    const char* src_ptr = src + src_offset * elem_size;
    char* dst_ptr = dst + dst_offset * elem_size;

    for (size_t b = 0; b < elem_size; ++b) {
        dst_ptr[b] = src_ptr[b];
    }
}

void strided_copy_dispatch(const void* src, void* dst,
                           int ndim,
                           const int64_t* sizes,
                           const int64_t* src_strides,
                           const int64_t* dst_strides,
                           DType dtype, cudaStream_t stream) {
    int64_t numel = compute_numel(sizes, ndim);
    if (numel == 0) return;

    size_t elem_size = dtype_size(dtype);

    // Copy sizes and strides to device
    // Use a single allocation for all three arrays
    int64_t* d_meta = nullptr;
    size_t meta_bytes = 3 * ndim * sizeof(int64_t);
    FORGE_CHECK(cudaMallocAsync(&d_meta, meta_bytes, stream));
    FORGE_CHECK(cudaMemcpyAsync(d_meta, sizes,
                                ndim * sizeof(int64_t),
                                cudaMemcpyHostToDevice, stream));
    FORGE_CHECK(cudaMemcpyAsync(d_meta + ndim, src_strides,
                                ndim * sizeof(int64_t),
                                cudaMemcpyHostToDevice, stream));
    FORGE_CHECK(cudaMemcpyAsync(d_meta + 2 * ndim, dst_strides,
                                ndim * sizeof(int64_t),
                                cudaMemcpyHostToDevice, stream));

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(numel), block);

    // Fast path: check if last dim is contiguous in source
    bool last_dim_contig = (ndim >= 2 && src_strides[ndim - 1] == 1);

    if (last_dim_contig) {
        strided_copy_kernel_lastdim_contig<<<grid, block, 0, stream>>>(
            static_cast<const char*>(src),
            static_cast<char*>(dst),
            ndim,
            d_meta,
            d_meta + ndim,
            d_meta + 2 * ndim,
            elem_size, numel, sizes[ndim - 1]);
    } else {
        strided_copy_kernel_general<<<grid, block, 0, stream>>>(
            static_cast<const char*>(src),
            static_cast<char*>(dst),
            ndim,
            d_meta,
            d_meta + ndim,
            d_meta + 2 * ndim,
            elem_size, numel);
    }

    FORGE_CHECK(cudaFreeAsync(d_meta, stream));
}

// ---------------------------------------------------------------------------
// cast_kernel: element-wise type conversion
// ---------------------------------------------------------------------------

// Generic cast: works for numeric types where static_cast is valid.
template <typename Src, typename Dst>
__global__ void cast_kernel(const Src* __restrict__ input,
                            Dst* __restrict__ output,
                            int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) output[idx] = static_cast<Dst>(input[idx]);
}

// Float32 → Float16
__global__ void cast_f32_to_f16(const float* __restrict__ input,
                                __half* __restrict__ output,
                                int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) output[idx] = __float2half(input[idx]);
}

// Float16 → Float32
__global__ void cast_f16_to_f32(const __half* __restrict__ input,
                                float* __restrict__ output,
                                int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) output[idx] = __half2float(input[idx]);
}

// Float32 → BFloat16
__global__ void cast_f32_to_bf16(const float* __restrict__ input,
                                 __nv_bfloat16* __restrict__ output,
                                 int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) output[idx] = __float2bfloat16(input[idx]);
}

// BFloat16 → Float32
__global__ void cast_bf16_to_f32(const __nv_bfloat16* __restrict__ input,
                                 float* __restrict__ output,
                                 int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) output[idx] = __bfloat162float(input[idx]);
}

// Any → Bool (nonzero test)
template <typename Src>
__global__ void cast_to_bool_kernel(const Src* __restrict__ input,
                                    uint8_t* __restrict__ output,
                                    int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) output[idx] = (input[idx] != static_cast<Src>(0)) ? 1 : 0;
}

__global__ void cast_f16_to_bool(const __half* __restrict__ input,
                                 uint8_t* __restrict__ output,
                                 int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) output[idx] = (__half2float(input[idx]) != 0.0f) ? 1 : 0;
}

__global__ void cast_bf16_to_bool(const __nv_bfloat16* __restrict__ input,
                                  uint8_t* __restrict__ output,
                                  int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) output[idx] = (__bfloat162float(input[idx]) != 0.0f) ? 1 : 0;
}

void cast_dispatch(const void* src, void* dst, int64_t numel,
                   DType src_dtype, DType dst_dtype, cudaStream_t stream) {
    if (numel == 0 || src_dtype == dst_dtype) return;

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(numel), block);

#define LAUNCH(SRC, DST) \
    cast_kernel<SRC, DST><<<grid, block, 0, stream>>>( \
        static_cast<const SRC*>(src), static_cast<DST*>(dst), numel)

    // ------- FROM Float32 -------
    if (src_dtype == DType::Float32) {
        switch (dst_dtype) {
            case DType::Float16:
                cast_f32_to_f16<<<grid, block, 0, stream>>>(
                    static_cast<const float*>(src),
                    static_cast<__half*>(dst), numel);
                return;
            case DType::BFloat16:
                cast_f32_to_bf16<<<grid, block, 0, stream>>>(
                    static_cast<const float*>(src),
                    static_cast<__nv_bfloat16*>(dst), numel);
                return;
            case DType::Int32:   LAUNCH(float, int32_t); return;
            case DType::Int64:   LAUNCH(float, int64_t); return;
            case DType::Int8:    LAUNCH(float, int8_t);  return;
            case DType::Bool:
                cast_to_bool_kernel<float><<<grid, block, 0, stream>>>(
                    static_cast<const float*>(src),
                    static_cast<uint8_t*>(dst), numel);
                return;
            default: break;
        }
    }

    // ------- TO Float32 -------
    if (dst_dtype == DType::Float32) {
        switch (src_dtype) {
            case DType::Float16:
                cast_f16_to_f32<<<grid, block, 0, stream>>>(
                    static_cast<const __half*>(src),
                    static_cast<float*>(dst), numel);
                return;
            case DType::BFloat16:
                cast_bf16_to_f32<<<grid, block, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(src),
                    static_cast<float*>(dst), numel);
                return;
            case DType::Int32:   LAUNCH(int32_t, float); return;
            case DType::Int64:   LAUNCH(int64_t, float); return;
            case DType::Int8:    LAUNCH(int8_t, float);  return;
            case DType::Bool:    LAUNCH(uint8_t, float); return;
            default: break;
        }
    }

    // ------- Direct non-Float32 conversions -------
    // Int32 ↔ Int64
    if (src_dtype == DType::Int32 && dst_dtype == DType::Int64) {
        LAUNCH(int32_t, int64_t); return;
    }
    if (src_dtype == DType::Int64 && dst_dtype == DType::Int32) {
        LAUNCH(int64_t, int32_t); return;
    }

    // Any → Bool
    if (dst_dtype == DType::Bool) {
        switch (src_dtype) {
            case DType::Int32:
                cast_to_bool_kernel<int32_t><<<grid, block, 0, stream>>>(
                    static_cast<const int32_t*>(src),
                    static_cast<uint8_t*>(dst), numel);
                return;
            case DType::Int64:
                cast_to_bool_kernel<int64_t><<<grid, block, 0, stream>>>(
                    static_cast<const int64_t*>(src),
                    static_cast<uint8_t*>(dst), numel);
                return;
            case DType::Int8:
                cast_to_bool_kernel<int8_t><<<grid, block, 0, stream>>>(
                    static_cast<const int8_t*>(src),
                    static_cast<uint8_t*>(dst), numel);
                return;
            case DType::Float16:
                cast_f16_to_bool<<<grid, block, 0, stream>>>(
                    static_cast<const __half*>(src),
                    static_cast<uint8_t*>(dst), numel);
                return;
            case DType::BFloat16:
                cast_bf16_to_bool<<<grid, block, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(src),
                    static_cast<uint8_t*>(dst), numel);
                return;
            default: break;
        }
    }

    // Unsupported direct conversion — caller should route through Float32
    std::fprintf(stderr,
        "Forge: cast_dispatch unsupported: %s → %s (route through Float32)\n",
        dtype_name(src_dtype), dtype_name(dst_dtype));
    std::abort();

#undef LAUNCH
}

} // namespace detail
} // namespace forge
