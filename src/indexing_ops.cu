// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Indexing and selection operations: kernels + dispatch.
#include "forge/indexing_ops.h"
#include "forge/tensor_ops.h"
#include "forge/error.h"
#include "internal/cuda_utils.cuh"

#include <cstdio>
#include <cstring>
#include <algorithm>

namespace forge {

// =====================================================================
// Internal helpers
// =====================================================================

namespace {

inline cudaStream_t resolve_stream(cudaStream_t stream) {
    return stream ? stream : getCurrentStream();
}

inline int normalize_dim(int dim, int ndim) {
    if (dim < 0) dim += ndim;
    return dim;
}

} // anonymous namespace

// =====================================================================
// index_select kernel
// =====================================================================

// Decompose: linear_idx → (outer, k, inner)
// output[outer][k][inner] = input[outer][index[k]][inner]
__global__ void index_select_kernel(
    const float* __restrict__   input,
    const int64_t* __restrict__ index,
    float* __restrict__         output,
    int64_t                     outer_size,
    int64_t                     index_size,
    int64_t                     inner_size,
    int64_t                     src_dim_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * index_size * inner_size;
    if (idx >= total) return;

    int64_t inner = idx % inner_size;
    int64_t k = (idx / inner_size) % index_size;
    int64_t outer = idx / (index_size * inner_size);

    int64_t src_idx = index[k];
    int64_t src_off = outer * src_dim_size * inner_size + src_idx * inner_size + inner;
    output[idx] = input[src_off];
}

Tensor index_select(const Tensor& input, int dim, const Tensor& index) {
    cudaStream_t stream = resolve_stream(nullptr);
    Tensor x = input.contiguous();
    Tensor idx = index.contiguous();
    int nd = x.ndim();
    dim = normalize_dim(dim, nd);

    int64_t K = idx.numel();

    // Compute output shape
    int64_t out_sizes[kMaxDim];
    for (int i = 0; i < nd; i++)
        out_sizes[i] = (i == dim) ? K : x.size(i);

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= x.size(i);
    int64_t inner_size = 1;
    for (int i = dim + 1; i < nd; i++) inner_size *= x.size(i);

    int64_t out_numel = outer_size * K * inner_size;
    Tensor out = Tensor::empty(out_sizes, nd, DType::Float32, stream);
    if (out_numel == 0) return out;

    constexpr int BLOCK = 256;
    int grid = internal::divup(static_cast<int>(out_numel), BLOCK);

    index_select_kernel<<<grid, BLOCK, 0, stream>>>(
        x.data<float>(), idx.data<int64_t>(), out.data<float>(),
        outer_size, K, inner_size, x.size(dim));

    return out;
}

// =====================================================================
// gather kernel
// =====================================================================

// out[i0][i1]...[i_n] = input[i0]...[index[i0..i_n] at dim]...[i_n]
// For contiguous tensors:
//   input_offset = (outer * input_dim_size + index_val) * inner_size + inner
__global__ void gather_kernel(
    const float* __restrict__   input,
    const int64_t* __restrict__ index,
    float* __restrict__         output,
    int64_t                     outer_size,
    int64_t                     dim_size,    // input.size(dim)
    int64_t                     inner_size,
    int64_t                     idx_dim_size) // index.size(dim)
{
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * idx_dim_size * inner_size;
    if (linear >= total) return;

    int64_t inner = linear % inner_size;
    int64_t k = (linear / inner_size) % idx_dim_size;
    int64_t outer = linear / (idx_dim_size * inner_size);

    int64_t idx_val = index[linear];
    int64_t src_off = outer * dim_size * inner_size + idx_val * inner_size + inner;
    output[linear] = input[src_off];
}

Tensor gather(const Tensor& input, int dim, const Tensor& index) {
    cudaStream_t stream = resolve_stream(nullptr);
    Tensor x = input.contiguous();
    Tensor idx = index.contiguous();
    int nd = x.ndim();
    dim = normalize_dim(dim, nd);

    // Output shape == index shape
    int64_t out_numel = idx.numel();
    Tensor out = Tensor::empty(idx.sizes(), idx.ndim(), DType::Float32, stream);
    if (out_numel == 0) return out;

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= idx.size(i);
    int64_t inner_size = 1;
    for (int i = dim + 1; i < idx.ndim(); i++) inner_size *= idx.size(i);

    constexpr int BLOCK = 256;
    int grid = internal::divup(static_cast<int>(out_numel), BLOCK);

    gather_kernel<<<grid, BLOCK, 0, stream>>>(
        x.data<float>(), idx.data<int64_t>(), out.data<float>(),
        outer_size, x.size(dim), inner_size, idx.size(dim));

    return out;
}

// =====================================================================
// scatter / scatter_add_ kernels
// =====================================================================

// scatter: self[outer][index[linear]][inner] = src[linear]
__global__ void scatter_kernel(
    const float* __restrict__   src,
    const int64_t* __restrict__ index,
    float* __restrict__         output,
    int64_t                     outer_size,
    int64_t                     out_dim_size,
    int64_t                     inner_size,
    int64_t                     idx_dim_size)
{
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * idx_dim_size * inner_size;
    if (linear >= total) return;

    int64_t inner = linear % inner_size;
    int64_t k = (linear / inner_size) % idx_dim_size;
    int64_t outer = linear / (idx_dim_size * inner_size);

    int64_t idx_val = index[linear];
    int64_t dst_off = outer * out_dim_size * inner_size + idx_val * inner_size + inner;
    output[dst_off] = src[linear];
}

__global__ void scatter_add_kernel(
    const float* __restrict__   src,
    const int64_t* __restrict__ index,
    float* __restrict__         output,
    int64_t                     outer_size,
    int64_t                     out_dim_size,
    int64_t                     inner_size,
    int64_t                     idx_dim_size)
{
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * idx_dim_size * inner_size;
    if (linear >= total) return;

    int64_t inner = linear % inner_size;
    int64_t k = (linear / inner_size) % idx_dim_size;
    int64_t outer = linear / (idx_dim_size * inner_size);

    int64_t idx_val = index[linear];
    int64_t dst_off = outer * out_dim_size * inner_size + idx_val * inner_size + inner;
    atomicAdd(&output[dst_off], src[linear]);
}

Tensor scatter(const Tensor& input, int dim, const Tensor& index, const Tensor& src) {
    cudaStream_t stream = resolve_stream(nullptr);
    Tensor x = input.contiguous();
    Tensor idx = index.contiguous();
    Tensor s = src.contiguous();
    int nd = x.ndim();
    dim = normalize_dim(dim, nd);

    // Output is a copy of input
    Tensor out = Tensor::empty(x.sizes(), nd, DType::Float32, stream);
    out.copy_(x);

    int64_t numel = idx.numel();
    if (numel == 0) return out;

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= idx.size(i);
    int64_t inner_size = 1;
    for (int i = dim + 1; i < idx.ndim(); i++) inner_size *= idx.size(i);

    constexpr int BLOCK = 256;
    int grid = internal::divup(static_cast<int>(numel), BLOCK);

    scatter_kernel<<<grid, BLOCK, 0, stream>>>(
        s.data<float>(), idx.data<int64_t>(), out.data<float>(),
        outer_size, x.size(dim), inner_size, idx.size(dim));

    return out;
}

void scatter_add_(Tensor& self, int dim, const Tensor& index, const Tensor& src) {
    cudaStream_t stream = resolve_stream(nullptr);
    Tensor idx = index.contiguous();
    Tensor s = src.contiguous();
    int nd = self.ndim();
    dim = normalize_dim(dim, nd);

    int64_t numel = idx.numel();
    if (numel == 0) return;

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= idx.size(i);
    int64_t inner_size = 1;
    for (int i = dim + 1; i < idx.ndim(); i++) inner_size *= idx.size(i);

    constexpr int BLOCK = 256;
    int grid = internal::divup(static_cast<int>(numel), BLOCK);

    scatter_add_kernel<<<grid, BLOCK, 0, stream>>>(
        s.data<float>(), idx.data<int64_t>(),
        static_cast<float*>(self.data_ptr()),
        outer_size, self.size(dim), inner_size, idx.size(dim));

    self.bump_version();
}

// =====================================================================
// masked_fill_ kernel
// =====================================================================

__global__ void masked_fill_contiguous_kernel(
    float* __restrict__         data,
    const uint8_t* __restrict__ mask,
    float                       value,
    int64_t                     numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel && mask[idx]) data[idx] = value;
}

__global__ void masked_fill_broadcast_kernel(
    float* __restrict__         data,
    const uint8_t* __restrict__ mask,
    float                       value,
    int                         ndim,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ data_strides,
    const int64_t* __restrict__ mask_strides,
    int64_t                     numel)
{
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= numel) return;

    int64_t remaining = linear;
    int64_t data_off = 0, mask_off = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        int64_t coord = remaining % sizes[d];
        remaining /= sizes[d];
        data_off += coord * data_strides[d];
        mask_off += coord * mask_strides[d];
    }

    if (mask[mask_off]) data[data_off] = value;
}

namespace {

inline void expand_strides(const int64_t* in_sizes, const int64_t* in_strides,
                           int in_ndim, const int64_t* out_sizes, int out_ndim,
                           int64_t* expanded) {
    int off = out_ndim - in_ndim;
    for (int i = 0; i < off; ++i)
        expanded[i] = 0;
    for (int i = 0; i < in_ndim; ++i)
        expanded[off + i] = (in_sizes[i] == out_sizes[off + i]) ? in_strides[i] : 0;
}

} // anonymous namespace

void masked_fill_(Tensor& self, const Tensor& mask, float value) {
    cudaStream_t stream = resolve_stream(nullptr);
    int64_t numel = self.numel();
    if (numel == 0) return;

    // Check if same shape and both contiguous
    bool same_shape = (self.ndim() == mask.ndim());
    if (same_shape) {
        for (int i = 0; i < self.ndim(); i++) {
            if (self.size(i) != mask.size(i)) { same_shape = false; break; }
        }
    }

    constexpr int BLOCK = 256;
    int grid = internal::divup(static_cast<int>(numel), BLOCK);

    if (same_shape && self.is_contiguous() && mask.is_contiguous()) {
        masked_fill_contiguous_kernel<<<grid, BLOCK, 0, stream>>>(
            static_cast<float*>(self.data_ptr()),
            static_cast<const uint8_t*>(mask.data_ptr()),
            value, numel);
    } else {
        // Broadcast mask to self's shape
        int nd = self.ndim();
        int64_t mask_strides_exp[kMaxDim];
        expand_strides(mask.sizes(), mask.strides(), mask.ndim(),
                       self.sizes(), nd, mask_strides_exp);

        int64_t* d_meta = nullptr;
        size_t meta_bytes = 3 * nd * sizeof(int64_t);
        FORGE_CHECK(cudaMallocAsync(&d_meta, meta_bytes, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta,          self.sizes(),   nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + nd,     self.strides(), nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + 2 * nd, mask_strides_exp, nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        masked_fill_broadcast_kernel<<<grid, BLOCK, 0, stream>>>(
            static_cast<float*>(self.data_ptr()),
            static_cast<const uint8_t*>(mask.data_ptr()),
            value, nd, d_meta, d_meta + nd, d_meta + 2 * nd, numel);

        FORGE_CHECK(cudaFreeAsync(d_meta, stream));
    }
    self.bump_version();
}

// =====================================================================
// masked_select: two-pass (count + compact)
// =====================================================================

__global__ void masked_count_kernel(
    const uint8_t* __restrict__ mask,
    int* __restrict__           count,
    int64_t                     numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel && mask[idx]) {
        atomicAdd(count, 1);
    }
}

__global__ void masked_compact_kernel(
    const float* __restrict__   input,
    const uint8_t* __restrict__ mask,
    float* __restrict__         output,
    int* __restrict__           counter,
    int64_t                     numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel && mask[idx]) {
        int pos = atomicAdd(counter, 1);
        output[pos] = input[idx];
    }
}

Tensor masked_select(const Tensor& input, const Tensor& mask) {
    cudaStream_t stream = resolve_stream(nullptr);
    Tensor x = input.contiguous();
    Tensor m = mask.contiguous();
    int64_t numel = x.numel();

    if (numel == 0)
        return Tensor::empty({0}, DType::Float32, stream);

    // Pass 1: count true elements
    int* d_count = nullptr;
    FORGE_CHECK(cudaMallocAsync(&d_count, sizeof(int), stream));
    FORGE_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), stream));

    constexpr int BLOCK = 256;
    int grid = internal::divup(static_cast<int>(numel), BLOCK);

    masked_count_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const uint8_t*>(m.data_ptr()),
        d_count, numel);

    int h_count = 0;
    FORGE_CHECK(cudaMemcpyAsync(&h_count, d_count, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    FORGE_CHECK(cudaStreamSynchronize(stream));

    if (h_count == 0) {
        FORGE_CHECK(cudaFreeAsync(d_count, stream));
        return Tensor::empty({0}, DType::Float32, stream);
    }

    // Pass 2: compact
    Tensor out = Tensor::empty({static_cast<int64_t>(h_count)}, DType::Float32, stream);

    // Reset counter for pass 2
    FORGE_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), stream));

    masked_compact_kernel<<<grid, BLOCK, 0, stream>>>(
        x.data<float>(),
        static_cast<const uint8_t*>(m.data_ptr()),
        out.data<float>(), d_count, numel);

    FORGE_CHECK(cudaFreeAsync(d_count, stream));
    return out;
}

} // namespace forge
