// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Comparison, logical, and conditional operations: kernels + dispatch.
//
// Comparison ops use cross-type kernels (float input -> uint8_t Bool output).
// Logical ops and where() have their own kernel variants.
#include "forge/comparison_ops.h"
#include "forge/tensor_ops.h" // for detail::broadcast_shapes
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

inline void expand_strides(const int64_t* in_sizes, const int64_t* in_strides,
                           int in_ndim, const int64_t* out_sizes, int out_ndim,
                           int64_t* expanded) {
    int off = out_ndim - in_ndim;
    for (int i = 0; i < off; ++i)
        expanded[i] = 0;
    for (int i = 0; i < in_ndim; ++i)
        expanded[off + i] = (in_sizes[i] == out_sizes[off + i]) ? in_strides[i] : 0;
}

inline cudaStream_t resolve_stream(cudaStream_t stream) {
    return stream ? stream : getCurrentStream();
}

} // anonymous namespace

// =====================================================================
// Comparison functor structs (float, float -> bool)
// =====================================================================

struct EqOp {
    __device__ __forceinline__ uint8_t operator()(float a, float b) const { return a == b ? 1 : 0; }
};
struct NeOp {
    __device__ __forceinline__ uint8_t operator()(float a, float b) const { return a != b ? 1 : 0; }
};
struct LtOp {
    __device__ __forceinline__ uint8_t operator()(float a, float b) const { return a < b ? 1 : 0; }
};
struct LeOp {
    __device__ __forceinline__ uint8_t operator()(float a, float b) const { return a <= b ? 1 : 0; }
};
struct GtOp {
    __device__ __forceinline__ uint8_t operator()(float a, float b) const { return a > b ? 1 : 0; }
};
struct GeOp {
    __device__ __forceinline__ uint8_t operator()(float a, float b) const { return a >= b ? 1 : 0; }
};

// =====================================================================
// Logical functor structs (bool, bool -> bool)
// =====================================================================

struct LogicalAndOp {
    __device__ __forceinline__ uint8_t operator()(uint8_t a, uint8_t b) const { return (a && b) ? 1 : 0; }
};
struct LogicalOrOp {
    __device__ __forceinline__ uint8_t operator()(uint8_t a, uint8_t b) const { return (a || b) ? 1 : 0; }
};
struct LogicalXorOp {
    __device__ __forceinline__ uint8_t operator()(uint8_t a, uint8_t b) const { return (a != b) ? 1 : 0; }
};
struct LogicalNotOp {
    __device__ __forceinline__ uint8_t operator()(uint8_t a) const { return a ? 0 : 1; }
};

// =====================================================================
// Special value check functors (float -> bool)
// =====================================================================

struct IsNanOp {
    __device__ __forceinline__ uint8_t operator()(float x) const { return ::isnan(x) ? 1 : 0; }
};
struct IsInfOp {
    __device__ __forceinline__ uint8_t operator()(float x) const { return ::isinf(x) ? 1 : 0; }
};
struct IsFiniteOp {
    __device__ __forceinline__ uint8_t operator()(float x) const { return ::isfinite(x) ? 1 : 0; }
};

// Scalar comparison wrapper (namespace-scope for NVCC)
template <typename CmpOp>
struct ScalarCmpWrap {
    CmpOp base_op;
    float s;
    __device__ __forceinline__ uint8_t operator()(float x) const {
        return base_op(x, s);
    }
};

// =====================================================================
// Kernel templates: comparison (float -> bool)
// =====================================================================

// Binary comparison: contiguous
template <typename CmpOp>
__global__ void cmp_contiguous_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    uint8_t* __restrict__     out,
    int64_t                   numel,
    CmpOp                     op)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) out[idx] = op(a[idx], b[idx]);
}

// Binary comparison: strided
template <typename CmpOp>
__global__ void cmp_strided_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    uint8_t* __restrict__     out,
    int                       ndim,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ a_strides,
    const int64_t* __restrict__ b_strides,
    const int64_t* __restrict__ out_strides,
    int64_t                   numel,
    CmpOp                     op)
{
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= numel) return;

    int64_t remaining = linear;
    int64_t a_off = 0, b_off = 0, out_off = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        int64_t coord = remaining % sizes[d];
        remaining /= sizes[d];
        a_off   += coord * a_strides[d];
        b_off   += coord * b_strides[d];
        out_off += coord * out_strides[d];
    }
    out[out_off] = op(a[a_off], b[b_off]);
}

// Unary float->bool: contiguous
template <typename UnaryOp>
__global__ void unary_f2b_contiguous_kernel(
    const float* __restrict__ in,
    uint8_t* __restrict__     out,
    int64_t                   numel,
    UnaryOp                   op)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) out[idx] = op(in[idx]);
}

// Unary float->bool: strided
template <typename UnaryOp>
__global__ void unary_f2b_strided_kernel(
    const float* __restrict__ in,
    uint8_t* __restrict__     out,
    int                       ndim,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ in_strides,
    const int64_t* __restrict__ out_strides,
    int64_t                   numel,
    UnaryOp                   op)
{
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= numel) return;

    int64_t remaining = linear;
    int64_t in_off = 0, out_off = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        int64_t coord = remaining % sizes[d];
        remaining /= sizes[d];
        in_off  += coord * in_strides[d];
        out_off += coord * out_strides[d];
    }
    out[out_off] = op(in[in_off]);
}

// =====================================================================
// Kernel templates: logical (bool -> bool)
// =====================================================================

template <typename LogOp>
__global__ void logical_binary_contiguous_kernel(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    uint8_t* __restrict__       out,
    int64_t                     numel,
    LogOp                       op)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) out[idx] = op(a[idx], b[idx]);
}

template <typename LogOp>
__global__ void logical_unary_contiguous_kernel(
    const uint8_t* __restrict__ in,
    uint8_t* __restrict__       out,
    int64_t                     numel,
    LogOp                       op)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) out[idx] = op(in[idx]);
}

// =====================================================================
// where kernel
// =====================================================================

__global__ void where_contiguous_kernel(
    const uint8_t* __restrict__ cond,
    const float* __restrict__   x,
    const float* __restrict__   y,
    float* __restrict__         out,
    int64_t                     numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) out[idx] = cond[idx] ? x[idx] : y[idx];
}

__global__ void where_strided_kernel(
    const uint8_t* __restrict__ cond,
    const float* __restrict__   x,
    const float* __restrict__   y,
    float* __restrict__         out,
    int                         ndim,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ cond_strides,
    const int64_t* __restrict__ x_strides,
    const int64_t* __restrict__ y_strides,
    const int64_t* __restrict__ out_strides,
    int64_t                     numel)
{
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= numel) return;

    int64_t remaining = linear;
    int64_t c_off = 0, x_off = 0, y_off = 0, o_off = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        int64_t coord = remaining % sizes[d];
        remaining /= sizes[d];
        c_off += coord * cond_strides[d];
        x_off += coord * x_strides[d];
        y_off += coord * y_strides[d];
        o_off += coord * out_strides[d];
    }
    out[o_off] = cond[c_off] ? x[x_off] : y[y_off];
}

// =====================================================================
// Reduction kernels for any() / all()
// =====================================================================

__global__ void reduce_any_kernel(
    const uint8_t* __restrict__ data,
    int* __restrict__           result,
    int64_t                     numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel && data[idx]) {
        atomicOr(result, 1);
    }
}

__global__ void reduce_all_kernel(
    const uint8_t* __restrict__ data,
    int* __restrict__           result,
    int64_t                     numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel && !data[idx]) {
        atomicAnd(result, 0);
    }
}

// =====================================================================
// Dispatch helpers
// =====================================================================

// Comparison: float,float -> Bool (with broadcasting)
template <typename CmpOp>
static Tensor cmp_op_impl(const Tensor& a, const Tensor& b, CmpOp op) {
    cudaStream_t stream = resolve_stream(nullptr);

    int64_t out_sizes[kMaxDim];
    int out_ndim = detail::broadcast_shapes(
        a.sizes(), a.ndim(), b.sizes(), b.ndim(), out_sizes);

    int64_t out_numel = compute_numel(out_sizes, out_ndim);
    if (out_numel == 0)
        return Tensor::empty(out_sizes, out_ndim, DType::Bool, stream);

    Tensor out = Tensor::empty(out_sizes, out_ndim, DType::Bool, stream);

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(out_numel), block);

    bool same_shape = (a.ndim() == b.ndim());
    if (same_shape) {
        for (int i = 0; i < a.ndim(); ++i) {
            if (a.size(i) != b.size(i)) { same_shape = false; break; }
        }
    }

    if (same_shape && a.is_contiguous() && b.is_contiguous()) {
        cmp_contiguous_kernel<<<grid, block, 0, stream>>>(
            a.data<float>(), b.data<float>(),
            static_cast<uint8_t*>(out.data_ptr()),
            out_numel, op);
    } else {
        int64_t a_strides_exp[kMaxDim], b_strides_exp[kMaxDim];
        expand_strides(a.sizes(), a.strides(), a.ndim(),
                       out_sizes, out_ndim, a_strides_exp);
        expand_strides(b.sizes(), b.strides(), b.ndim(),
                       out_sizes, out_ndim, b_strides_exp);

        int64_t* d_meta = nullptr;
        size_t meta_bytes = 4 * out_ndim * sizeof(int64_t);
        FORGE_CHECK(cudaMallocAsync(&d_meta, meta_bytes, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta,                out_sizes,       out_ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + out_ndim,     a_strides_exp,   out_ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + 2 * out_ndim, b_strides_exp,   out_ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + 3 * out_ndim, out.strides(),   out_ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        cmp_strided_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(a.data_ptr()),
            static_cast<const float*>(b.data_ptr()),
            static_cast<uint8_t*>(out.data_ptr()), out_ndim,
            d_meta, d_meta + out_ndim, d_meta + 2 * out_ndim,
            d_meta + 3 * out_ndim, out_numel, op);

        FORGE_CHECK(cudaFreeAsync(d_meta, stream));
    }
    return out;
}

// Comparison: float,scalar -> Bool
template <typename CmpOp>
static Tensor cmp_scalar_op_impl(const Tensor& a, float scalar, CmpOp op) {
    cudaStream_t stream = resolve_stream(nullptr);
    int64_t numel = a.numel();
    if (numel == 0)
        return Tensor::empty(a.sizes(), a.ndim(), DType::Bool, stream);

    Tensor out = Tensor::empty(a.sizes(), a.ndim(), DType::Bool, stream);

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(numel), block);

    if (a.is_contiguous()) {
        unary_f2b_contiguous_kernel<<<grid, block, 0, stream>>>(
            a.data<float>(),
            static_cast<uint8_t*>(out.data_ptr()),
            numel, ScalarCmpWrap<CmpOp>{op, scalar});
    } else {
        int nd = a.ndim();
        int64_t* d_meta = nullptr;
        size_t meta_bytes = 3 * nd * sizeof(int64_t);
        FORGE_CHECK(cudaMallocAsync(&d_meta, meta_bytes, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta,          a.sizes(),   nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + nd,     a.strides(), nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + 2 * nd, out.strides(), nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        unary_f2b_strided_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(a.data_ptr()),
            static_cast<uint8_t*>(out.data_ptr()), nd,
            d_meta, d_meta + nd, d_meta + 2 * nd,
            numel, ScalarCmpWrap<CmpOp>{op, scalar});

        FORGE_CHECK(cudaFreeAsync(d_meta, stream));
    }
    return out;
}

// Unary float->Bool (for isnan, isinf, isfinite)
template <typename UnaryOp>
static Tensor unary_f2b_impl(const Tensor& a, UnaryOp op) {
    cudaStream_t stream = resolve_stream(nullptr);
    int64_t numel = a.numel();
    if (numel == 0)
        return Tensor::empty(a.sizes(), a.ndim(), DType::Bool, stream);

    Tensor out = Tensor::empty(a.sizes(), a.ndim(), DType::Bool, stream);

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(numel), block);

    if (a.is_contiguous()) {
        unary_f2b_contiguous_kernel<<<grid, block, 0, stream>>>(
            a.data<float>(),
            static_cast<uint8_t*>(out.data_ptr()),
            numel, op);
    } else {
        int nd = a.ndim();
        int64_t* d_meta = nullptr;
        size_t meta_bytes = 3 * nd * sizeof(int64_t);
        FORGE_CHECK(cudaMallocAsync(&d_meta, meta_bytes, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta,          a.sizes(),   nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + nd,     a.strides(), nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + 2 * nd, out.strides(), nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        unary_f2b_strided_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(a.data_ptr()),
            static_cast<uint8_t*>(out.data_ptr()), nd,
            d_meta, d_meta + nd, d_meta + 2 * nd,
            numel, op);

        FORGE_CHECK(cudaFreeAsync(d_meta, stream));
    }
    return out;
}

// =====================================================================
// Public free functions: comparison
// =====================================================================

Tensor eq(const Tensor& a, const Tensor& b) { return cmp_op_impl(a, b, EqOp{}); }
Tensor ne(const Tensor& a, const Tensor& b) { return cmp_op_impl(a, b, NeOp{}); }
Tensor lt(const Tensor& a, const Tensor& b) { return cmp_op_impl(a, b, LtOp{}); }
Tensor le(const Tensor& a, const Tensor& b) { return cmp_op_impl(a, b, LeOp{}); }
Tensor gt(const Tensor& a, const Tensor& b) { return cmp_op_impl(a, b, GtOp{}); }
Tensor ge(const Tensor& a, const Tensor& b) { return cmp_op_impl(a, b, GeOp{}); }

Tensor eq(const Tensor& a, float s) { return cmp_scalar_op_impl(a, s, EqOp{}); }
Tensor ne(const Tensor& a, float s) { return cmp_scalar_op_impl(a, s, NeOp{}); }
Tensor lt(const Tensor& a, float s) { return cmp_scalar_op_impl(a, s, LtOp{}); }
Tensor le(const Tensor& a, float s) { return cmp_scalar_op_impl(a, s, LeOp{}); }
Tensor gt(const Tensor& a, float s) { return cmp_scalar_op_impl(a, s, GtOp{}); }
Tensor ge(const Tensor& a, float s) { return cmp_scalar_op_impl(a, s, GeOp{}); }

// =====================================================================
// Public free functions: logical
// =====================================================================

Tensor logical_and(const Tensor& a, const Tensor& b) {
    cudaStream_t stream = resolve_stream(nullptr);
    int64_t numel = a.numel();
    Tensor out = Tensor::empty(a.sizes(), a.ndim(), DType::Bool, stream);
    if (numel == 0) return out;

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(numel), block);
    logical_binary_contiguous_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(a.data_ptr()),
        static_cast<const uint8_t*>(b.data_ptr()),
        static_cast<uint8_t*>(out.data_ptr()),
        numel, LogicalAndOp{});
    return out;
}

Tensor logical_or(const Tensor& a, const Tensor& b) {
    cudaStream_t stream = resolve_stream(nullptr);
    int64_t numel = a.numel();
    Tensor out = Tensor::empty(a.sizes(), a.ndim(), DType::Bool, stream);
    if (numel == 0) return out;

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(numel), block);
    logical_binary_contiguous_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(a.data_ptr()),
        static_cast<const uint8_t*>(b.data_ptr()),
        static_cast<uint8_t*>(out.data_ptr()),
        numel, LogicalOrOp{});
    return out;
}

Tensor logical_xor(const Tensor& a, const Tensor& b) {
    cudaStream_t stream = resolve_stream(nullptr);
    int64_t numel = a.numel();
    Tensor out = Tensor::empty(a.sizes(), a.ndim(), DType::Bool, stream);
    if (numel == 0) return out;

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(numel), block);
    logical_binary_contiguous_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(a.data_ptr()),
        static_cast<const uint8_t*>(b.data_ptr()),
        static_cast<uint8_t*>(out.data_ptr()),
        numel, LogicalXorOp{});
    return out;
}

Tensor logical_not(const Tensor& a) {
    cudaStream_t stream = resolve_stream(nullptr);
    int64_t numel = a.numel();
    Tensor out = Tensor::empty(a.sizes(), a.ndim(), DType::Bool, stream);
    if (numel == 0) return out;

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(numel), block);
    logical_unary_contiguous_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(a.data_ptr()),
        static_cast<uint8_t*>(out.data_ptr()),
        numel, LogicalNotOp{});
    return out;
}

// =====================================================================
// Special value checks
// =====================================================================

Tensor isnan(const Tensor& a)    { return unary_f2b_impl(a, IsNanOp{}); }
Tensor isinf(const Tensor& a)    { return unary_f2b_impl(a, IsInfOp{}); }
Tensor isfinite(const Tensor& a) { return unary_f2b_impl(a, IsFiniteOp{}); }

// =====================================================================
// where(condition, x, y)
// =====================================================================

Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y) {
    cudaStream_t stream = resolve_stream(nullptr);

    // Broadcast all three shapes together
    int64_t tmp_sizes[kMaxDim], out_sizes[kMaxDim];
    int tmp_ndim = detail::broadcast_shapes(
        condition.sizes(), condition.ndim(), x.sizes(), x.ndim(), tmp_sizes);
    int out_ndim = detail::broadcast_shapes(
        tmp_sizes, tmp_ndim, y.sizes(), y.ndim(), out_sizes);

    int64_t out_numel = compute_numel(out_sizes, out_ndim);
    if (out_numel == 0)
        return Tensor::empty(out_sizes, out_ndim, x.dtype(), stream);

    Tensor out = Tensor::empty(out_sizes, out_ndim, x.dtype(), stream);

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(out_numel), block);

    // Check if all same shape and contiguous
    bool all_same = (condition.ndim() == x.ndim() && x.ndim() == y.ndim());
    if (all_same) {
        for (int i = 0; i < condition.ndim(); ++i) {
            if (condition.size(i) != x.size(i) || x.size(i) != y.size(i)) {
                all_same = false; break;
            }
        }
    }

    if (all_same && condition.is_contiguous() && x.is_contiguous() && y.is_contiguous()) {
        where_contiguous_kernel<<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(condition.data_ptr()),
            x.data<float>(), y.data<float>(),
            out.data<float>(), out_numel);
    } else {
        // Strided path
        int64_t c_str[kMaxDim], x_str[kMaxDim], y_str[kMaxDim];
        expand_strides(condition.sizes(), condition.strides(), condition.ndim(),
                       out_sizes, out_ndim, c_str);
        expand_strides(x.sizes(), x.strides(), x.ndim(),
                       out_sizes, out_ndim, x_str);
        expand_strides(y.sizes(), y.strides(), y.ndim(),
                       out_sizes, out_ndim, y_str);

        // Upload: sizes + c_strides + x_strides + y_strides + out_strides = 5*ndim
        int64_t* d_meta = nullptr;
        size_t meta_bytes = 5 * out_ndim * sizeof(int64_t);
        FORGE_CHECK(cudaMallocAsync(&d_meta, meta_bytes, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta,                out_sizes,     out_ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + out_ndim,     c_str,         out_ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + 2 * out_ndim, x_str,         out_ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + 3 * out_ndim, y_str,         out_ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + 4 * out_ndim, out.strides(), out_ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        where_strided_kernel<<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(condition.data_ptr()),
            static_cast<const float*>(x.data_ptr()),
            static_cast<const float*>(y.data_ptr()),
            out.data<float>(), out_ndim,
            d_meta, d_meta + out_ndim, d_meta + 2 * out_ndim,
            d_meta + 3 * out_ndim, d_meta + 4 * out_ndim,
            out_numel);

        FORGE_CHECK(cudaFreeAsync(d_meta, stream));
    }
    return out;
}

// =====================================================================
// allclose
// =====================================================================

bool allclose(const Tensor& a, const Tensor& b, float rtol, float atol) {
    Tensor diff = forge::abs(forge::sub(a, b));
    Tensor threshold = forge::add(forge::mul(forge::abs(b), rtol), atol);
    Tensor mask = forge::le(diff, threshold);
    return forge::all(mask);
}

// =====================================================================
// any() / all() reductions
// =====================================================================

bool any(const Tensor& a) {
    cudaStream_t stream = resolve_stream(nullptr);

    // Make contiguous for simplicity
    Tensor ac = a.contiguous();
    int64_t numel = ac.numel();
    if (numel == 0) return false;

    // Allocate device result (initialized to 0)
    int* d_result = nullptr;
    FORGE_CHECK(cudaMallocAsync(&d_result, sizeof(int), stream));
    FORGE_CHECK(cudaMemsetAsync(d_result, 0, sizeof(int), stream));

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(numel), block);

    reduce_any_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(ac.data_ptr()),
        d_result, numel);

    int h_result = 0;
    FORGE_CHECK(cudaMemcpyAsync(&h_result, d_result, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    FORGE_CHECK(cudaStreamSynchronize(stream));
    FORGE_CHECK(cudaFreeAsync(d_result, stream));

    return h_result != 0;
}

bool all(const Tensor& a) {
    cudaStream_t stream = resolve_stream(nullptr);

    Tensor ac = a.contiguous();
    int64_t numel = ac.numel();
    if (numel == 0) return true; // vacuous truth

    int* d_result = nullptr;
    FORGE_CHECK(cudaMallocAsync(&d_result, sizeof(int), stream));
    // Initialize to 1 (all true)
    int one = 1;
    FORGE_CHECK(cudaMemcpyAsync(d_result, &one, sizeof(int),
                                cudaMemcpyHostToDevice, stream));

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(numel), block);

    reduce_all_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(ac.data_ptr()),
        d_result, numel);

    int h_result = 0;
    FORGE_CHECK(cudaMemcpyAsync(&h_result, d_result, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    FORGE_CHECK(cudaStreamSynchronize(stream));
    FORGE_CHECK(cudaFreeAsync(d_result, stream));

    return h_result != 0;
}

} // namespace forge
