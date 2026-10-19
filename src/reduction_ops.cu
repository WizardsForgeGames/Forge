// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Reduction operations: kernels, dispatch, and public Tensor-level API.
//
// Full-tensor reductions: two-pass partial/final pattern (from nn/reduce.cu).
// Dimension-wise: outer/reduce/inner decomposition with block_reduce specialization.
// Statistical: var, std_dev, logsumexp composed from primitives.
#include "forge/reduction_ops.h"
#include "forge/tensor_ops.h"
#include "forge/autograd.h"
#include "forge/error.h"
#include "forge/primitives/reduce.cuh"
#include "internal/cuda_utils.cuh"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cfloat>
#include <cmath>

// Forward declarations for autograd recording (defined in autograd_ops.cu)
namespace forge {
void record_sum(Tensor& out, const Tensor& a);
void record_sum_dim(Tensor& out, const Tensor& a, int dim, bool keepdim);
void record_mean(Tensor& out, const Tensor& a);
void record_mean_dim(Tensor& out, const Tensor& a, int dim, bool keepdim);
} // namespace forge

namespace forge {

// =====================================================================
// Internal helpers
// =====================================================================

namespace {

inline cudaStream_t resolve_stream(cudaStream_t stream) {
    return stream ? stream : getCurrentStream();
}

// Compute output shape after reducing along dim.
// Returns output ndim.
inline int compute_reduced_shape(
    const int64_t* in_sizes, int in_ndim,
    int dim, bool keepdim,
    int64_t* out_sizes)
{
    int out_ndim = 0;
    for (int i = 0; i < in_ndim; i++) {
        if (i == dim) {
            if (keepdim) out_sizes[out_ndim++] = 1;
        } else {
            out_sizes[out_ndim++] = in_sizes[i];
        }
    }
    // Edge: if all dims removed, return scalar {1}
    if (out_ndim == 0) {
        out_sizes[0] = 1;
        out_ndim = 1;
    }
    return out_ndim;
}

// Decompose a tensor around the reduction dimension.
inline void compute_outer_reduce_inner(
    const int64_t* sizes, int ndim, int dim,
    int64_t& outer_size, int64_t& reduce_size, int64_t& inner_size)
{
    outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= sizes[i];
    reduce_size = sizes[dim];
    inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) inner_size *= sizes[i];
}

inline int normalize_dim(int dim, int ndim) {
    if (dim < 0) dim += ndim;
    return dim;
}

} // anonymous namespace

// =====================================================================
// Transform functors (applied per-element before reduction)
// =====================================================================

struct IdentityTransform {
    __device__ __forceinline__ float operator()(float x) const { return x; }
};

struct SquareTransform {
    __device__ __forceinline__ float operator()(float x) const { return x * x; }
};

struct AbsTransform {
    __device__ __forceinline__ float operator()(float x) const { return fabsf(x); }
};

struct PowAbsTransform {
    float p;
    __device__ __forceinline__ float operator()(float x) const { return powf(fabsf(x), p); }
};

// =====================================================================
// Two-pass full-tensor reduction (from nn/reduce.cu pattern)
// =====================================================================

template <typename Op, typename Transform>
static __global__ void reduce_partial_kernel(
    const float* __restrict__ x,
    float* __restrict__       partial,
    int                       n,
    Transform                 xform,
    float                     post_scale)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float val = Op::identity();
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
        val = Op::combine(val, xform(x[i]));

    val = primitives::block_reduce<Op>(val, smem);

    if (tid == 0) partial[blockIdx.x] = val * post_scale;
}

template <typename Op>
static __global__ void reduce_final_kernel(
    const float* __restrict__ partial,
    float* __restrict__       out,
    int                       n_partials,
    float                     post_scale)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float val = Op::identity();
    for (int i = tid; i < n_partials; i += blockDim.x)
        val = Op::combine(val, partial[i]);

    val = primitives::block_reduce<Op>(val, smem);

    if (tid == 0) out[0] = val * post_scale;
}

template <typename Op, typename Transform>
static void reduce_impl(
    const float* x,
    float*       out,
    int          n,
    Transform    xform,
    cudaStream_t stream,
    float        post_scale = 1.0f)
{
    if (n <= 0) return;

    constexpr int BLOCK = 256;
    int max_grid = internal::get_sm_count() * 2;
    int natural_grid = internal::divup(n, BLOCK);
    int grid = (natural_grid < max_grid) ? natural_grid : max_grid;

    size_t smem = (BLOCK / 32) * sizeof(float);

    if (grid == 1) {
        reduce_partial_kernel<Op><<<1, BLOCK, smem, stream>>>(
            x, out, n, xform, post_scale);
        return;
    }

    float* d_partial = nullptr;
    FORGE_CHECK(cudaMalloc(&d_partial, grid * sizeof(float)));

    reduce_partial_kernel<Op><<<grid, BLOCK, smem, stream>>>(
        x, d_partial, n, xform, 1.0f);

    int final_block = (grid <= 256) ? 256 : 1024;
    size_t final_smem = (final_block / 32) * sizeof(float);
    reduce_final_kernel<Op><<<1, final_block, final_smem, stream>>>(
        d_partial, out, grid, post_scale);

    cudaFreeAsync(d_partial, stream);
}

// Single-thread post-processing kernel (for sqrt, powf, etc.)
__global__ void scalar_post_sqrt_kernel(float* __restrict__ out) {
    out[0] = sqrtf(out[0]);
}

__global__ void scalar_post_pow_kernel(float* __restrict__ out, float inv_p) {
    out[0] = powf(out[0], inv_p);
}

// =====================================================================
// Dimension-wise reduction kernels
// =====================================================================

// General kernel: one thread per output element, serial loop over reduce dim.
template <typename Op>
__global__ void reduce_dim_general_kernel(
    const float* __restrict__ x,
    float* __restrict__       out,
    int64_t                   outer_size,
    int64_t                   reduce_size,
    int64_t                   inner_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_out = outer_size * inner_size;
    if (idx >= total_out) return;

    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;
    int64_t base = outer * reduce_size * inner_size + inner;

    float val = Op::identity();
    for (int64_t r = 0; r < reduce_size; r++)
        val = Op::combine(val, x[base + r * inner_size]);

    out[idx] = val;
}

// Inner-dim specialization: one block per output slice, parallel block_reduce.
// Used when inner_size == 1 (reducing the last dimension).
template <typename Op>
__global__ void reduce_dim_inner_kernel(
    const float* __restrict__ x,
    float* __restrict__       out,
    int                       reduce_size,
    int64_t                   outer_size)
{
    extern __shared__ float smem[];
    int64_t outer = blockIdx.x;
    if (outer >= outer_size) return;

    const float* row = x + outer * reduce_size;
    float val = Op::identity();
    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x)
        val = Op::combine(val, row[i]);

    val = primitives::block_reduce<Op>(val, smem);

    if (threadIdx.x == 0) out[outer] = val;
}

// Dispatch: chooses general vs inner-dim kernel.
template <typename Op>
static Tensor reduce_dim_impl(const Tensor& input, int dim, bool keepdim,
                               float post_scale = 1.0f) {
    cudaStream_t stream = resolve_stream(nullptr);
    Tensor x = input.contiguous();
    int nd = x.ndim();
    dim = normalize_dim(dim, nd);

    int64_t outer_size, reduce_size, inner_size;
    compute_outer_reduce_inner(x.sizes(), nd, dim,
                               outer_size, reduce_size, inner_size);

    int64_t out_sizes[kMaxDim];
    int out_ndim = compute_reduced_shape(x.sizes(), nd, dim, keepdim, out_sizes);
    int64_t out_numel = outer_size * inner_size;

    Tensor out = Tensor::empty(out_sizes, out_ndim, DType::Float32, stream);

    if (out_numel == 0 || reduce_size == 0) return out;

    constexpr int BLOCK = 256;

    if (inner_size == 1 && reduce_size >= 32) {
        // Last-dim: parallel block_reduce
        size_t smem = (BLOCK / 32) * sizeof(float);
        int grid = static_cast<int>(outer_size);
        reduce_dim_inner_kernel<Op><<<grid, BLOCK, smem, stream>>>(
            x.data<float>(), out.data<float>(),
            static_cast<int>(reduce_size), outer_size);
    } else {
        // General: one thread per output element
        int grid = internal::divup(static_cast<int>(out_numel), BLOCK);
        reduce_dim_general_kernel<Op><<<grid, BLOCK, 0, stream>>>(
            x.data<float>(), out.data<float>(),
            outer_size, reduce_size, inner_size);
    }

    // Apply post_scale if needed (for mean: 1/reduce_size)
    if (post_scale != 1.0f) {
        out.mul_(post_scale);
    }

    return out;
}

// =====================================================================
// ValIdx infrastructure for max/min with indices
// =====================================================================

struct ValIdx {
    float val;
    int   idx;
};

struct MaxValIdxOp {
    static __device__ __forceinline__ ValIdx identity() { return {-FLT_MAX, -1}; }
    static __device__ __forceinline__ ValIdx combine(ValIdx a, ValIdx b) {
        return (a.val >= b.val) ? a : b;
    }
};

struct MinValIdxOp {
    static __device__ __forceinline__ ValIdx identity() { return {FLT_MAX, -1}; }
    static __device__ __forceinline__ ValIdx combine(ValIdx a, ValIdx b) {
        return (a.val <= b.val) ? a : b;
    }
};

template <typename ValIdxOp>
__global__ void reduce_dim_validx_kernel(
    const float* __restrict__ x,
    float* __restrict__       out_vals,
    int64_t* __restrict__     out_idxs,
    int64_t                   outer_size,
    int64_t                   reduce_size,
    int64_t                   inner_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_out = outer_size * inner_size;
    if (idx >= total_out) return;

    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;
    int64_t base = outer * reduce_size * inner_size + inner;

    ValIdx acc = ValIdxOp::identity();
    for (int64_t r = 0; r < reduce_size; r++) {
        float v = x[base + r * inner_size];
        ValIdx cand = {v, static_cast<int>(r)};
        acc = ValIdxOp::combine(acc, cand);
    }

    out_vals[idx] = acc.val;
    out_idxs[idx] = static_cast<int64_t>(acc.idx);
}

template <typename ValIdxOp>
static std::pair<Tensor, Tensor> reduce_dim_validx_impl(
    const Tensor& input, int dim, bool keepdim)
{
    cudaStream_t stream = resolve_stream(nullptr);
    Tensor x = input.contiguous();
    int nd = x.ndim();
    dim = normalize_dim(dim, nd);

    int64_t outer_size, reduce_size, inner_size;
    compute_outer_reduce_inner(x.sizes(), nd, dim,
                               outer_size, reduce_size, inner_size);

    int64_t out_sizes[kMaxDim];
    int out_ndim = compute_reduced_shape(x.sizes(), nd, dim, keepdim, out_sizes);
    int64_t out_numel = outer_size * inner_size;

    Tensor values = Tensor::empty(out_sizes, out_ndim, DType::Float32, stream);
    Tensor indices = Tensor::empty(out_sizes, out_ndim, DType::Int64, stream);

    if (out_numel == 0 || reduce_size == 0)
        return {std::move(values), std::move(indices)};

    constexpr int BLOCK = 256;
    int grid = internal::divup(static_cast<int>(out_numel), BLOCK);

    reduce_dim_validx_kernel<ValIdxOp><<<grid, BLOCK, 0, stream>>>(
        x.data<float>(), values.data<float>(), indices.data<int64_t>(),
        outer_size, reduce_size, inner_size);

    return {std::move(values), std::move(indices)};
}

// =====================================================================
// Public: full-tensor reductions
// =====================================================================

Tensor sum(const Tensor& input) {
    cudaStream_t stream = resolve_stream(nullptr);
    Tensor x = input.contiguous();
    int64_t n = x.numel();
    Tensor out = Tensor::empty({1}, DType::Float32, stream);
    if (n == 0) { out.fill_(0.0f); return out; }
    reduce_impl<primitives::SumOp>(
        x.data<float>(), out.data<float>(), static_cast<int>(n),
        IdentityTransform{}, stream);
    if (should_record(input)) record_sum(out, input);
    return out;
}

Tensor mean(const Tensor& input) {
    cudaStream_t stream = resolve_stream(nullptr);
    Tensor x = input.contiguous();
    int64_t n = x.numel();
    Tensor out = Tensor::empty({1}, DType::Float32, stream);
    if (n == 0) { out.fill_(0.0f); return out; }
    reduce_impl<primitives::SumOp>(
        x.data<float>(), out.data<float>(), static_cast<int>(n),
        IdentityTransform{}, stream, 1.0f / static_cast<float>(n));
    if (should_record(input)) record_mean(out, input);
    return out;
}

Tensor prod(const Tensor& input) {
    cudaStream_t stream = resolve_stream(nullptr);
    Tensor x = input.contiguous();
    int64_t n = x.numel();
    Tensor out = Tensor::empty({1}, DType::Float32, stream);
    if (n == 0) { out.fill_(1.0f); return out; }
    reduce_impl<primitives::ProdOp>(
        x.data<float>(), out.data<float>(), static_cast<int>(n),
        IdentityTransform{}, stream);
    return out;
}

Tensor amax(const Tensor& input) {
    cudaStream_t stream = resolve_stream(nullptr);
    Tensor x = input.contiguous();
    int64_t n = x.numel();
    Tensor out = Tensor::empty({1}, DType::Float32, stream);
    if (n == 0) { out.fill_(-FLT_MAX); return out; }
    reduce_impl<primitives::MaxOp>(
        x.data<float>(), out.data<float>(), static_cast<int>(n),
        IdentityTransform{}, stream);
    return out;
}

Tensor amin(const Tensor& input) {
    cudaStream_t stream = resolve_stream(nullptr);
    Tensor x = input.contiguous();
    int64_t n = x.numel();
    Tensor out = Tensor::empty({1}, DType::Float32, stream);
    if (n == 0) { out.fill_(FLT_MAX); return out; }
    reduce_impl<primitives::MinOp>(
        x.data<float>(), out.data<float>(), static_cast<int>(n),
        IdentityTransform{}, stream);
    return out;
}

Tensor norm(const Tensor& input, float p) {
    cudaStream_t stream = resolve_stream(nullptr);
    Tensor x = input.contiguous();
    int64_t n = x.numel();
    Tensor out = Tensor::empty({1}, DType::Float32, stream);
    if (n == 0) { out.fill_(0.0f); return out; }

    if (p == 1.0f) {
        // L1 norm: sum of absolute values
        reduce_impl<primitives::SumOp>(
            x.data<float>(), out.data<float>(), static_cast<int>(n),
            AbsTransform{}, stream);
    } else if (p == 2.0f) {
        // L2 norm: sqrt(sum of squares)
        reduce_impl<primitives::SumOp>(
            x.data<float>(), out.data<float>(), static_cast<int>(n),
            SquareTransform{}, stream);
        scalar_post_sqrt_kernel<<<1, 1, 0, stream>>>(out.data<float>());
    } else if (::std::isinf(p)) {
        // L-inf norm: max absolute value
        reduce_impl<primitives::MaxOp>(
            x.data<float>(), out.data<float>(), static_cast<int>(n),
            AbsTransform{}, stream);
    } else {
        // General Lp: (sum |x|^p)^(1/p)
        reduce_impl<primitives::SumOp>(
            x.data<float>(), out.data<float>(), static_cast<int>(n),
            PowAbsTransform{p}, stream);
        scalar_post_pow_kernel<<<1, 1, 0, stream>>>(out.data<float>(), 1.0f / p);
    }
    return out;
}

// =====================================================================
// Public: dimension-wise reductions
// =====================================================================

Tensor sum(const Tensor& input, int dim, bool keepdim) {
    Tensor out = reduce_dim_impl<primitives::SumOp>(input, dim, keepdim);
    if (should_record(input)) record_sum_dim(out, input, dim, keepdim);
    return out;
}

Tensor mean(const Tensor& input, int dim, bool keepdim) {
    Tensor x = input.contiguous();
    int d = normalize_dim(dim, x.ndim());
    float scale = 1.0f / static_cast<float>(x.size(d));
    Tensor out = reduce_dim_impl<primitives::SumOp>(input, dim, keepdim, scale);
    if (should_record(input)) record_mean_dim(out, input, dim, keepdim);
    return out;
}

Tensor prod(const Tensor& input, int dim, bool keepdim) {
    return reduce_dim_impl<primitives::ProdOp>(input, dim, keepdim);
}

Tensor amax(const Tensor& input, int dim, bool keepdim) {
    return reduce_dim_impl<primitives::MaxOp>(input, dim, keepdim);
}

Tensor amin(const Tensor& input, int dim, bool keepdim) {
    return reduce_dim_impl<primitives::MinOp>(input, dim, keepdim);
}

std::pair<Tensor, Tensor> max(const Tensor& input, int dim, bool keepdim) {
    return reduce_dim_validx_impl<MaxValIdxOp>(input, dim, keepdim);
}

std::pair<Tensor, Tensor> min(const Tensor& input, int dim, bool keepdim) {
    return reduce_dim_validx_impl<MinValIdxOp>(input, dim, keepdim);
}

Tensor argmax(const Tensor& input, int dim) {
    auto [values, indices] = reduce_dim_validx_impl<MaxValIdxOp>(input, dim, false);
    return indices;
}

Tensor argmin(const Tensor& input, int dim) {
    auto [values, indices] = reduce_dim_validx_impl<MinValIdxOp>(input, dim, false);
    return indices;
}

// =====================================================================
// Public: statistical reductions (composed from primitives)
// =====================================================================

Tensor var(const Tensor& input, int dim, bool keepdim, int correction) {
    int d = normalize_dim(dim, input.ndim());
    int64_t N = input.size(d);
    Tensor m = forge::mean(input, dim, /*keepdim=*/true);
    Tensor diff = forge::sub(input, m);
    Tensor sq = forge::square(diff);
    Tensor s = forge::sum(sq, dim, keepdim);
    float denom = static_cast<float>(N - correction);
    if (denom <= 0.0f) denom = 1.0f; // guard against N <= correction
    return forge::div(s, denom);
}

Tensor std_dev(const Tensor& input, int dim, bool keepdim, int correction) {
    return forge::sqrt(forge::var(input, dim, keepdim, correction));
}

Tensor logsumexp(const Tensor& input, int dim, bool keepdim) {
    Tensor m = forge::amax(input, dim, /*keepdim=*/true);
    Tensor shifted = forge::sub(input, m);
    Tensor exp_shifted = forge::exp(shifted);
    Tensor s = forge::sum(exp_shifted, dim, keepdim);
    Tensor log_s = forge::log(s);
    // Squeeze m to match log_s shape
    Tensor m_out = forge::amax(input, dim, keepdim);
    return forge::add(log_s, m_out);
}

} // namespace forge
