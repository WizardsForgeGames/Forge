// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/reduce.cuh>
#include <forge/primitives/reduce.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace nn {

// ============================================================
//  Transform functors (applied per-element before reduction)
// ============================================================

struct IdentityTransform {
    __device__ __forceinline__ float operator()(float x) const { return x; }
};

struct SquareTransform {
    __device__ __forceinline__ float operator()(float x) const { return x * x; }
};

struct AbsTransform {
    __device__ __forceinline__ float operator()(float x) const { return fabsf(x); }
};

// ============================================================
//  Generic two-pass scalar reduction
// ============================================================

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

    // Adaptive grid: target 2 waves, but don't exceed natural grid
    int max_grid = internal::get_sm_count() * 2;
    int natural_grid = internal::divup(n, BLOCK);
    int grid = (natural_grid < max_grid) ? natural_grid : max_grid;

    size_t smem = (BLOCK / 32) * sizeof(float);

    if (grid == 1) {
        reduce_partial_kernel<Op><<<1, BLOCK, smem, stream>>>(
            x, out, n, xform, post_scale);
        return;
    }

    // Two-pass: partials → final
    float* d_partial = nullptr;
    cudaMalloc(&d_partial, grid * sizeof(float));

    reduce_partial_kernel<Op><<<grid, BLOCK, smem, stream>>>(
        x, d_partial, n, xform, 1.0f);

    int final_block = (grid <= 256) ? 256 : 1024;
    size_t final_smem = (final_block / 32) * sizeof(float);
    reduce_final_kernel<Op><<<1, final_block, final_smem, stream>>>(
        d_partial, out, grid, post_scale);

    cudaFreeAsync(d_partial, stream);
}

// ============================================================
//  Public API
// ============================================================

void reduce_sum(
    const float* x,
    float*       out,
    int          n,
    cudaStream_t stream)
{
    reduce_impl<primitives::SumOp>(x, out, n, IdentityTransform{}, stream);
}

void reduce_sumsq(
    const float* x,
    float*       out,
    int          n,
    cudaStream_t stream)
{
    reduce_impl<primitives::SumOp>(x, out, n, SquareTransform{}, stream);
}

void reduce_absmax(
    const float* x,
    float*       out,
    int          n,
    cudaStream_t stream)
{
    reduce_impl<primitives::MaxOp>(x, out, n, AbsTransform{}, stream);
}

void reduce_mean(
    const float* x,
    float*       out,
    int          n,
    cudaStream_t stream)
{
    reduce_impl<primitives::SumOp>(
        x, out, n, IdentityTransform{}, stream, 1.0f / (float)n);
}

// ============================================================
//  reduce_sum_dim0: [D0, D1] → [D1]
// ============================================================

static __global__ void reduce_sum_dim0_kernel(
    const float* __restrict__ x,
    float* __restrict__       out,
    int                       D0,
    int                       D1)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D1) return;

    float sum = 0.0f;
    for (int i = 0; i < D0; i++)
        sum += x[i * D1 + j];

    out[j] = sum;
}

void reduce_sum_dim0(
    const float* x,
    float*       out,
    int          D0,
    int          D1,
    cudaStream_t stream)
{
    if (D0 <= 0 || D1 <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(D1, block);
    reduce_sum_dim0_kernel<<<grid, block, 0, stream>>>(x, out, D0, D1);
}

// ============================================================
//  reduce_mean_dim0: [D0, D1] → [D1], with 1/D0 scaling
// ============================================================

static __global__ void reduce_mean_dim0_kernel(
    const float* __restrict__ x,
    float* __restrict__       out,
    int                       D0,
    int                       D1,
    float                     inv_D0)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D1) return;

    float sum = 0.0f;
    for (int i = 0; i < D0; i++)
        sum += x[i * D1 + j];

    out[j] = sum * inv_D0;
}

void reduce_mean_dim0(
    const float* x,
    float*       out,
    int          D0,
    int          D1,
    cudaStream_t stream)
{
    if (D0 <= 0 || D1 <= 0) return;
    float inv_D0 = 1.0f / static_cast<float>(D0);
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(D1, block);
    reduce_mean_dim0_kernel<<<grid, block, 0, stream>>>(x, out, D0, D1, inv_D0);
}

}} // namespace forge::nn
