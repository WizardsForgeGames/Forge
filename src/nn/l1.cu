// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/l1.cuh>
#include <forge/nn/reduce.cuh>
#include <forge/primitives/reduce.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace nn {

// ============================================================
//  L1 forward
// ============================================================

static __global__ void l1_forward_small_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__       loss,
    int                       n)
{
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        sum += fabsf(pred[i] - target[i]);
    }

    sum = primitives::block_reduce<primitives::SumOp>(sum, smem);

    if (tid == 0)
        loss[0] = sum / (float)n;
}

static __global__ void abs_diff_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__       diff_abs,
    int                       n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    diff_abs[idx] = fabsf(pred[idx] - target[idx]);
}

void l1_forward(
    const float* pred,
    const float* target,
    float*       loss,
    int          n,
    cudaStream_t stream)
{
    if (n <= 0) return;

    // Small path: single kernel
    if (n <= 65536) {
        int block = (n <= 256) ? 256 : 1024;
        l1_forward_small_kernel<<<1, block, (block / 32) * sizeof(float), stream>>>(
            pred, target, loss, n);
        return;
    }

    // Large path: abs_diff → reduce_mean
    float* d_diff = nullptr;
    cudaMalloc(&d_diff, n * sizeof(float));

    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(n, block);
    abs_diff_kernel<<<grid, block, 0, stream>>>(pred, target, d_diff, n);

    reduce_mean(d_diff, loss, n, stream);

    cudaFreeAsync(d_diff, stream);
}

// ============================================================
//  L1 backward
// ============================================================

static __global__ void l1_backward_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__       d_pred,
    int                       n,
    float                     coeff)  // scale / n
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float diff = pred[idx] - target[idx];
    // sign(diff): +1 if diff > 0, -1 if diff < 0, 0 if diff == 0
    d_pred[idx] = coeff * copysignf(1.0f, diff) * (diff != 0.0f);
}

void l1_backward(
    const float* pred,
    const float* target,
    float*       d_pred,
    int          n,
    float        scale,
    cudaStream_t stream)
{
    if (n <= 0) return;
    float coeff = scale / (float)n;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(n, block);
    l1_backward_kernel<<<grid, block, 0, stream>>>(pred, target, d_pred, n, coeff);
}

}} // namespace forge::nn
