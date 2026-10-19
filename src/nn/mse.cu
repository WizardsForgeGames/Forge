// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/mse.cuh>
#include <forge/nn/reduce.cuh>
#include <forge/primitives/reduce.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace nn {

// ============================================================
//  MSE forward
// ============================================================
// Strategy: compute squared differences into a temp buffer,
// then reduce_sum and divide by n.
// For small n we fuse into a single kernel.

static __global__ void mse_forward_small_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__       loss,
    int                       n)
{
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        float d = pred[i] - target[i];
        sum += d * d;
    }

    sum = primitives::block_reduce<primitives::SumOp>(sum, smem);

    if (tid == 0)
        loss[0] = sum / (float)n;
}

static __global__ void squared_diff_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__       diff_sq,
    int                       n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float d = pred[idx] - target[idx];
    diff_sq[idx] = d * d;
}

void mse_forward(
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
        mse_forward_small_kernel<<<1, block, (block / 32) * sizeof(float), stream>>>(
            pred, target, loss, n);
        return;
    }

    // Large path: squared_diff → reduce_mean (fused scale)
    float* d_diff = nullptr;
    cudaMalloc(&d_diff, n * sizeof(float));

    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(n, block);
    squared_diff_kernel<<<grid, block, 0, stream>>>(pred, target, d_diff, n);

    reduce_mean(d_diff, loss, n, stream);

    cudaFreeAsync(d_diff, stream);
}

// ============================================================
//  MSE backward
// ============================================================

static __global__ void mse_backward_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__       d_pred,
    int                       n,
    float                     coeff)  // scale * 2/n
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_pred[idx] = coeff * (pred[idx] - target[idx]);
}

void mse_backward(
    const float* pred,
    const float* target,
    float*       d_pred,
    int          n,
    float        scale,
    cudaStream_t stream)
{
    if (n <= 0) return;
    float coeff = scale * 2.0f / (float)n;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(n, block);
    mse_backward_kernel<<<grid, block, 0, stream>>>(pred, target, d_pred, n, coeff);
}

}} // namespace forge::nn
