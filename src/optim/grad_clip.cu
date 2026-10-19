// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/optim/grad_clip.cuh>
#include <forge/primitives/reduce.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace optim {

__global__ void grad_norm_sq_kernel(
    const float* __restrict__ grad,
    int                       n,
    float* __restrict__       out_norm_sq)
{
    extern __shared__ float smem[];
    float val = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        val += grad[i] * grad[i];

    val = primitives::block_reduce<primitives::SumOp>(val, smem);

    if (threadIdx.x == 0) atomicAdd(out_norm_sq, val);
}

__global__ void grad_scale_kernel(
    float* __restrict__ grad,
    int                 n,
    float               scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad[idx] *= scale;
}

__global__ void compute_clip_scale_kernel(
    float* __restrict__       out_scale,
    const float* __restrict__ norm_sq,
    float                     max_grad_norm,
    float                     inv_B)
{
    float scale = inv_B;
    if (max_grad_norm > 0.0f) {
        float raw_norm = sqrtf(*norm_sq);
        float eff_norm = raw_norm * inv_B;
        if (eff_norm > max_grad_norm) {
            scale = max_grad_norm / raw_norm;
        }
    }
    *out_scale = scale;
}

// --- Launchers ---

void grad_norm_sq(
    const float* grad,
    int          n,
    float*       out_norm_sq,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int block = internal::kDefaultBlockSize;
    int max_grid = internal::get_sm_count() * 2;
    int grid = internal::divup(n, block);
    if (grid > max_grid) grid = max_grid;
    size_t smem = (block / 32) * sizeof(float);
    grad_norm_sq_kernel<<<grid, block, smem, stream>>>(grad, n, out_norm_sq);
}

void grad_scale(
    float*       grad,
    int          n,
    float        scale,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(n, block);
    grad_scale_kernel<<<grid, block, 0, stream>>>(grad, n, scale);
}

void compute_clip_scale(
    float*       out_scale,
    const float* norm_sq,
    float        max_grad_norm,
    float        inv_B,
    cudaStream_t stream)
{
    compute_clip_scale_kernel<<<1, 1, 0, stream>>>(out_scale, norm_sq, max_grad_norm, inv_B);
}

}} // namespace forge::optim
