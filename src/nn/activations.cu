// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/activations.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace nn {

__global__ void leaky_relu_backward_kernel(
    const float* __restrict__ z,
    const float* __restrict__ da,
    float* __restrict__       dz,
    int                       n,
    float                     slope)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dz[idx] = (z[idx] > 0.0f) ? da[idx] : slope * da[idx];
}

void leaky_relu_backward(
    const float* z,
    const float* da,
    float*       dz,
    int          n,
    float        slope,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(n, block);
    leaky_relu_backward_kernel<<<grid, block, 0, stream>>>(z, da, dz, n, slope);
}

__global__ void tanh_backward_kernel(
    const float* __restrict__ da,
    const float* __restrict__ a_out,
    float* __restrict__       dz,
    int                       n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float a = a_out[idx];
    dz[idx] = da[idx] * (1.0f - a * a);
}

void tanh_backward(
    const float* da,
    const float* a_out,
    float*       dz,
    int          n,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(n, block);
    tanh_backward_kernel<<<grid, block, 0, stream>>>(da, a_out, dz, n);
}

__global__ void silu_backward_kernel(
    const float* __restrict__ z,
    const float* __restrict__ da,
    float* __restrict__       dz,
    int                       n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float zi = z[idx];
    float s = 1.0f / (1.0f + expf(-zi));
    dz[idx] = da[idx] * (s + zi * s * (1.0f - s));
}

void silu_backward(
    const float* z,
    const float* da,
    float*       dz,
    int          n,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(n, block);
    silu_backward_kernel<<<grid, block, 0, stream>>>(z, da, dz, n);
}

__global__ void celu_backward_kernel(
    const float* __restrict__ z,
    const float* __restrict__ da,
    float* __restrict__       dz,
    int                       n,
    float                     alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float zi = z[idx];
    dz[idx] = (zi > 0.0f) ? da[idx] : da[idx] * expf(zi / alpha);
}

void celu_backward(
    const float* z,
    const float* da,
    float*       dz,
    int          n,
    float        alpha,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(n, block);
    celu_backward_kernel<<<grid, block, 0, stream>>>(z, da, dz, n, alpha);
}

}} // namespace forge::nn
