// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/bias.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace nn {

__global__ void bias_grad_accumulate_kernel(
    const float* __restrict__ d_upstream,
    float* __restrict__       d_bias,
    int                       N,
    int                       B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sum = 0.0f;
    for (int b = 0; b < B; b++) {
        sum += d_upstream[i * B + b];
    }
    d_bias[i] += sum;
}

void bias_grad_accumulate(
    const float* d_upstream,
    float*       d_bias,
    int          N,
    int          B,
    cudaStream_t stream)
{
    if (N <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(N, block);
    bias_grad_accumulate_kernel<<<grid, block, 0, stream>>>(d_upstream, d_bias, N, B);
}

}} // namespace forge::nn
