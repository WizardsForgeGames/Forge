// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Bias gradient accumulation: sum over batch dimension.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

/// Accumulate bias gradient: d_bias[i] += sum_b d_upstream[i * B + b]
/// where d_upstream is [N, B] col-major.
__global__ void bias_grad_accumulate_kernel(
    const float* __restrict__ d_upstream,
    float* __restrict__       d_bias,
    int                       N,
    int                       B);

void bias_grad_accumulate(
    const float* d_upstream,
    float*       d_bias,
    int          N,
    int          B,
    cudaStream_t stream = nullptr);

}} // namespace forge::nn
