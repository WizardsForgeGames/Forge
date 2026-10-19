// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// FP32 Linear layer: forward, backward-data, backward-weight.
// Thin wrappers around forge::gemm::fp32_gemm with the standard
// neural-network transpose conventions.
//
// Weight layout: W[N_out, N_in] row-major  (each row is one output neuron).
// Bias layout:   b[N_out]                  (one scalar per output neuron).
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

/// Forward: output[B, N] = input[B, K] @ W[N, K]^T + bias[N]
///
/// Row-major.  B = batch size, N = output features, K = input features.
/// bias may be nullptr (no bias).
void linear_forward(
    const float* input,     // [B, K]
    const float* weight,    // [N, K]
    const float* bias,      // [N] or nullptr
    float*       output,    // [B, N]
    int          B,
    int          N,
    int          K,
    cudaStream_t stream = nullptr);

/// Backward data: d_input[B, K] = d_output[B, N] @ W[N, K]
///
/// Gradient w.r.t. the input, given upstream gradient d_output.
void linear_backward_data(
    const float* d_output,  // [B, N]
    const float* weight,    // [N, K]
    float*       d_input,   // [B, K]
    int          B,
    int          N,
    int          K,
    cudaStream_t stream = nullptr);

/// Backward weight: d_weight[N, K] = d_output[B, N]^T @ input[B, K]
///
/// beta controls accumulation: 0 = overwrite, 1 = accumulate.
void linear_backward_weight(
    const float* d_output,  // [B, N]
    const float* input,     // [B, K]
    float*       d_weight,  // [N, K]
    int          B,
    int          N,
    int          K,
    float        beta   = 0.0f,
    cudaStream_t stream = nullptr);

}} // namespace forge::nn
