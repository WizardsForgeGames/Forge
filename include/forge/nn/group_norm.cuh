// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Fused GroupNorm + GELU for 3D volumes (NCHW layout).
//
// Forward: y = GELU(gamma * (x - mean) / sqrt(var + eps) + beta)
// Backward: dx, d_gamma, d_beta from d_output, using recomputed x_hat.
//
// One block per (n, group) pair. Mean/rstd saved for backward.
// Designed for the ThermalUNet ResBlock pattern: GroupNorm → GELU → Conv3d.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

/// Fused GroupNorm + GELU forward.
///
/// Input:  x      [N, C, D, H, W]  (NCHW, contiguous)
/// Output: y      [N, C, D, H, W]
///         mean   [N, G]  — saved for backward
///         rstd   [N, G]  — saved for backward (1/sqrt(var+eps))
///
/// gamma, beta: [C] affine parameters (per-channel).
void group_norm_gelu_forward(
    const float* x,         // [N, C, D, H, W]
    const float* gamma,     // [C]
    const float* beta,      // [C]
    float*       y,         // [N, C, D, H, W]
    float*       mean,      // [N, G]
    float*       rstd,      // [N, G]
    int N, int C, int spatial,  // spatial = D * H * W
    int G,                      // number of groups
    float eps = 1e-5f,
    cudaStream_t stream = nullptr);

/// Fused GroupNorm + GELU backward.
///
/// Computes d_input, d_gamma, d_beta from d_output.
/// Recomputes GroupNorm output on-the-fly (avoids storing activations).
///
/// d_gamma, d_beta are atomically accumulated (caller must zero them first).
void group_norm_gelu_backward(
    const float* d_output,  // [N, C, D, H, W]
    const float* x,         // [N, C, D, H, W] — original input
    const float* gamma,     // [C]
    const float* beta,      // [C]
    const float* mean,      // [N, G]
    const float* rstd,      // [N, G]
    float*       d_input,   // [N, C, D, H, W]
    float*       d_gamma,   // [C] — atomically accumulated
    float*       d_beta,    // [C] — atomically accumulated
    int N, int C, int spatial,
    int G,
    cudaStream_t stream = nullptr);

// =====================================================================
// Standalone GroupNorm (no GELU fusion)
// =====================================================================

/// GroupNorm forward (no activation).
///
/// Input:  x      [N, C, ...spatial...]  (NCHW, contiguous)
/// Output: y      [N, C, ...spatial...]
///         mean   [N, G]  — saved for backward
///         rstd   [N, G]  — saved for backward (1/sqrt(var+eps))
///
/// gamma, beta: [C] affine parameters (per-channel).
void group_norm_forward(
    const float* x,         // [N, C, spatial]
    const float* gamma,     // [C]
    const float* beta,      // [C]
    float*       y,         // [N, C, spatial]
    float*       mean,      // [N, G]
    float*       rstd,      // [N, G]
    int N, int C, int spatial,
    int G,
    float eps = 1e-5f,
    cudaStream_t stream = nullptr);

/// GroupNorm backward (no activation).
///
/// Computes d_input, d_gamma, d_beta from d_output.
/// d_gamma, d_beta are atomically accumulated (caller must zero them first).
void group_norm_backward(
    const float* d_output,  // [N, C, spatial]
    const float* x,         // [N, C, spatial] — original input
    const float* gamma,     // [C]
    const float* mean,      // [N, G]
    const float* rstd,      // [N, G]
    float*       d_input,   // [N, C, spatial]
    float*       d_gamma,   // [C] — atomically accumulated
    float*       d_beta,    // [C] — atomically accumulated
    int N, int C, int spatial,
    int G,
    cudaStream_t stream = nullptr);

}} // namespace forge::nn
