// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Conv3d: 3D convolution via vol2col + GEMM.
// Supports padding, stride, dilation, and grouped convolution.
//
// Weight layout: W[C_out, C_in/groups, kD, kH, kW] row-major.
// Data layout:   NCDHW — input[N, C_in, D, H, W], output[N, C_out, D_out, H_out, W_out].
//
// For training the ThermalUNet (32^3 grid, 3D convolutions).
// Uses vol2col to unroll 3D patches into a matrix, then Forge's GEMM.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

/// Forward: output = conv3d(input, weight) + bias
///
/// D_out = (D + 2*padD - dilD*(kD-1) - 1) / strideD + 1
/// (same for H_out, W_out)
///
/// bias may be nullptr (no bias).
///
/// workspace: pre-allocated buffer for vol2col.
///   Size: C_in_per_group * kD * kH * kW * D_out * H_out * W_out floats.
///   If nullptr, will allocate internally (slower due to sync alloc).
void conv3d_forward(
    const float* input,     // [N, C_in, D, H, W]
    const float* weight,    // [C_out, C_in/groups, kD, kH, kW]
    const float* bias,      // [C_out] or nullptr
    float*       output,    // [N, C_out, D_out, H_out, W_out]
    float*       workspace, // vol2col buffer or nullptr
    int N, int C_in, int D, int H, int W,
    int C_out, int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD     = 1,
    int dilH     = 1,
    int dilW     = 1,
    int groups   = 1,
    cudaStream_t stream = nullptr);

/// Backward data: d_input from d_output and weight.
void conv3d_backward_data(
    const float* d_output,  // [N, C_out, D_out, H_out, W_out]
    const float* weight,    // [C_out, C_in/groups, kD, kH, kW]
    float*       d_input,   // [N, C_in, D, H, W]
    float*       workspace, // vol2col buffer or nullptr
    int N, int C_in, int D, int H, int W,
    int C_out, int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD     = 1,
    int dilH     = 1,
    int dilW     = 1,
    int groups   = 1,
    cudaStream_t stream = nullptr);

/// Backward weight: d_weight from d_output and input.
/// Optionally computes d_bias if d_bias != nullptr.
/// beta: 0 = overwrite, 1 = accumulate.
void conv3d_backward_weight(
    const float* d_output,  // [N, C_out, D_out, H_out, W_out]
    const float* input,     // [N, C_in, D, H, W]
    float*       d_weight,  // [C_out, C_in/groups, kD, kH, kW]
    float*       d_bias,    // [C_out] or nullptr
    float*       workspace, // vol2col buffer or nullptr
    int N, int C_in, int D, int H, int W,
    int C_out, int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD     = 1,
    int dilH     = 1,
    int dilW     = 1,
    int groups   = 1,
    float beta   = 0.0f,
    cudaStream_t stream = nullptr);

/// Compute vol2col workspace size in floats for a single sample & group.
inline int conv3d_workspace_size(
    int C_in_per_group, int kD, int kH, int kW,
    int D_out, int H_out, int W_out)
{
    return C_in_per_group * kD * kH * kW * D_out * H_out * W_out;
}

}} // namespace forge::nn
