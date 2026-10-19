// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Implicit GEMM Conv3D — fuses vol2col addressing into the GEMM.
//
// Instead of materializing a [K, N] column matrix (vol2col) and then
// doing a separate GEMM, the implicit GEMM reads input patches on-the-fly
// during the matrix multiply. This eliminates the intermediate buffer and
// its memory traffic, which is the main bottleneck of vol2col+GEMM.
//
// Algorithm:
//   output[cout, d_out, h_out, w_out] = sum_{cin, kd, kh, kw}
//     weight[cout, cin, kd, kh, kw] * input[cin, d_out*sD+kd*dD-pD, ...]
//
// Tiled as a GEMM:  C[M, N] = A[M, K] * B[K, N]
//   M = C_out          (output channels — weight rows)
//   K = C_in * kD * kH * kW    (filter volume — reduction dim)
//   N = D_out * H_out * W_out  (output spatial — output columns)
//
//   A = weight[C_out, K]       (stored explicitly)
//   B = "virtual" col matrix   (read from input on-the-fly via addressing)
//
// Designed for ThermalUNet conv sizes: k=3x3x3 pad=1 s=1, 32^3/16^3/8^3 grids.
// Also supports k=1x1x1, k=2x2x2 s=2, arbitrary padding/stride/dilation.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

/// Implicit GEMM Conv3D forward.
/// No workspace needed — all computation done in-register + shared memory.
void conv3d_implicit_forward(
    const float* input,     // [N, C_in, D, H, W]
    const float* weight,    // [C_out, C_in/groups, kD, kH, kW]
    const float* bias,      // [C_out] or nullptr
    float*       output,    // [N, C_out, D_out, H_out, W_out]
    int N, int C_in, int D, int H, int W,
    int C_out, int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD     = 1,
    int dilH     = 1,
    int dilW     = 1,
    int groups   = 1,
    cudaStream_t stream = nullptr);

/// Implicit GEMM Conv3D backward data (d_input from d_output and weight).
void conv3d_implicit_backward_data(
    const float* d_output,  // [N, C_out, D_out, H_out, W_out]
    const float* weight,    // [C_out, C_in/groups, kD, kH, kW]
    float*       d_input,   // [N, C_in, D, H, W]
    int N, int C_in, int D, int H, int W,
    int C_out, int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD     = 1,
    int dilH     = 1,
    int dilW     = 1,
    int groups   = 1,
    cudaStream_t stream = nullptr);

}} // namespace forge::nn
