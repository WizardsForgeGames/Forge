// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Conv2d: 2D convolution via im2col + GEMM.
// Supports padding, stride, dilation, and grouped convolution.
//
// Weight layout: W[C_out, C_in/groups, kH, kW] row-major (PyTorch convention).
// Data layout:   NCHW — input[N, C_in, H, W], output[N, C_out, H_out, W_out].
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

/// Forward: output[N, C_out, H_out, W_out] = conv2d(input, weight) + bias
///
/// H_out = (H + 2*padH - dilH*(kH-1) - 1) / strideH + 1
/// W_out = (W + 2*padW - dilW*(kW-1) - 1) / strideW + 1
///
/// bias may be nullptr (no bias).
void conv2d_forward(
    const float* input,     // [N, C_in, H, W]
    const float* weight,    // [C_out, C_in/groups, kH, kW]
    const float* bias,      // [C_out] or nullptr
    float*       output,    // [N, C_out, H_out, W_out]
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH     = 1,
    int dilW     = 1,
    int groups   = 1,
    cudaStream_t stream = nullptr);

/// Backward data: d_input[N, C_in, H, W] from d_output[N, C_out, H_out, W_out]
void conv2d_backward_data(
    const float* d_output,  // [N, C_out, H_out, W_out]
    const float* weight,    // [C_out, C_in/groups, kH, kW]
    float*       d_input,   // [N, C_in, H, W]
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH     = 1,
    int dilW     = 1,
    int groups   = 1,
    cudaStream_t stream = nullptr);

/// Backward weight: d_weight[C_out, C_in/groups, kH, kW] from d_output and input.
/// Optionally computes d_bias[C_out] if d_bias != nullptr.
/// beta controls accumulation: 0 = overwrite, 1 = accumulate.
void conv2d_backward_weight(
    const float* d_output,  // [N, C_out, H_out, W_out]
    const float* input,     // [N, C_in, H, W]
    float*       d_weight,  // [C_out, C_in/groups, kH, kW]
    float*       d_bias,    // [C_out] or nullptr
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH     = 1,
    int dilW     = 1,
    int groups   = 1,
    float beta   = 0.0f,
    cudaStream_t stream = nullptr);

}} // namespace forge::nn

// Tensor-level conv2d with autograd support (declared in forge namespace)
namespace forge {
class Tensor;

/// Conv2d operating on Tensor objects with autograd graph recording.
/// input: [N, C_in, H, W], weight: [C_out, C_in/groups, kH, kW], bias: [C_out] (optional)
Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int padH, int padW,
              int strideH, int strideW,
              int dilH = 1, int dilW = 1,
              int groups = 1);

/// Convenience overload without bias
Tensor conv2d(const Tensor& input, const Tensor& weight,
              int padH, int padW,
              int strideH, int strideW,
              int dilH = 1, int dilW = 1,
              int groups = 1);

} // namespace forge
