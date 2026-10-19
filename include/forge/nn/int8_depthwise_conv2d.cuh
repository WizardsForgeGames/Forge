// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// INT8 Depthwise Conv2d forward (inference only).
//
// Specialized kernel for depthwise convolution (groups = C_in = C_out).
// Uses direct sliding window (no im2col/GEMM — K=9 for 3×3 is too small).
// INT8 weights with per-channel FP16 scale, FP16 input/output.
//
// Data layout: NCHW.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace forge { namespace nn {

/// INT8 Depthwise Conv2d forward (inference only).
///
/// True depthwise: groups = C = C_in = C_out.
/// One thread per output pixel, direct 3×3 sliding window.
/// No workspace needed.
///
/// Input:  FP16 [N, C, H, W]
/// Weight: INT8 [C, kH*kW] (per-channel pre-quantized, row-major)
/// Output: FP16 [N, C, H_out, W_out]
void int8_depthwise_conv2d_forward(
    const __half* input,       // [N, C, H, W]
    const int8_t* weight,      // [C, kH*kW]
    const __half* w_scale,     // [C] per-channel weight scale
    const __half* bias,        // [C] or nullptr
    __half*       output,      // [N, C, H_out, W_out]
    int N, int C, int H, int W,
    int kH, int kW,
    int padH, int padW,
    int strideH  = 1,
    int strideW  = 1,
    int dilH     = 1,
    int dilW     = 1,
    cudaStream_t stream = nullptr);

}} // namespace forge::nn
