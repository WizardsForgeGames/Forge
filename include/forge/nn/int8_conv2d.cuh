// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// INT8 Conv2d forward (inference only) via im2col + unified INT8 GEMM.
//
// Uses per-row activation quantization (im2col patches) and per-channel
// weight quantization with fused dequant + bias in the GEMM epilogue.
//
// Data layout: NCHW (input/output), weights are [C_out, C_in*kH*kW] row-major.
// Caller provides workspace buffer (no internal allocation).
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace forge { namespace nn {

/// Compute workspace size (bytes) needed for int8_conv2d_forward.
///
/// Workspace holds: FP16 im2col buffer, INT8 quantized buffer,
/// per-row activation scales, and GEMM output staging buffer.
size_t int8_conv2d_workspace_size(
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH  = 1,
    int dilW  = 1);

/// INT8 Conv2d forward (inference only).
///
/// Performs: im2col → quantize → INT8 GEMM (dequant+bias) → NCHW transpose.
/// Automatically dispatches to wmma (sm_80+) or dp4a (sm_61+).
///
/// Input:  FP16 [N, C_in, H, W]   (NCHW)
/// Weight: INT8 [C_out, C_in*kH*kW] (pre-quantized, row-major)
/// Output: FP16 [N, C_out, H_out, W_out] (NCHW)
///
/// workspace must be at least int8_conv2d_workspace_size() bytes.
void int8_conv2d_forward(
    const __half* input,       // [N, C_in, H, W]
    const int8_t* weight,      // [C_out, C_in*kH*kW]
    const __half* w_scale,     // [C_out]
    const __half* bias,        // [C_out] or nullptr
    __half*       output,      // [N, C_out, H_out, W_out]
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW,
    void*         workspace,
    cudaStream_t  stream = nullptr);

}} // namespace forge::nn
