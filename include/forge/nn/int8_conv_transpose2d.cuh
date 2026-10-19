// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// INT8 ConvTranspose2d forward (inference only), specialized for k=2 s=2.
//
// FSR4's decoder uses ConvTranspose2d(k=2, s=2) for 2× upsampling.
// Since stride == kernel, each output pixel comes from exactly one input
// pixel — no overlap. This enables a clean GEMM-based approach:
//
//   1. Transpose input [C_in, H*W] → [H*W, C_in]
//   2. Quantize → INT8 + scales
//   3. INT8 GEMM: [H*W, C_in] @ combined_weight[4*C_out, C_in]^T → [H*W, 4*C_out]
//      (combined_weight packs all 4 kernel positions into one matrix)
//   4. Scatter + bias: deinterleave to output[C_out, 2*H, 2*W]
//
// Data layout: NCHW.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace forge { namespace nn {

/// Compute workspace size (bytes) for int8_conv_transpose2d_k2s2_forward.
size_t int8_conv_transpose2d_k2s2_workspace_size(
    int C_in, int H, int W, int C_out);

/// INT8 ConvTranspose2d forward, specialized for kernel=2, stride=2.
///
/// Input:  FP16 [N, C_in, H, W]
/// Weight: INT8 [4*C_out, C_in] (repacked: row co*4+kh*2+kw = weight[ci, co, kh, kw])
/// Output: FP16 [N, C_out, 2*H, 2*W]
///
/// w_scale is [C_out] (per-output-channel), internally replicated to [4*C_out].
/// bias is [C_out] or nullptr, added during scatter (not in GEMM).
void int8_conv_transpose2d_k2s2_forward(
    const __half* input,       // [N, C_in, H, W]
    const int8_t* weight,      // [4*C_out, C_in] (repacked)
    const __half* w_scale,     // [C_out] per-output-channel
    const __half* bias,        // [C_out] or nullptr
    __half*       output,      // [N, C_out, 2*H, 2*W]
    int N, int C_in, int H, int W,
    int C_out,
    void*         workspace,
    cudaStream_t  stream = nullptr);

}} // namespace forge::nn
