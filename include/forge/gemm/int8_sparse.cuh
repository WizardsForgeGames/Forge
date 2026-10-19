// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Sparse INT8 Tensor Core GEMM via PTX mma.sp instruction.
// 2:4 structured sparsity for 2x throughput over dense INT8.
// 263 TOPS on RTX 4090.
//
// W_comp: (N_out, K/2) compressed weight (2:4 structured sparsity)
// x_int8: (M, K) quantized activation
// w_meta: (N_out, K/32) packed uint32 metadata
//
// Requires: K % 64 == 0, N_out % 16 == 0.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace forge { namespace gemm {

/// Sparse INT8 TC GEMM with fused dequant + bias.
/// Automatically selects standard vs. warp-specialized kernel variant.
/// Supports split-K for small M / large K scenarios.
void sparse_int8_gemm(
    const int8_t*   W_comp,     // (N_out, K/2) device
    const int8_t*   x_int8,     // (M, K) device
    const uint32_t* w_meta,     // (N_out, K/32) device
    const float*    act_scale,  // (M,) device
    const __half*   w_scale,    // (N_out,) device
    const __half*   bias,       // (N_out,) device or nullptr
    __half*         output,     // (M, N_out) device
    int             M,
    int             K,
    int             N_out,
    float*          splitk_workspace = nullptr,
    int             split_k = 1,
    cudaStream_t    stream  = nullptr);

/// Convenience: quant + sparse GEMM + dequant.
void sparse_int8_linear(
    const __half*   x,           // (M, K) fp16 device
    const int8_t*   W_comp,      // (N_out, K/2) int8 device
    const uint32_t* w_meta,      // (N_out, K/32) device
    const __half*   w_scale,     // (N_out,) fp16 device
    const __half*   bias,        // (N_out,) fp16 device or nullptr
    __half*         output,      // (M, N_out) fp16 device
    int             M,
    int             K,
    int             N_out,
    int8_t*         scratch_x_int8,
    float*          scratch_x_scale,
    float*          splitk_workspace = nullptr,
    int             split_k = 1,
    cudaStream_t    stream  = nullptr);

/// Split-K reduce: sum fp32 workspace partials -> fp16 output + bias.
void splitk_reduce(
    const float*  workspace,    // (M, N_out) fp32 device
    const __half* bias,         // (N_out,) device or nullptr
    __half*       output,       // (M, N_out) fp16 device
    int           M,
    int           N_out,
    cudaStream_t  stream = nullptr);

}} // namespace forge::gemm
