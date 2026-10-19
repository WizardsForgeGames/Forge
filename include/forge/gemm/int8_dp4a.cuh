// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Dense INT8 dp4a GEMM with fused dequant + bias.
// Uses __dp4a() intrinsics for sm_61+ (Pascal, Turing without TC).
//
// Y[m,n] = sum_k(A[m,k] * B[n,k]) * act_scale[m] * w_scale[n] + bias[n]
//
// A: (M, K) row-major int8
// B: (N, K) row-major int8 (note: N-K layout, same as wmma variant)
// Output: (M, N) fp16
//
// Requires: sm_61+ (Pascal dp4a). No alignment constraint on K.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace forge { namespace gemm {

/// Dense INT8 dp4a GEMM with fused dequant + bias.
/// For GPUs without tensor cores (sm_61 Pascal, sm_75 Turing non-TC).
/// API mirrors int8_gemm_dequant() exactly.
void int8_dp4a_gemm_dequant(
    const int8_t* A,            // (M, K) device
    const int8_t* B,            // (N, K) device
    const float*  act_scale,    // (M,) device
    const __half* w_scale,      // (N,) device
    const __half* bias,         // (N,) device, or nullptr
    __half*       output,       // (M, N) device
    int           M,
    int           K,
    int           N,
    cudaStream_t  stream = nullptr);

/// Convenience: quantize activation + dp4a GEMM + dequant in one call.
void int8_dp4a_linear(
    const __half* x,            // (M, K) fp16 device
    const int8_t* w_int8,       // (N, K) int8 device (pre-quantized weight)
    const __half* w_scale,      // (N,) fp16 device
    const __half* bias,         // (N,) fp16 device or nullptr
    __half*       output,       // (M, N) fp16 device
    int           M,
    int           K,
    int           N,
    int8_t*       scratch_a_int8,  // (M, K) int8 workspace
    float*        scratch_a_scale, // (M,) float workspace
    cudaStream_t  stream = nullptr);

}} // namespace forge::gemm
