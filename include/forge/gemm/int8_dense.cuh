// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Dense INT8 Tensor Core GEMM with fused dequant + bias.
// Uses nvcuda::wmma intrinsics. Two tile configs selected at runtime.
//
// Y[m,n] = sum_k(A[m,k] * B[n,k]) * act_scale[m] * w_scale[n] + bias[n]
//
// A: (M, K) row-major int8
// B: (N, K) row-major int8 (note: N-K layout)
// Output: (M, N) fp16
//
// Requires: K % 64 == 0, N % 16 == 0.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace forge { namespace gemm {

/// Dense INT8 TC GEMM with fused dequant + bias.
/// Automatically selects 128x64 (narrow) or 128x128 (wide) tile.
void int8_gemm_dequant(
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

/// Convenience: quantize activation + GEMM + dequant in one call.
void int8_linear(
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
