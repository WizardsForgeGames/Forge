// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Unified INT8 GEMM dispatch. Automatically selects the best backend
// based on GPU compute capability:
//   sm_80+ (Ampere/Ada) → wmma tensor core path  (int8_dense.cu)
//   sm_61+ (Pascal/Turing) → dp4a scalar path     (int8_dp4a.cu)
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace forge { namespace gemm {

/// Returns the compute capability (major*10 + minor) of the current device.
/// Cached after first call.
int get_compute_capability(int device = 0);

/// Returns true if the current device supports wmma INT8 (sm_80+).
bool has_wmma_int8(int device = 0);

/// Auto-dispatching INT8 GEMM. Routes to wmma or dp4a based on CC.
/// Same signature as int8_gemm_dequant / int8_dp4a_gemm_dequant.
void int8_gemm(
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

}} // namespace forge::gemm
