// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Fused dequantization + bias: INT32 (M,N) * act_scale(M) * w_scale(N) + bias(N) -> FP16 (M,N).
// One block per row.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace forge { namespace quant {

__global__ void fused_dequantize_bias_kernel(
    const int32_t* __restrict__ input,
    const float* __restrict__   act_scale,
    const __half* __restrict__  w_scale,
    const __half* __restrict__  bias,
    __half* __restrict__        output,
    int                         N);

void fused_dequantize_bias(
    const int32_t* input,       // (M, N) device
    const float*   act_scale,   // (M,) device
    const __half*  w_scale,     // (N,) device
    const __half*  bias,        // (N,) device, nullptr for no bias
    __half*        output,      // (M, N) device
    int            M,
    int            N,
    cudaStream_t   stream = nullptr);

}} // namespace forge::quant
