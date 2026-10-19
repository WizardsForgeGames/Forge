// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Fused per-token activation quantization: FP16 (M, K) -> INT8 (M, K) + scales (M,).
// One block per row. Two-pass: parallel max reduction, then quantize.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace forge { namespace quant {

__global__ void fused_quantize_activation_kernel(
    const __half* __restrict__ input,
    int8_t* __restrict__       output,
    float* __restrict__        scales,
    int                        K);

void fused_quantize_activation(
    const __half* input,        // (M, K) device
    int8_t*       output,       // (M, K) device
    float*        scales,       // (M,) device
    int           M,
    int           K,
    cudaStream_t  stream = nullptr);

}} // namespace forge::quant
