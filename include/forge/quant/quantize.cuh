// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Fused per-token activation quantization: FP16 (M, K) -> INT8 (M, K) + scales (M,).
// One block per row. When K % 8 == 0, K <= 8192 and input/output are 16/8-byte aligned,
// a single-pass kernel holds the row in registers (16-byte loads, one read of the input;
// evict-first loads when the input exceeds L2). Otherwise the two-pass kernel below
// (parallel max reduction, then quantize) runs. Both produce identical output.
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
