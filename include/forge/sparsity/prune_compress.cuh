// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// 2:4 structured sparsity: prune to 2-of-4 pattern and compress.
// For each group of 4 elements along K, keeps the 2 with largest magnitude.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace forge { namespace sparsity {

/// Prune to 2:4 pattern and compress.
/// weight: (N, K) int8 dense -> compressed: (N, K/2) int8 + metadata_raw: (N*K/4) uint8
/// One thread per group of 4 elements.
__global__ void prune_compress_kernel(
    const int8_t* __restrict__  weight,
    int8_t* __restrict__        compressed,
    uint8_t* __restrict__       metadata_raw,
    int                         N,
    int                         K);

void prune_compress_24(
    const int8_t* weight,       // (N, K) device
    int8_t*       compressed,   // (N, K/2) device
    uint8_t*      metadata_raw, // (N * K/4) device
    int           N,
    int           K,
    cudaStream_t  stream = nullptr);

}} // namespace forge::sparsity
