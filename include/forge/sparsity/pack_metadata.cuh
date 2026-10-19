// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Pack raw metadata nibbles into per-row uint32 chunks (hardware format).
// Each uint32 packs 8 consecutive nibbles (one 32-element dense K-chunk).
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace forge { namespace sparsity {

/// Pack raw metadata nibbles into hardware format.
/// meta_raw: (N * K/4) uint8 -> meta_packed: (N * K/32) uint32
__global__ void pack_metadata_kernel(
    const uint8_t* __restrict__ meta_raw,
    uint32_t* __restrict__      meta_packed,
    int                         N,
    int                         K);

void pack_metadata(
    const uint8_t* meta_raw,    // (N * K/4) device
    uint32_t*      meta_packed, // (N * K/32) device
    int            N,
    int            K,
    cudaStream_t   stream = nullptr);

}} // namespace forge::sparsity
