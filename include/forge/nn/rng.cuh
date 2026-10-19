// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// On-device random number generation using cuRAND Philox engine.
// Deterministic given (seed, offset): call with incrementing offset
// for fresh, reproducible sequences.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace forge { namespace nn {

/// Fill buffer with samples from N(mean, std).
/// seed: Philox seed (same seed + offset = same sequence).
/// offset: subsequence offset — increment by n after each call.
void fill_normal(
    float*       data,
    int          n,
    float        mean,
    float        std,
    uint64_t     seed,
    uint64_t     offset = 0,
    cudaStream_t stream = nullptr);

/// Fill buffer with samples from U(lo, hi).
void fill_uniform(
    float*       data,
    int          n,
    float        lo,
    float        hi,
    uint64_t     seed,
    uint64_t     offset = 0,
    cudaStream_t stream = nullptr);

}} // namespace forge::nn
