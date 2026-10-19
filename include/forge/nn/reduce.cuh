// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Reduction kernels: sum-to-scalar, mean over leading dimension.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

/// Sum all elements to a scalar: out[0] = sum(x[0..n-1]).
/// Overwrites out[0]. out must point to at least 1 device float.
void reduce_sum(
    const float* x,
    float*       out,
    int          n,
    cudaStream_t stream = nullptr);

/// Sum of squares to a scalar: out[0] = sum(x[i]^2).
void reduce_sumsq(
    const float* x,
    float*       out,
    int          n,
    cudaStream_t stream = nullptr);

/// Absolute maximum to a scalar: out[0] = max(|x[i]|).
void reduce_absmax(
    const float* x,
    float*       out,
    int          n,
    cudaStream_t stream = nullptr);

/// Mean to a scalar: out[0] = sum(x[0..n-1]) / n.
void reduce_mean(
    const float* x,
    float*       out,
    int          n,
    cudaStream_t stream = nullptr);

/// Mean over the leading dimension.
/// x is [D0, D1] row-major, out is [D1].
/// out[j] = (1/D0) * sum_{i=0}^{D0-1} x[i * D1 + j]
void reduce_mean_dim0(
    const float* x,
    float*       out,
    int          D0,
    int          D1,
    cudaStream_t stream = nullptr);

/// Sum over the leading dimension (no averaging).
/// x is [D0, D1] row-major, out is [D1].
/// out[j] = sum_{i=0}^{D0-1} x[i * D1 + j]
void reduce_sum_dim0(
    const float* x,
    float*       out,
    int          D0,
    int          D1,
    cudaStream_t stream = nullptr);

}} // namespace forge::nn
