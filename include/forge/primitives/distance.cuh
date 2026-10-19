// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Distance and norm primitives for device code.
// All per-thread: caller is responsible for any cross-thread reduction.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace primitives {

/// Squared Euclidean distance: sum_d (a[d] - b[d])^2
__device__ __forceinline__ float dist_sq(const float* a, const float* b, int D) {
    float acc = 0.0f;
    for (int d = 0; d < D; d++) {
        float diff = a[d] - b[d];
        acc += diff * diff;
    }
    return acc;
}

/// Squared L2 norm: sum_d x[d]^2
__device__ __forceinline__ float norm_sq(const float* x, int D) {
    float acc = 0.0f;
    for (int d = 0; d < D; d++)
        acc += x[d] * x[d];
    return acc;
}

/// Poincare ball distance: acosh(1 + 2 * ||a-b||^2 / ((1-||a||^2)(1-||b||^2)))
/// Assumes a and b are inside the unit ball (norm < 1).
__device__ __forceinline__ float dist_poincare(const float* a, const float* b, int D) {
    float diff_sq = 0.0f, na_sq = 0.0f, nb_sq = 0.0f;
    for (int d = 0; d < D; d++) {
        float da = a[d], db = b[d];
        float diff = da - db;
        diff_sq += diff * diff;
        na_sq   += da * da;
        nb_sq   += db * db;
    }
    float denom = fmaxf((1.0f - na_sq) * (1.0f - nb_sq), 1e-10f);
    float arg   = 1.0f + 2.0f * diff_sq / denom;
    return acoshf(fmaxf(arg, 1.0f));
}

}} // namespace forge::primitives
