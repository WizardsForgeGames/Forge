// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Hyperbolic geometry primitives for the Poincare ball model.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace primitives {

/// Project point x onto the Poincare ball of radius max_norm.
/// If ||x|| >= max_norm, rescales x in-place so ||x|| = max_norm - eps.
__device__ __forceinline__ void poincare_project(float* x, int D, float max_norm) {
    float nsq = 0.0f;
    for (int d = 0; d < D; d++)
        nsq += x[d] * x[d];
    if (nsq >= max_norm * max_norm) {
        float scale = (max_norm - 1e-5f) / sqrtf(nsq);
        for (int d = 0; d < D; d++)
            x[d] *= scale;
    }
}

/// Conformal factor for the Poincare ball.
/// Returns ((1 - norm_sq)^power) / 4.
/// norm_sq should be pre-computed as sum of x[d]^2.
__device__ __forceinline__ float conformal_factor(float norm_sq, float power) {
    float base = fmaxf(1.0f - norm_sq, 1e-10f);
    return powf(base, power) * 0.25f;
}

/// Gradient of boundary repulsion term: -lambda * log(1 - ||x||^2).
/// ADDS the gradient in-place to grad[0..D-1] (accumulation pattern).
__device__ __forceinline__ void boundary_repulsion_grad(
    float* grad, const float* x, int D, float lambda)
{
    float nsq = 0.0f;
    for (int d = 0; d < D; d++)
        nsq += x[d] * x[d];
    float denom = fmaxf(1.0f - nsq, 1e-10f);
    float coeff = 2.0f * lambda / denom;
    for (int d = 0; d < D; d++)
        grad[d] += coeff * x[d];
}

}} // namespace forge::primitives
