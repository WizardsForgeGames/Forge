// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Gradient norm computation and clipping, entirely on-device.
// The compute_clip_scale kernel eliminates the D->H->D roundtrip that naive
// gradient clipping requires.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace optim {

/// Partial-sum reduction: accumulate sum of grad[i]^2 into out_norm_sq.
/// Uses atomicAdd, so caller MUST zero out_norm_sq before calling.
/// Call once per parameter group, then read the combined norm.
__global__ void grad_norm_sq_kernel(
    const float* __restrict__ grad,
    int                       n,
    float* __restrict__       out_norm_sq);

/// Elementwise gradient scaling: grad[i] *= scale.
__global__ void grad_scale_kernel(
    float* __restrict__ grad,
    int                 n,
    float               scale);

/// Compute combined normalize + clip scale on GPU (single-thread kernel).
/// Output: scale = min(inv_B, max_grad_norm / (sqrt(norm_sq) * inv_B_denominator))
/// where inv_B = 1/batch_size (for gradient averaging).
__global__ void compute_clip_scale_kernel(
    float* __restrict__       out_scale,
    const float* __restrict__ norm_sq,
    float                     max_grad_norm,
    float                     inv_B);

// --- Launchers ---

void grad_norm_sq(
    const float* grad,
    int          n,
    float*       out_norm_sq,   // device [1], caller zeroes beforehand
    cudaStream_t stream = nullptr);

void grad_scale(
    float*       grad,
    int          n,
    float        scale,
    cudaStream_t stream = nullptr);

void compute_clip_scale(
    float*       out_scale,     // device [1]
    const float* norm_sq,       // device [1]
    float        max_grad_norm,
    float        inv_B,
    cudaStream_t stream = nullptr);

}} // namespace forge::optim
