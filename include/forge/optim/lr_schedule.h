// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Learning rate scheduler: linear warmup + cosine annealing.
// Includes GPU-side update_step_scalars kernel for CUDA Graph compatibility.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace optim {

/// CPU-side LR scheduler: warmup linear ramp + cosine annealing with optional restarts.
struct LRScheduler {
    int   warmup_steps    = 0;
    int   total_steps     = 1;
    float min_lr_fraction = 0.01f;
    int   num_restarts    = 1;    ///< SGDR: cosine cycles (1 = no restarts)

    LRScheduler() = default;
    LRScheduler(int warmup, int total, float min_frac = 0.01f, int restarts = 1);

    /// Returns lr multiplier in [min_lr_fraction, 1.0].
    float get_lr_scale(int step) const;
};

/// GPU kernel: compute step_scalars[3] = {lr_scale, bc1, bc2} on device.
/// Eliminates host-to-device scalar upload per step (CUDA Graph compatible).
__global__ void update_step_scalars_kernel(
    float* __restrict__ step_scalars,
    int                 adam_t,
    int                 warmup_steps,
    int                 total_steps,
    float               min_lr_frac,
    float               beta1,
    float               beta2,
    int                 num_restarts = 1);

/// Launcher for update_step_scalars_kernel (single-thread kernel).
void update_step_scalars(
    float*       step_scalars,   // device [3]
    int          adam_t,
    int          warmup_steps,
    int          total_steps,
    float        min_lr_frac,
    float        beta1,
    float        beta2,
    int          num_restarts = 1,
    cudaStream_t stream = nullptr);

}} // namespace forge::optim
