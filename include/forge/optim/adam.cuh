// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Adam optimizer kernel. Fully generic, zero domain coupling.
// step_scalars and grad_scale live on device for CUDA Graph compatibility.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace optim {

/// Adam step kernel. All pointers are device memory.
///
/// step_scalars[3] = {lr_scale, bias_correction1, bias_correction2} on device.
/// grad_scale[1]   = combined gradient normalization + clipping factor on device.
///                    Pass nullptr to skip gradient scaling.
///
/// norm_sq: if non-null and grad_scale is null, computes clip_scale inline
///          from norm_sq[0], max_grad_norm, and inv_B. This eliminates the
///          separate compute_clip_scale <<<1,1>>> kernel launch.
///          CUDA Graph compatible (pointer fixed, value changes).
__global__ void adam_step_kernel(
    float* __restrict__       param,
    float* __restrict__       grad,
    float* __restrict__       m,
    float* __restrict__       v,
    int                       n,
    float                     base_lr,
    const float* __restrict__ step_scalars,
    float                     beta1,
    float                     beta2,
    float                     eps,
    float                     weight_decay,
    const float* __restrict__ grad_scale,
    bool                      do_clamp,
    float                     clamp_lo,
    float                     clamp_hi,
    bool                      zero_grad,
    const float* __restrict__ norm_sq = nullptr,
    float                     max_grad_norm = 0.0f,
    float                     inv_B = 1.0f);

/// C++ launcher -- selects grid/block automatically.
///
/// zero_grad: if true, writes 0 to grad[i] after consuming it.
///            Eliminates need for a separate memset before the next backward pass.
///            The cache line is already loaded, so the write is essentially free.
///
/// norm_sq/max_grad_norm/inv_B: when norm_sq is non-null, computes gradient
///     clipping inline (eliminates compute_clip_scale <<<1,1>>>).
///     Pass grad_scale=nullptr when using inline clipping.
void adam_step(
    float*       param,
    float*       grad,
    float*       m,
    float*       v,
    int          n,
    float        base_lr,
    const float* step_scalars,  // device [3]
    float        beta1       = 0.9f,
    float        beta2       = 0.999f,
    float        eps         = 1e-8f,
    float        weight_decay = 0.0f,
    const float* grad_scale  = nullptr,
    bool         do_clamp    = false,
    float        clamp_lo    = 0.0f,
    float        clamp_hi    = 0.0f,
    cudaStream_t stream      = nullptr,
    bool         zero_grad   = false,
    const float* norm_sq     = nullptr,
    float        max_grad_norm = 0.0f,
    float        inv_B       = 1.0f);

/// Fused optimizer step: computes LR schedule, bias correction, and gradient
/// clipping scale inline, eliminating update_step_scalars and compute_clip_scale
/// kernel launches. NOT CUDA Graph compatible (adam_t is a kernel arg).
///
/// Training loop becomes: memset(norm_sq) -> grad_norm_sq() -> fused_adam_step()
/// (3 operations instead of 5)
void fused_adam_step(
    float*       param,
    float*       grad,
    float*       m,
    float*       v,
    int          n,
    float        base_lr,
    int          adam_t,           // current step (1-indexed)
    int          warmup_steps,
    int          total_steps,
    float        min_lr_frac,
    float        beta1       = 0.9f,
    float        beta2       = 0.999f,
    float        eps         = 1e-8f,
    float        weight_decay = 0.0f,
    const float* norm_sq     = nullptr,  // device [1], nullptr = no clipping
    float        max_grad_norm = 0.0f,
    float        inv_B       = 1.0f,
    bool         do_clamp    = false,
    float        clamp_lo    = 0.0f,
    float        clamp_hi    = 0.0f,
    cudaStream_t stream      = nullptr,
    bool         zero_grad   = false,
    int          num_restarts = 1);

}} // namespace forge::optim
