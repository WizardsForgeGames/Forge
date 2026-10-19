// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/optim/adam.cuh>
#include <forge/optim/lr_schedule.h>
#include "cuda_utils.cuh"

namespace forge { namespace optim {

// ---------- LR schedule device function ----------

__device__ float device_lr_scale(int step, int warmup_steps, int total_steps,
                                 float min_lr_frac, int num_restarts = 1) {
    if (step < warmup_steps)
        return (float)(step + 1) / (float)warmup_steps;
    int decay_steps = total_steps - warmup_steps;
    if (decay_steps <= 0) return 1.0f;

    // SGDR: divide decay phase into num_restarts equal cycles
    int nr = (num_restarts > 1) ? num_restarts : 1;
    int cycle_len = decay_steps / nr;
    if (cycle_len <= 0) cycle_len = decay_steps;

    int decay_step = step - warmup_steps;
    int cycle_step = decay_step % cycle_len;

    float progress = (float)cycle_step / (float)cycle_len;
    float cosine = 0.5f * (1.0f + cosf(3.14159265f * progress));
    return min_lr_frac + (1.0f - min_lr_frac) * cosine;
}

__global__ void update_step_scalars_kernel(
    float* __restrict__ step_scalars,
    int                 adam_t,
    int                 warmup_steps,
    int                 total_steps,
    float               min_lr_frac,
    float               beta1,
    float               beta2,
    int                 num_restarts)
{
    float lr = device_lr_scale(adam_t, warmup_steps, total_steps, min_lr_frac, num_restarts);
    step_scalars[0] = lr;
    step_scalars[1] = 1.0f - powf(beta1, (float)adam_t);
    step_scalars[2] = 1.0f - powf(beta2, (float)adam_t);
}

void update_step_scalars(
    float*       step_scalars,
    int          adam_t,
    int          warmup_steps,
    int          total_steps,
    float        min_lr_frac,
    float        beta1,
    float        beta2,
    int          num_restarts,
    cudaStream_t stream)
{
    update_step_scalars_kernel<<<1, 1, 0, stream>>>(
        step_scalars, adam_t, warmup_steps, total_steps, min_lr_frac, beta1, beta2, num_restarts);
}

// ---------- Adam optimizer kernel ----------

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
    const float* __restrict__ norm_sq,
    float                     max_grad_norm,
    float                     inv_B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float lr_scale = step_scalars[0];
    float bc1      = step_scalars[1];
    float bc2      = step_scalars[2];

    // Gradient scaling: if norm_sq provided, compute clip_scale inline
    // (eliminates separate compute_clip_scale <<<1,1>>> kernel).
    // Falls back to grad_scale pointer if provided, or 1.0 if neither.
    float gs;
    if (norm_sq) {
        gs = inv_B;
        if (max_grad_norm > 0.0f) {
            float raw_norm = sqrtf(*norm_sq);
            float eff_norm = raw_norm * inv_B;
            if (eff_norm > max_grad_norm)
                gs = max_grad_norm / raw_norm;
        }
    } else {
        gs = grad_scale ? (*grad_scale) : 1.0f;
    }

    float g = grad[idx] * gs;
    if (zero_grad) grad[idx] = 0.0f;
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
    float m_hat = m[idx] / bc1;
    float v_hat = v[idx] / bc2;
    float update = m_hat / (sqrtf(v_hat) + eps);
    if (weight_decay > 0.0f) update += weight_decay * param[idx];
    param[idx] -= (base_lr * lr_scale) * update;
    if (do_clamp) {
        param[idx] = fmaxf(clamp_lo, fminf(clamp_hi, param[idx]));
    }
}

// ---------- Fused Adam step (non-graph path) ----------
// Computes LR schedule, bias correction, and clip scale inline.
// Eliminates both update_step_scalars and compute_clip_scale kernel launches.

__global__ void fused_adam_step_kernel(
    float* __restrict__       param,
    float* __restrict__       grad,
    float* __restrict__       m,
    float* __restrict__       v,
    int                       n,
    float                     base_lr,
    int                       adam_t,
    int                       warmup_steps,
    int                       total_steps,
    float                     min_lr_frac,
    float                     beta1,
    float                     beta2,
    float                     eps,
    float                     weight_decay,
    const float* __restrict__ norm_sq,
    float                     max_grad_norm,
    float                     inv_B,
    bool                      do_clamp,
    float                     clamp_lo,
    float                     clamp_hi,
    bool                      zero_grad,
    int                       num_restarts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // LR schedule + bias correction (replaces update_step_scalars <<<1,1>>>)
    float lr_scale = device_lr_scale(adam_t, warmup_steps, total_steps,
                                      min_lr_frac, num_restarts);
    float bc1 = 1.0f - powf(beta1, (float)adam_t);
    float bc2 = 1.0f - powf(beta2, (float)adam_t);

    // Clip scale (replaces compute_clip_scale <<<1,1>>>)
    float gs = inv_B;
    if (norm_sq && max_grad_norm > 0.0f) {
        float raw_norm = sqrtf(*norm_sq);
        float eff_norm = raw_norm * inv_B;
        if (eff_norm > max_grad_norm)
            gs = max_grad_norm / raw_norm;
    }

    float g = grad[idx] * gs;
    if (zero_grad) grad[idx] = 0.0f;
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
    float m_hat = m[idx] / bc1;
    float v_hat = v[idx] / bc2;
    float update = m_hat / (sqrtf(v_hat) + eps);
    if (weight_decay > 0.0f) update += weight_decay * param[idx];
    param[idx] -= (base_lr * lr_scale) * update;
    if (do_clamp)
        param[idx] = fmaxf(clamp_lo, fminf(clamp_hi, param[idx]));
}

void adam_step(
    float*       param,
    float*       grad,
    float*       m,
    float*       v,
    int          n,
    float        base_lr,
    const float* step_scalars,
    float        beta1,
    float        beta2,
    float        eps,
    float        weight_decay,
    const float* grad_scale,
    bool         do_clamp,
    float        clamp_lo,
    float        clamp_hi,
    cudaStream_t stream,
    bool         zero_grad,
    const float* norm_sq,
    float        max_grad_norm,
    float        inv_B)
{
    if (n <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(n, block);
    adam_step_kernel<<<grid, block, 0, stream>>>(
        param, grad, m, v, n, base_lr, step_scalars,
        beta1, beta2, eps, weight_decay, grad_scale,
        do_clamp, clamp_lo, clamp_hi, zero_grad,
        norm_sq, max_grad_norm, inv_B);
}

void fused_adam_step(
    float*       param,
    float*       grad,
    float*       m,
    float*       v,
    int          n,
    float        base_lr,
    int          adam_t,
    int          warmup_steps,
    int          total_steps,
    float        min_lr_frac,
    float        beta1,
    float        beta2,
    float        eps,
    float        weight_decay,
    const float* norm_sq,
    float        max_grad_norm,
    float        inv_B,
    bool         do_clamp,
    float        clamp_lo,
    float        clamp_hi,
    cudaStream_t stream,
    bool         zero_grad,
    int          num_restarts)
{
    if (n <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(n, block);
    fused_adam_step_kernel<<<grid, block, 0, stream>>>(
        param, grad, m, v, n, base_lr,
        adam_t, warmup_steps, total_steps, min_lr_frac,
        beta1, beta2, eps, weight_decay,
        norm_sq, max_grad_norm, inv_B,
        do_clamp, clamp_lo, clamp_hi, zero_grad, num_restarts);
}

}} // namespace forge::optim
