// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Sigmoid gating kernels for modulating activations.
//
// Forward:  gated[i] = sigmoid(w[unit] * x[i] + b[unit]) * x[i]
// Backward: given d_gated[i], computes dx[i], dw[unit], db[unit].
//
// Layout: x is [N, B] (N units, B batch), w and b are [N] (per-unit params).
// The unit index for element i is i / B (integer division).
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

/// Forward: apply per-unit sigmoid gate to activations.
/// gated[i] = sigmoid(w[i/B] * x[i] + b[i/B]) * x[i]
void sigmoid_gate_forward(
    const float* x,
    const float* w,
    const float* b,
    float*       gated,
    int          N,
    int          B,
    cudaStream_t stream = nullptr);

/// Backward: given d_gated (gradient w.r.t. gated output), compute dx and
/// dw, db (reduced over batch dimension).
///
/// dx[i]    = d_gated[i] * (gate + x[i] * gate' * w[i/B])
/// dw[unit] = sum_s d_gated[unit,s] * x[unit,s]^2 * gate'
/// db[unit] = sum_s d_gated[unit,s] * x[unit,s]   * gate'
///
/// where gate  = sigmoid(w[unit] * x[i] + b[unit])
///       gate' = gate * (1 - gate)
///
/// NOTE: dw and db are written (not accumulated). Caller should zero them
/// beforehand if accumulating across multiple calls.
void sigmoid_gate_backward(
    const float* d_gated,
    const float* x,
    const float* w,
    const float* b,
    float*       dx,
    float*       dw,
    float*       db,
    int          N,
    int          B,
    cudaStream_t stream = nullptr);

}} // namespace forge::nn
