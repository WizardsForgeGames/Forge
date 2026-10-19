// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// L1 (Mean Absolute Error) loss (forward and backward).
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

/// L1 forward: loss[0] = (1/n) * sum(|pred[i] - target[i]|)
/// Single-pass parallel reduction.
void l1_forward(
    const float* pred,
    const float* target,
    float*       loss,      // device scalar [1]
    int          n,
    cudaStream_t stream = nullptr);

/// L1 backward: d_pred[i] = scale * sign(pred[i] - target[i]) / n
/// scale defaults to 1.0. Pass upstream gradient scalar for chain rule.
void l1_backward(
    const float* pred,
    const float* target,
    float*       d_pred,
    int          n,
    float        scale  = 1.0f,
    cudaStream_t stream = nullptr);

}} // namespace forge::nn
