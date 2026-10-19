// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Mean Squared Error loss (forward and backward).
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

/// MSE forward: loss[0] = (1/n) * sum((pred[i] - target[i])^2)
/// Single-pass parallel reduction.
void mse_forward(
    const float* pred,
    const float* target,
    float*       loss,      // device scalar [1]
    int          n,
    cudaStream_t stream = nullptr);

/// MSE backward: d_pred[i] = scale * (2/n) * (pred[i] - target[i])
/// scale defaults to 1.0. Pass upstream gradient scalar for chain rule.
void mse_backward(
    const float* pred,
    const float* target,
    float*       d_pred,
    int          n,
    float        scale  = 1.0f,
    cudaStream_t stream = nullptr);

}} // namespace forge::nn
