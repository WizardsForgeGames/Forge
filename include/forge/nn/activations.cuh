// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Activation function backward kernels.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

/// Leaky ReLU backward: dz[i] = (z[i] > 0) ? da[i] : slope * da[i]
__global__ void leaky_relu_backward_kernel(
    const float* __restrict__ z,
    const float* __restrict__ da,
    float* __restrict__       dz,
    int                       n,
    float                     slope);

void leaky_relu_backward(
    const float* z,
    const float* da,
    float*       dz,
    int          n,
    float        slope   = 0.01f,
    cudaStream_t stream  = nullptr);

/// Tanh backward: dz[i] = da[i] * (1 - a_out[i]^2)
__global__ void tanh_backward_kernel(
    const float* __restrict__ da,
    const float* __restrict__ a_out,
    float* __restrict__       dz,
    int                       n);

void tanh_backward(
    const float* da,
    const float* a_out,
    float*       dz,
    int          n,
    cudaStream_t stream = nullptr);

/// SiLU backward: dz[i] = da[i] * (sig(z[i]) + z[i] * sig(z[i]) * (1 - sig(z[i])))
__global__ void silu_backward_kernel(
    const float* __restrict__ z,
    const float* __restrict__ da,
    float* __restrict__       dz,
    int                       n);

void silu_backward(
    const float* z,
    const float* da,
    float*       dz,
    int          n,
    cudaStream_t stream = nullptr);

/// CELU backward: dz[i] = (z[i] > 0) ? da[i] : da[i] * exp(z[i] / alpha)
__global__ void celu_backward_kernel(
    const float* __restrict__ z,
    const float* __restrict__ da,
    float* __restrict__       dz,
    int                       n,
    float                     alpha);

void celu_backward(
    const float* z,
    const float* da,
    float*       dz,
    int          n,
    float        alpha  = 1.0f,
    cudaStream_t stream = nullptr);

}} // namespace forge::nn
