// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Activation function device primitives (forward and backward).
// Scalar, per-element -- call from within your own kernel.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace primitives {

// ---------- Leaky ReLU ----------

__device__ __forceinline__ float leaky_relu(float z, float slope = 0.01f) {
    return (z > 0.0f) ? z : slope * z;
}

__device__ __forceinline__ float leaky_relu_grad(float z, float dy, float slope = 0.01f) {
    return (z > 0.0f) ? dy : slope * dy;
}

// ---------- Tanh ----------

/// Tanh backward given the output of tanh (not the input).
/// dz = dy * (1 - tanh_out^2)
__device__ __forceinline__ float tanh_grad(float tanh_out, float dy) {
    return dy * (1.0f - tanh_out * tanh_out);
}

// ---------- GELU (tanh approximation) ----------
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

__device__ __forceinline__ float gelu(float z) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kCoeff = 0.044715f;
    float inner = kSqrt2OverPi * (z + kCoeff * z * z * z);
    return 0.5f * z * (1.0f + tanhf(inner));
}

/// GELU backward. Takes the original input z (not the output).
__device__ __forceinline__ float gelu_grad(float z, float dy) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kCoeff = 0.044715f;
    float inner = kSqrt2OverPi * (z + kCoeff * z * z * z);
    float t = tanhf(inner);
    float sech2 = 1.0f - t * t;
    float d_inner = kSqrt2OverPi * (1.0f + 3.0f * kCoeff * z * z);
    return dy * (0.5f * (1.0f + t) + 0.5f * z * sech2 * d_inner);
}

// ---------- Sigmoid ----------

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/// Sigmoid derivative given the output of sigmoid (not the input).
/// dsig = sig * (1 - sig)
__device__ __forceinline__ float sigmoid_grad(float sig_out) {
    return sig_out * (1.0f - sig_out);
}

// ---------- SiLU (Sigmoid Linear Unit / Swish) ----------
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

__device__ __forceinline__ float silu(float x) {
    float s = 1.0f / (1.0f + expf(-x));
    return x * s;
}

/// SiLU backward given the original input z (not the output).
/// dsilu/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
///          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
__device__ __forceinline__ float silu_grad(float z, float dy) {
    float s = 1.0f / (1.0f + expf(-z));
    return dy * (s + z * s * (1.0f - s));
}

// ---------- CELU ----------
// celu(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
// Equivalent to: (x > 0) ? x : alpha * (exp(x / alpha) - 1)

__device__ __forceinline__ float celu(float x, float alpha = 1.0f) {
    return (x > 0.0f) ? x : alpha * (expf(x / alpha) - 1.0f);
}

/// CELU backward given the original input z (not the output).
/// dcelu/dx = 1 if x > 0, exp(x / alpha) if x <= 0
__device__ __forceinline__ float celu_grad(float z, float dy, float alpha = 1.0f) {
    return (z > 0.0f) ? dy : dy * expf(z / alpha);
}

}} // namespace forge::primitives
