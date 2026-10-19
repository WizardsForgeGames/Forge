// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Fused softmax + cross-entropy loss (forward and backward).
// Forward: one block per sample, parallel reduction over V (vocab/output dim).
// Backward: elementwise probs - one_hot.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

/// Fused softmax + cross-entropy forward.
/// Computes: probs = softmax(logits), loss = -log(probs[target]), prediction accuracy.
///
/// Layout: logits and probs are [V, B] col-major.
/// Launch: <<<B, threads, threads * 8>>> where threads is power-of-2.
__global__ void softmax_ce_forward_kernel(
    float* __restrict__       logits,
    float* __restrict__       probs,
    const int* __restrict__   targets,
    float* __restrict__       losses,
    int* __restrict__         correct,
    int                       V,
    int                       B);

void softmax_ce_forward(
    float*       logits,       // [V, B] col-major, device
    float*       probs,        // [V, B] col-major, device (output)
    const int*   targets,      // [B] device
    float*       losses,       // [B] device (output)
    int*         correct,      // [B] device (output)
    int          V,
    int          B,
    int          threads = 256,
    cudaStream_t stream  = nullptr);

/// Softmax + CE backward: d_logits = probs - one_hot(target).
__global__ void softmax_ce_backward_kernel(
    const float* __restrict__ probs,
    const int* __restrict__   targets,
    float* __restrict__       d_logits,
    int                       V,
    int                       B);

void softmax_ce_backward(
    const float* probs,
    const int*   targets,
    float*       d_logits,
    int          V,
    int          B,
    cudaStream_t stream = nullptr);

/// Fused backward + bias gradient.
///
/// Computes d_logits = probs - one_hot(targets) and accumulates
/// d_bias[v] += sum_b d_logits[v, b] in a single pass.
/// One block per V-row with coalesced B-dimension access.
///
/// This replaces the separate backward + bias_grad_accumulate calls,
/// eliminating one full V*B read of d_logits.
__global__ void softmax_ce_backward_bias_kernel(
    const float* __restrict__ probs,
    const int* __restrict__   targets,
    float* __restrict__       d_logits,
    float* __restrict__       d_bias,
    int                       V,
    int                       B);

void softmax_ce_backward_bias(
    const float* probs,
    const int*   targets,
    float*       d_logits,
    float*       d_bias,
    int          V,
    int          B,
    int          threads = 256,
    cudaStream_t stream  = nullptr);

}} // namespace forge::nn
