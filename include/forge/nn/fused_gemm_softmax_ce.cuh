// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Fused linear projection + softmax + cross-entropy forward pass.
// Computes logits = W * X + bias via tiled dot products and applies
// online softmax (FlashAttention-style) so the logits tensor is never
// materialised in global memory, saving one full V*B DRAM round-trip.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

/// Fused GEMM + softmax + cross-entropy forward.
///
/// Computes:
///   logits[v, b] = sum_d(W[v, d] * X[d, b]) + bias[v]
///   probs[v, b]  = softmax(logits[:, b])_v
///   losses[b]    = -log(probs[target[b], b])
///   correct[b]   = (argmax(probs[:, b]) == target[b]) ? 1 : 0
///
/// The logits tensor is never written to global memory.
///
/// W:       [V, D]  row-major weight matrix
/// X:       [D, B]  row-major input activations
/// bias:    [V]     per-class bias (may be nullptr)
/// probs:   [V, B]  row-major output probabilities
/// targets: [B]     ground-truth class indices
/// losses:  [B]     per-sample CE losses
/// correct: [B]     per-sample correct-prediction flags
void fused_gemm_softmax_ce_forward(
    const float* W, const float* X, const float* bias,
    float* probs, const int* targets, float* losses, int* correct,
    int V, int D, int B,
    cudaStream_t stream = nullptr);

}} // namespace forge::nn
