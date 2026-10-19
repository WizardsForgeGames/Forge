// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Embedding gather (forward) and scatter (backward) kernels.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

/// Embedding gather: context[B * ctx_size] -> output[I, B] col-major
/// where I = ctx_size * embed_dim.
__global__ void embedding_gather_kernel(
    const int* __restrict__   context,
    const float* __restrict__ embeddings,
    float* __restrict__       output,
    int B, int ctx_size, int E, int I);

void embedding_gather(
    const int*   context,       // [B * ctx_size] device
    const float* embeddings,    // [V, E] device, row-major
    float*       output,        // [I, B] device, col-major, I = ctx_size * E
    int          B,
    int          ctx_size,
    int          E,
    cudaStream_t stream = nullptr);

/// Embedding scatter (backward): atomicAdd gradients back to d_embeddings.
__global__ void embedding_scatter_kernel(
    const int* __restrict__   context,
    const float* __restrict__ d_output,
    float* __restrict__       d_embeddings,
    int B, int ctx_size, int E, int I);

void embedding_scatter(
    const int*   context,
    const float* d_output,      // [I, B] col-major
    float*       d_embeddings,  // [V, E] row-major (atomicAdd target)
    int          B,
    int          ctx_size,
    int          E,
    cudaStream_t stream = nullptr);

}} // namespace forge::nn
