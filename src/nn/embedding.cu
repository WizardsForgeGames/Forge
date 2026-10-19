// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/embedding.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace nn {

__global__ void embedding_gather_kernel(
    const int* __restrict__   context,
    const float* __restrict__ embeddings,
    float* __restrict__       output,
    int B, int ctx_size, int E, int I)
{
    int b = blockIdx.x;
    int d = threadIdx.x;
    if (b >= B || d >= I) return;

    int c   = d / E;
    int e   = d % E;
    int tok = context[b * ctx_size + c];
    output[d * B + b] = embeddings[tok * E + e];
}

void embedding_gather(
    const int*   context,
    const float* embeddings,
    float*       output,
    int          B,
    int          ctx_size,
    int          E,
    cudaStream_t stream)
{
    if (B <= 0 || ctx_size <= 0 || E <= 0) return;
    int I = ctx_size * E;
    // One block per batch element, I threads per block.
    // For large I, fall back to multiple blocks.
    if (I <= 1024) {
        embedding_gather_kernel<<<B, I, 0, stream>>>(
            context, embeddings, output, B, ctx_size, E, I);
    } else {
        // Fallback for very large embeddings: use 2D grid
        int threads = 256;
        dim3 grid(B, internal::divup(I, threads));
        // Requires modified kernel -- for now, cap at I=1024
        embedding_gather_kernel<<<B, 1024, 0, stream>>>(
            context, embeddings, output, B, ctx_size, E, I);
    }
}

__global__ void embedding_scatter_kernel(
    const int* __restrict__   context,
    const float* __restrict__ d_output,
    float* __restrict__       d_embeddings,
    int B, int ctx_size, int E, int I)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * ctx_size * E;
    if (idx >= total) return;

    int be = idx % E;
    int bc = (idx / E) % ctx_size;
    int b  = idx / (ctx_size * E);

    int tok = context[b * ctx_size + bc];
    int d   = bc * E + be;
    atomicAdd(&d_embeddings[tok * E + be], d_output[d * B + b]);
}

void embedding_scatter(
    const int*   context,
    const float* d_output,
    float*       d_embeddings,
    int          B,
    int          ctx_size,
    int          E,
    cudaStream_t stream)
{
    if (B <= 0 || ctx_size <= 0 || E <= 0) return;
    int I     = ctx_size * E;
    int total = B * ctx_size * E;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(total, block);
    embedding_scatter_kernel<<<grid, block, 0, stream>>>(
        context, d_output, d_embeddings, B, ctx_size, E, I);
}

}} // namespace forge::nn
