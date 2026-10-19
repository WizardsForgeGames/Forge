// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/sparsity/pack_metadata.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace sparsity {

__global__ void pack_metadata_kernel(
    const uint8_t* __restrict__ meta_raw,
    uint32_t* __restrict__      meta_packed,
    int                         N,
    int                         K)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups_per_row = K / 4;
    const int chunks_per_row = K / 32;
    const int total = N * chunks_per_row;
    if (idx >= total) return;

    const int row   = idx / chunks_per_row;
    const int chunk = idx % chunks_per_row;

    uint32_t packed = 0;
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        uint8_t nib = meta_raw[row * groups_per_row + chunk * 8 + g];
        packed |= ((uint32_t)(nib & 0xF)) << (g * 4);
    }
    meta_packed[idx] = packed;
}

void pack_metadata(
    const uint8_t* meta_raw,
    uint32_t*      meta_packed,
    int            N,
    int            K,
    cudaStream_t   stream)
{
    if (N <= 0 || K <= 0) return;
    int total = N * (K / 32);
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(total, block);
    pack_metadata_kernel<<<grid, block, 0, stream>>>(meta_raw, meta_packed, N, K);
}

}} // namespace forge::sparsity
