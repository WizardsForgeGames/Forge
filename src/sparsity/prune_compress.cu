// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/sparsity/prune_compress.cuh>
#include "cuda_utils.cuh"
#include <cstdlib>

namespace forge { namespace sparsity {

__global__ void prune_compress_kernel(
    const int8_t* __restrict__  weight,
    int8_t* __restrict__        compressed,
    uint8_t* __restrict__       metadata_raw,
    int                         N,
    int                         K)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups_per_row = K / 4;
    const int total_groups = N * groups_per_row;
    if (idx >= total_groups) return;

    const int row   = idx / groups_per_row;
    const int group = idx % groups_per_row;

    // Load 4 elements
    const int base = row * K + group * 4;
    int8_t vals[4];
    vals[0] = weight[base + 0];
    vals[1] = weight[base + 1];
    vals[2] = weight[base + 2];
    vals[3] = weight[base + 3];

    // Find 2 smallest magnitude positions (to prune)
    int abs_vals[4];
    abs_vals[0] = abs((int)vals[0]);
    abs_vals[1] = abs((int)vals[1]);
    abs_vals[2] = abs((int)vals[2]);
    abs_vals[3] = abs((int)vals[3]);

    int min1_pos = 0, min2_pos = 1;
    if (abs_vals[min1_pos] > abs_vals[min2_pos]) {
        int tmp = min1_pos; min1_pos = min2_pos; min2_pos = tmp;
    }
    for (int i = 2; i < 4; i++) {
        if (abs_vals[i] < abs_vals[min1_pos]) {
            min2_pos = min1_pos;
            min1_pos = i;
        } else if (abs_vals[i] < abs_vals[min2_pos]) {
            min2_pos = i;
        }
    }

    // Kept positions (sorted ascending)
    int keep[2];
    int ki = 0;
    for (int i = 0; i < 4; i++) {
        if (i != min1_pos && i != min2_pos) {
            keep[ki++] = i;
        }
    }

    // Store compressed values
    const int comp_base = row * (K / 2) + group * 2;
    compressed[comp_base + 0] = vals[keep[0]];
    compressed[comp_base + 1] = vals[keep[1]];

    // Store raw metadata nibble: idx0 | (idx1 << 2)
    uint8_t nibble = (uint8_t)(keep[0] | (keep[1] << 2));
    metadata_raw[idx] = nibble;
}

void prune_compress_24(
    const int8_t* weight,
    int8_t*       compressed,
    uint8_t*      metadata_raw,
    int           N,
    int           K,
    cudaStream_t  stream)
{
    if (N <= 0 || K <= 0) return;
    int total_groups = N * (K / 4);
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(total_groups, block);
    prune_compress_kernel<<<grid, block, 0, stream>>>(
        weight, compressed, metadata_raw, N, K);
}

}} // namespace forge::sparsity
