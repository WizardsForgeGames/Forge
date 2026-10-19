// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/rng.cuh>
#include <curand_kernel.h>
#include "cuda_utils.cuh"

namespace forge { namespace nn {

// Each thread generates up to 4 values using Philox 4x32_10 (counter-based).
// Counter-based RNGs are ideal: no per-thread state to persist, stateless init
// from (seed, subsequence) pair.  Subsequence = offset + threadIdx provides
// guaranteed independence across calls with different offsets.

static __global__ void fill_normal_kernel(
    float* __restrict__ data,
    int                 n,
    float               mean,
    float               std,
    uint64_t            seed,
    uint64_t            offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, offset + idx, 0, &state);

    data[idx] = curand_normal(&state) * std + mean;
}

static __global__ void fill_uniform_kernel(
    float* __restrict__ data,
    int                 n,
    float               lo,
    float               hi,
    uint64_t            seed,
    uint64_t            offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, offset + idx, 0, &state);

    data[idx] = curand_uniform(&state) * (hi - lo) + lo;
}

void fill_normal(
    float*       data,
    int          n,
    float        mean,
    float        std,
    uint64_t     seed,
    uint64_t     offset,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(n, block);
    fill_normal_kernel<<<grid, block, 0, stream>>>(data, n, mean, std, seed, offset);
}

void fill_uniform(
    float*       data,
    int          n,
    float        lo,
    float        hi,
    uint64_t     seed,
    uint64_t     offset,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(n, block);
    fill_uniform_kernel<<<grid, block, 0, stream>>>(data, n, lo, hi, seed, offset);
}

}} // namespace forge::nn
