// Forge GPU Compute Library -- Internal
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Internal reduction wrappers. Delegate to forge::primitives.
#pragma once

#include <forge/primitives/reduce.cuh>

namespace forge { namespace internal {

__device__ __forceinline__ float warp_reduce_sum(float val) {
    return primitives::warp_reduce_sum(val);
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    return primitives::warp_reduce_max(val);
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    return primitives::block_reduce_sum(val, smem);
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    return primitives::block_reduce_max(val, smem);
}

}} // namespace forge::internal
