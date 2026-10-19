// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Public warp- and block-level reduction primitives.
// All functions are __device__ __forceinline__ -- zero overhead, composable
// inside user kernels.
//
// Provides:
//   1. Reduction op structs (SumOp, MaxOp, MinOp) with identity + combine
//   2. Generic templates: warp_reduce<Op>, block_reduce<Op>, block_broadcast<Op>
//   3. Backward-compatible monomorphic wrappers (warp_reduce_sum, etc.)
#pragma once

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace forge { namespace primitives {

// ===================================================================
// Reduction operation structs
// ===================================================================
// Each op provides:
//   static __device__ float identity()        -- neutral element
//   static __device__ float combine(a, b)     -- associative binary op
//   static __device__ float shfl_combine(val, offset) -- warp shuffle + combine

struct SumOp {
    static __device__ __forceinline__ float identity() { return 0.0f; }
    static __device__ __forceinline__ float combine(float a, float b) { return a + b; }
    static __device__ __forceinline__ float shfl_combine(float val, int offset) {
        return val + __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
};

struct MaxOp {
    static __device__ __forceinline__ float identity() { return -FLT_MAX; }
    static __device__ __forceinline__ float combine(float a, float b) { return fmaxf(a, b); }
    static __device__ __forceinline__ float shfl_combine(float val, int offset) {
        return fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
};

struct MinOp {
    static __device__ __forceinline__ float identity() { return FLT_MAX; }
    static __device__ __forceinline__ float combine(float a, float b) { return fminf(a, b); }
    static __device__ __forceinline__ float shfl_combine(float val, int offset) {
        return fminf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
};

struct ProdOp {
    static __device__ __forceinline__ float identity() { return 1.0f; }
    static __device__ __forceinline__ float combine(float a, float b) { return a * b; }
    static __device__ __forceinline__ float shfl_combine(float val, int offset) {
        return val * __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
};

// ===================================================================
// Generic warp-level reduction (full warp, 32 threads)
// ===================================================================
// Result is valid ONLY in lane 0 of each warp.

template <typename Op>
__device__ __forceinline__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = Op::shfl_combine(val, offset);
    return val;
}

// ===================================================================
// Generic block-level reduction (require shared memory)
// ===================================================================
// smem must have at least (blockDim.x / 32) floats.
// Return value is valid ONLY in thread 0 of the block.

template <typename Op>
__device__ __forceinline__ float block_reduce(float val, float* smem) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce<Op>(val);

    if (lane == 0) smem[wid] = val;
    __syncthreads();

    int num_warps = blockDim.x >> 5;
    val = (threadIdx.x < num_warps) ? smem[threadIdx.x] : Op::identity();
    if (wid == 0) val = warp_reduce<Op>(val);

    return val;
}

// ===================================================================
// Generic block broadcast: reduce + broadcast to ALL threads
// ===================================================================
// smem must have at least (blockDim.x / 32 + 1) floats.

template <typename Op>
__device__ __forceinline__ float block_broadcast(float val, float* smem) {
    val = block_reduce<Op>(val, smem);
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

// ===================================================================
// Backward-compatible monomorphic wrappers
// ===================================================================

// --- Warp-level ---

__device__ __forceinline__ float warp_reduce_sum(float val) {
    return warp_reduce<SumOp>(val);
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    return warp_reduce<MaxOp>(val);
}

__device__ __forceinline__ float warp_reduce_min(float val) {
    return warp_reduce<MinOp>(val);
}

__device__ __forceinline__ float warp_reduce_prod(float val) {
    return warp_reduce<ProdOp>(val);
}

// --- Block-level ---

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    return block_reduce<SumOp>(val, smem);
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    return block_reduce<MaxOp>(val, smem);
}

__device__ __forceinline__ float block_reduce_min(float val, float* smem) {
    return block_reduce<MinOp>(val, smem);
}

__device__ __forceinline__ float block_reduce_prod(float val, float* smem) {
    return block_reduce<ProdOp>(val, smem);
}

// --- Block broadcast ---

__device__ __forceinline__ float block_broadcast_sum(float val, float* smem) {
    return block_broadcast<SumOp>(val, smem);
}

__device__ __forceinline__ float block_broadcast_max(float val, float* smem) {
    return block_broadcast<MaxOp>(val, smem);
}

__device__ __forceinline__ float block_broadcast_min(float val, float* smem) {
    return block_broadcast<MinOp>(val, smem);
}

}} // namespace forge::primitives
