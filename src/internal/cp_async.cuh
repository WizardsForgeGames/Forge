// Forge GPU Compute Library -- Internal
// Copyright (c) 2026 Darren Sander. MIT License.
//
// cp.async infrastructure (SM80+, Ampere and later).
// Copies 16 bytes from global to shared memory without going through registers.
// Extracted from Sparsemma (int8_sparse_tc.cu).
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace forge { namespace internal {

// Copy 16 bytes from global to shared memory, cache in L1+L2.
__device__ __forceinline__ void cp_async_16(void* dst_smem, const void* src_gmem) {
    uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_addr), "l"(src_gmem)
    );
}

// Copy 16 bytes with cp.async.cg: cached in L2 only, bypassing L1 (no eviction-priority
// hint). Use for streamed data (activations) that won't be re-read from L1.
__device__ __forceinline__ void cp_async_16_stream(void* dst_smem, const void* src_gmem) {
    uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_addr), "l"(src_gmem)
    );
}

// Shared-address (u32) forms: the destination is already a shared-window address, so
// per-tile precomputed offsets can be used without a generic->shared conversion per copy.
__device__ __forceinline__ void cp_async_16_u32(uint32_t dst_smem, const void* src_gmem) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst_smem), "l"(src_gmem));
}
__device__ __forceinline__ void cp_async_16_stream_u32(uint32_t dst_smem, const void* src_gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_smem), "l"(src_gmem));
}
__device__ __forceinline__ void cp_async_8_u32(uint32_t dst_smem, const void* src_gmem) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(dst_smem), "l"(src_gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// XOR bank-conflict-free smem swizzle for A tiles (32 bytes/row, 2 x 16-byte chunks).
__device__ __forceinline__ int a_swiz(int row, int byte_col) {
    return (((byte_col >> 4) ^ ((row >> 2) & 1)) << 4) | (byte_col & 15);
}

// XOR bank-conflict-free smem swizzle for B tiles (64 bytes/row, 4 x 16-byte chunks).
__device__ __forceinline__ int b_swiz(int row, int byte_col) {
    return (((byte_col >> 4) ^ ((row >> 1) & 3)) << 4) | (byte_col & 15);
}

}} // namespace forge::internal
