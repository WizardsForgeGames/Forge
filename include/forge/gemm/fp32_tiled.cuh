// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// FP32 tiled GEMM with shared-memory data reuse.
// Supports all transpose combinations (NN, NT, TN, TT) with coalesced
// global memory access and bank-conflict-free shared memory layout.
// Auto-selects tile configuration based on problem dimensions.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace gemm {

/// Transpose flag for GEMM operands.
enum class Op { N, T };

/// General FP32 tiled GEMM (row-major convention).
///
/// Computes: C = alpha * op(A) * op(B) + beta * C
///
/// op(A) is [M, K], op(B) is [K, N], C is [M, N].
/// All matrices stored **row-major** (C/CUDA standard).
///
/// Leading dimensions (stride between rows):
///   lda: >= K if transA == N, >= M if transA == T
///   ldb: >= N if transB == N, >= K if transB == T
///   ldc: >= N
///
/// Optional bias: pointer to [M] vector added to each row of the result.
///
/// Tile configuration is auto-selected based on problem dimensions:
///   - Large K, small M: TM=16, TN=32, TK=64 (amortize sync overhead)
///   - Large M:          TM=64, TN=32, TK=32, 2x1 register tiling
///   - Balanced default: TM=32, TN=32, TK=32
void fp32_gemm(
    Op transA, Op transB,
    int M, int N, int K,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float beta,
    float* C, int ldc,
    const float* bias = nullptr,
    cudaStream_t stream = nullptr);

/// Split-K GEMM: partitions the reduction dimension K across multiple
/// thread blocks (blockIdx.z) for better SM utilisation when M is small.
///
/// Each K-slice computes partial dot products into a workspace buffer.
/// A lightweight reduction kernel then sums the partials and applies
/// alpha, beta, and bias.
///
/// workspace must point to at least M * N * K_splits floats of device memory.
/// K_splits is the number of K-partitions (typically 2-8).
///
/// Returns the minimum workspace size in floats: M * N * K_splits.
inline int fp32_gemm_splitk_workspace(int M, int N, int K_splits) {
    return M * N * K_splits;
}

void fp32_gemm_splitk(
    Op transA, Op transB,
    int M, int N, int K,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float beta,
    float* C, int ldc,
    const float* bias,
    float* workspace,
    int K_splits,
    cudaStream_t stream = nullptr);

}} // namespace forge::gemm
