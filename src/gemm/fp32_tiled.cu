// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/gemm/fp32_tiled.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace gemm {

// ============================================================
//  Tiled GEMM kernel template
// ============================================================
//
// Template parameters:
//   TRANS_A/TRANS_B: transpose flags (compile-time for coalesced indexing)
//   TM, TN: output tile dimensions (M and N)
//   TK: reduction tile dimension
//   REG_M: register tiling factor in M dimension (each thread computes
//          REG_M output rows, sharing the B-tile load across them)
//
// Threads per block: (TM / REG_M) * TN
// Shared memory: s_A[TM][TK+1] + s_B[TN][TK+1]  (+1 avoids bank conflicts)
//
// Both tiles use the reduction dimension (TK) as the stride-1 (column) index.
// This ensures the inner compute loop (iterating over tk) has stride-1 access
// to shared memory, enabling the compiler to use vector loads (LDS.128).
//
// Row-major convention:
//   A[i,j] = A_ptr[i * lda + j]  → stride-1 in j (the column index)
//   op(A)[m,k] with !TRANS_A: A[m * lda + k]  → stride-1 in k
//   op(A)[m,k] with  TRANS_A: A[k * lda + m]  → stride-1 in m
//
// Coalesced loading strategy: the fast (stride-1) dimension is always
// assigned to the inner index of the cooperative load so that consecutive
// threads access consecutive memory addresses.

template <bool TRANS_A, bool TRANS_B, int TM, int TN, int TK, int REG_M = 1>
__global__ void fp32_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__       C,
    const float* __restrict__ bias,
    int M, int N, int K,
    float alpha, float beta,
    int lda, int ldb, int ldc)
{
    constexpr int THREADS = (TM / REG_M) * TN;

    int m_start = blockIdx.x * TM;
    int n_start = blockIdx.y * TN;

    // Thread → output element mapping
    int local_m_base = (threadIdx.x / TN) * REG_M;
    int local_n      = threadIdx.x % TN;

    float acc[REG_M];
    #pragma unroll
    for (int r = 0; r < REG_M; r++) acc[r] = 0.0f;

    __shared__ float s_A[TM][TK + 1];   // [row=m, col=k] stride-1 in k
    __shared__ float s_B[TN][TK + 1];   // [row=n, col=k] stride-1 in k

    for (int k0 = 0; k0 < K; k0 += TK) {
        int tile_k = min(TK, K - k0);

        // ---- Load A tile: op(A)[m_start..+TM, k0..+TK] → s_A[tm][tk] ----
        for (int idx = threadIdx.x; idx < TM * TK; idx += THREADS) {
            int tm, tk;
            if constexpr (!TRANS_A) {
                // A[m, k] — stride-1 in k → fast index = k
                tm = idx / TK;  tk = idx % TK;
            } else {
                // A[k, m] — stride-1 in m → fast index = m
                tk = idx / TM;  tm = idx % TM;
            }
            int m = m_start + tm;
            int k = k0 + tk;
            float val = 0.0f;
            if (m < M && tk < tile_k) {
                val = TRANS_A ? A[k * lda + m] : A[m * lda + k];
            }
            s_A[tm][tk] = val;
        }

        // ---- Load B tile: op(B)[k0..+TK, n_start..+TN] → s_B[tn][tk] ----
        for (int idx = threadIdx.x; idx < TK * TN; idx += THREADS) {
            int tk, tn;
            if constexpr (!TRANS_B) {
                // B[k, n] — stride-1 in n → fast index = n
                tk = idx / TN;  tn = idx % TN;
            } else {
                // B[n, k] — stride-1 in k → fast index = k
                tn = idx / TK;  tk = idx % TK;
            }
            int k = k0 + tk;
            int n = n_start + tn;
            float val = 0.0f;
            if (tk < tile_k && n < N) {
                val = TRANS_B ? B[n * ldb + k] : B[k * ldb + n];
            }
            s_B[tn][tk] = val;
        }
        __syncthreads();

        // ---- Partial-product accumulation with register tiling ----
        // Both s_A[m][tk] and s_B[n][tk] have tk as stride-1 dimension,
        // enabling the compiler to use vector loads for consecutive tk.
        #pragma unroll
        for (int tk = 0; tk < TK; tk++) {
            float b_val = s_B[local_n][tk];
            #pragma unroll
            for (int r = 0; r < REG_M; r++)
                acc[r] += s_A[local_m_base + r][tk] * b_val;
        }
        __syncthreads();
    }

    // ---- Epilogue: write results ----
    int n = n_start + local_n;
    #pragma unroll
    for (int r = 0; r < REG_M; r++) {
        int m = m_start + local_m_base + r;
        if (m < M && n < N) {
            float result = alpha * acc[r];
            if (bias) result += bias[m];
            if (beta != 0.0f) result += beta * C[m * ldc + n];
            C[m * ldc + n] = result;
        }
    }
}

// ============================================================
//  Dispatch helpers
// ============================================================

// Dispatch to the correct (TRANS_A, TRANS_B) instantiation for a given tile.
template <int TM, int TN, int TK, int REG_M>
static void dispatch_trans(
    Op transA, Op transB,
    int M, int N, int K,
    float alpha, const float* A, int lda,
    const float* B, int ldb,
    float beta, float* C, int ldc,
    const float* bias, cudaStream_t stream)
{
    constexpr int THREADS = (TM / REG_M) * TN;
    dim3 grid(internal::divup(M, TM), internal::divup(N, TN));

    bool tA = (transA == Op::T);
    bool tB = (transB == Op::T);

    #define FORGE_LAUNCH(TA, TB) \
        fp32_gemm_kernel<TA, TB, TM, TN, TK, REG_M> \
            <<<grid, THREADS, 0, stream>>>( \
                A, B, C, bias, M, N, K, alpha, beta, lda, ldb, ldc)

    if      (!tA && !tB) { FORGE_LAUNCH(false, false); }
    else if ( tA && !tB) { FORGE_LAUNCH(true,  false); }
    else if (!tA &&  tB) { FORGE_LAUNCH(false, true);  }
    else                 { FORGE_LAUNCH(true,  true);   }

    #undef FORGE_LAUNCH
}

// ============================================================
//  Public API
// ============================================================

void fp32_gemm(
    Op transA, Op transB,
    int M, int N, int K,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float beta,
    float* C, int ldc,
    const float* bias,
    cudaStream_t stream)
{
    if (M <= 0 || N <= 0 || K <= 0) return;

    // Auto-select tile configuration based on problem dimensions.
    //
    // Large-K profile (TM=16, TN=32, TK=64):
    //   When the reduction dimension K dwarfs M, a large TK amortizes
    //   __syncthreads() overhead and smaller TM keeps block count high.
    //   Example: backward act-grad with M=256, K=5000.
    //
    // Large-M profile (TM=64, TN=32, TK=32, 2x1 register tiling):
    //   When M is the dominant dimension, 2x1 register tiling lets each
    //   thread compute two M-rows while sharing the B-tile load.
    //   Example: forward readout or backward wt-grad with M=5000.
    //
    // Balanced profile (TM=32, TN=32, TK=32):
    //   General-purpose fallback for square-ish problems.

    if (K >= M * 4 && M <= 512) {
        dispatch_trans<16, 32, 64, 1>(
            transA, transB, M, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc, bias, stream);
    } else if (M >= 512) {
        dispatch_trans<64, 32, 32, 2>(
            transA, transB, M, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc, bias, stream);
    } else {
        dispatch_trans<32, 32, 32, 1>(
            transA, transB, M, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc, bias, stream);
    }
}

// ============================================================
//  Split-K GEMM kernel
// ============================================================
//
// Same tiling strategy as the standard kernel, but the K dimension
// is partitioned across blockIdx.z.  Each block computes partial
// dot products for its K-slice and writes raw (no alpha/beta/bias)
// results to a workspace buffer.  A second kernel reduces the
// partials into the final output.
//
// Use case: small-M problems (e.g. M=256, K=5000) where the
// standard grid has too few blocks for good SM utilisation.
// With K_splits=4, block count goes from 256 → 1024.

template <bool TRANS_A, bool TRANS_B, int TM, int TN, int TK, int REG_M = 1>
__global__ void fp32_gemm_splitk_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__       workspace,
    int M, int N, int K,
    int lda, int ldb,
    int K_per_split)
{
    constexpr int THREADS = (TM / REG_M) * TN;

    int m_start = blockIdx.x * TM;
    int n_start = blockIdx.y * TN;
    int k_split = blockIdx.z;

    int k_begin = k_split * K_per_split;
    int k_end   = min(k_begin + K_per_split, K);

    int local_m_base = (threadIdx.x / TN) * REG_M;
    int local_n      = threadIdx.x % TN;

    float acc[REG_M];
    #pragma unroll
    for (int r = 0; r < REG_M; r++) acc[r] = 0.0f;

    __shared__ float s_A[TM][TK + 1];
    __shared__ float s_B[TN][TK + 1];

    for (int k0 = k_begin; k0 < k_end; k0 += TK) {
        int tile_k = min(TK, k_end - k0);

        for (int idx = threadIdx.x; idx < TM * TK; idx += THREADS) {
            int tm, tk;
            if constexpr (!TRANS_A) { tm = idx / TK; tk = idx % TK; }
            else                    { tk = idx / TM; tm = idx % TM; }
            int m = m_start + tm;
            int k = k0 + tk;
            float val = 0.0f;
            if (m < M && tk < tile_k)
                val = TRANS_A ? A[k * lda + m] : A[m * lda + k];
            s_A[tm][tk] = val;
        }

        for (int idx = threadIdx.x; idx < TK * TN; idx += THREADS) {
            int tk, tn;
            if constexpr (!TRANS_B) { tk = idx / TN; tn = idx % TN; }
            else                    { tn = idx / TK; tk = idx % TK; }
            int k = k0 + tk;
            int n = n_start + tn;
            float val = 0.0f;
            if (tk < tile_k && n < N)
                val = TRANS_B ? B[n * ldb + k] : B[k * ldb + n];
            s_B[tn][tk] = val;
        }
        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < TK; tk++) {
            float b_val = s_B[local_n][tk];
            #pragma unroll
            for (int r = 0; r < REG_M; r++)
                acc[r] += s_A[local_m_base + r][tk] * b_val;
        }
        __syncthreads();
    }

    // Write partial result — layout: workspace[k_split * M * N + m * N + n]
    int n = n_start + local_n;
    #pragma unroll
    for (int r = 0; r < REG_M; r++) {
        int m = m_start + local_m_base + r;
        if (m < M && n < N)
            workspace[k_split * M * N + m * N + n] = acc[r];
    }
}

// Reduction: sum K_splits partials, apply alpha/beta/bias
__global__ void splitk_reduce_kernel(
    const float* __restrict__ workspace,
    float* __restrict__       C,
    const float* __restrict__ bias,
    int M, int N, int K_splits,
    float alpha, float beta, int ldc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = idx / N;
    int n = idx % N;
    if (m >= M) return;

    float sum = 0.0f;
    for (int s = 0; s < K_splits; s++)
        sum += workspace[s * M * N + m * N + n];

    float result = alpha * sum;
    if (bias) result += bias[m];
    if (beta != 0.0f) result += beta * C[m * ldc + n];
    C[m * ldc + n] = result;
}

// Dispatch helpers for split-K
template <int TM, int TN, int TK, int REG_M>
static void dispatch_splitk_trans(
    Op transA, Op transB,
    int M, int N, int K,
    float alpha, const float* A, int lda,
    const float* B, int ldb,
    float beta, float* C, int ldc,
    const float* bias, float* workspace, int K_splits,
    cudaStream_t stream)
{
    constexpr int THREADS = (TM / REG_M) * TN;
    int K_per_split = internal::divup(K, K_splits);
    dim3 grid(internal::divup(M, TM), internal::divup(N, TN), K_splits);

    bool tA = (transA == Op::T);
    bool tB = (transB == Op::T);

    #define FORGE_LAUNCH_SK(TA, TB) \
        fp32_gemm_splitk_kernel<TA, TB, TM, TN, TK, REG_M> \
            <<<grid, THREADS, 0, stream>>>( \
                A, B, workspace, M, N, K, lda, ldb, K_per_split)

    if      (!tA && !tB) { FORGE_LAUNCH_SK(false, false); }
    else if ( tA && !tB) { FORGE_LAUNCH_SK(true,  false); }
    else if (!tA &&  tB) { FORGE_LAUNCH_SK(false, true);  }
    else                 { FORGE_LAUNCH_SK(true,  true);   }

    #undef FORGE_LAUNCH_SK

    // Reduce partials → final output
    int total = M * N;
    int rblock = internal::kDefaultBlockSize;
    int rgrid  = internal::divup(total, rblock);
    splitk_reduce_kernel<<<rgrid, rblock, 0, stream>>>(
        workspace, C, bias, M, N, K_splits, alpha, beta, ldc);
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
    cudaStream_t stream)
{
    if (M <= 0 || N <= 0 || K <= 0 || K_splits <= 0) return;

    // Same tile selection as the standard GEMM
    if (K >= M * 4 && M <= 512) {
        dispatch_splitk_trans<16, 32, 64, 1>(
            transA, transB, M, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc, bias,
            workspace, K_splits, stream);
    } else if (M >= 512) {
        dispatch_splitk_trans<64, 32, 32, 2>(
            transA, transB, M, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc, bias,
            workspace, K_splits, stream);
    } else {
        dispatch_splitk_trans<32, 32, 32, 1>(
            transA, transB, M, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc, bias,
            workspace, K_splits, stream);
    }
}

}} // namespace forge::gemm
