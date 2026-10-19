// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Dense INT8 GEMM via __dp4a() intrinsics for sm_61+ (Pascal, Turing non-TC).
//
// Each __dp4a(a,b,c) does: c += dot(int8x4(a), int8x4(b)), giving 4 INT8 MACs
// per CUDA core per cycle. On GTX 1080 (2560 cores @ 1.7 GHz) this yields ~35 TOPS.
//
// Tile design:
//   BLOCK_M=64, BLOCK_K=64, BLOCK_N=64|128 (template)
//   256 threads, each computing a THREAD_M × THREAD_N output tile
//   Shared memory: 8-12 KB (high occupancy on Pascal's 48 KB limit)
//   128-bit (int4) vectorized global→smem loads
//   Fused dequant + bias epilogue (no staging buffer needed)

#include <forge/gemm/int8_dp4a.cuh>
#include <forge/quant/quantize.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace gemm {

static constexpr int DP4A_WIDE_N_THRESHOLD = 2048;

template <int BLOCK_N_T>
__global__ void int8_dp4a_gemm_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    const float*  __restrict__ act_scale,
    const __half* __restrict__ w_scale,
    const __half* __restrict__ bias,
    __half*       __restrict__ C,
    const int M, const int K, const int N)
{
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_K = 64;

    // Per-thread output tile: 4×4 (narrow) or 4×8 (wide)
    constexpr int THREAD_M = 4;
    constexpr int THREAD_N = (BLOCK_N_T <= 64) ? 4 : 8;
    constexpr int THREADS_ROW = BLOCK_M / THREAD_M;       // 16
    constexpr int THREADS_COL = BLOCK_N_T / THREAD_N;     // 16
    constexpr int NUM_THREADS = THREADS_ROW * THREADS_COL; // 256
    constexpr int K_STEPS = BLOCK_K / 4;                   // 16

    const int block_m = blockIdx.x * BLOCK_M;
    const int block_n = blockIdx.y * BLOCK_N_T;
    const int tid = threadIdx.x;
    const int thread_row = (tid / THREADS_COL) * THREAD_M;
    const int thread_col = (tid % THREADS_COL) * THREAD_N;

    __shared__ int8_t smem_A[BLOCK_M][BLOCK_K];
    __shared__ int8_t smem_B[BLOCK_N_T][BLOCK_K];

    // Per-thread accumulators (int32)
    int32_t acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++)
            acc[i][j] = 0;

    // Precompute whether K is nicely aligned for fast int4 loads.
    // When K % 16 != 0, row pointers are unaligned → use byte loads.
    const bool k_aligned = (K % 16 == 0);

    // K-loop over tiles
    for (int k_outer = 0; k_outer < K; k_outer += BLOCK_K) {
        // Load A tile: [BLOCK_M × BLOCK_K]
        {
            int4* dst = reinterpret_cast<int4*>(&smem_A[0][0]);
            constexpr int total = (BLOCK_M * BLOCK_K) / 16;
            #pragma unroll 4
            for (int i = tid; i < total; i += NUM_THREADS) {
                const int row = i / (BLOCK_K / 16);
                const int col4 = i % (BLOCK_K / 16);
                const int m_row = block_m + row;
                const int k_pos = k_outer + col4 * 16;
                if (m_row < M && k_pos < K) {
                    if (k_aligned && k_pos + 16 <= K) {
                        // Fast: aligned 128-bit load
                        const int4* src = reinterpret_cast<const int4*>(
                            A + (size_t)m_row * K + k_pos);
                        dst[i] = *src;
                    } else {
                        // Safe: byte-by-byte into aligned int4 local
                        int4 val = make_int4(0, 0, 0, 0);
                        int8_t* vp = reinterpret_cast<int8_t*>(&val);
                        const int8_t* sp = A + (size_t)m_row * K + k_pos;
                        const int cnt = min(16, K - k_pos);
                        for (int b = 0; b < cnt; b++) vp[b] = sp[b];
                        dst[i] = val;
                    }
                } else {
                    dst[i] = make_int4(0, 0, 0, 0);
                }
            }
        }

        // Load B tile: [BLOCK_N × BLOCK_K] (same logic)
        {
            int4* dst = reinterpret_cast<int4*>(&smem_B[0][0]);
            constexpr int total = (BLOCK_N_T * BLOCK_K) / 16;
            #pragma unroll 4
            for (int i = tid; i < total; i += NUM_THREADS) {
                const int row = i / (BLOCK_K / 16);
                const int col4 = i % (BLOCK_K / 16);
                const int n_row = block_n + row;
                const int k_pos = k_outer + col4 * 16;
                if (n_row < N && k_pos < K) {
                    if (k_aligned && k_pos + 16 <= K) {
                        const int4* src = reinterpret_cast<const int4*>(
                            B + (size_t)n_row * K + k_pos);
                        dst[i] = *src;
                    } else {
                        int4 val = make_int4(0, 0, 0, 0);
                        int8_t* vp = reinterpret_cast<int8_t*>(&val);
                        const int8_t* sp = B + (size_t)n_row * K + k_pos;
                        const int cnt = min(16, K - k_pos);
                        for (int b = 0; b < cnt; b++) vp[b] = sp[b];
                        dst[i] = val;
                    }
                } else {
                    dst[i] = make_int4(0, 0, 0, 0);
                }
            }
        }

        __syncthreads();

        // dp4a compute: each thread processes its THREAD_M × THREAD_N tile
        #pragma unroll
        for (int k = 0; k < K_STEPS; k++) {
            const int k_off = k * 4;

            // Load A fragment: THREAD_M packed int32 values
            int a_reg[THREAD_M];
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++)
                a_reg[i] = *reinterpret_cast<const int*>(
                    &smem_A[thread_row + i][k_off]);

            // Load B fragment: THREAD_N packed int32 values
            int b_reg[THREAD_N];
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++)
                b_reg[j] = *reinterpret_cast<const int*>(
                    &smem_B[thread_col + j][k_off]);

            // Outer product accumulation via dp4a
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++)
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++)
                    acc[i][j] = __dp4a(a_reg[i], b_reg[j], acc[i][j]);
        }

        __syncthreads();
    }

    // Epilogue: dequant + bias, write directly to global memory
    // (no staging buffer needed — each thread owns its accumulators)
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        const int gr = block_m + thread_row + i;
        if (gr >= M) continue;
        const float a_scale = act_scale[gr];

        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            const int gc = block_n + thread_col + j;
            if (gc < N) {
                float val = static_cast<float>(acc[i][j]);
                val *= a_scale * __half2float(w_scale[gc]);
                if (bias != nullptr) val += __half2float(bias[gc]);
                C[(size_t)gr * N + gc] = __float2half(val);
            }
        }
    }
}

// --- Launchers ---

void int8_dp4a_gemm_dequant(
    const int8_t* A,
    const int8_t* B,
    const float*  act_scale,
    const __half* w_scale,
    const __half* bias,
    __half*       output,
    int           M,
    int           K,
    int           N,
    cudaStream_t  stream)
{
    if (M <= 0 || K <= 0 || N <= 0) return;

    constexpr int BLOCK_M = 64;
    constexpr int NUM_THREADS = 256;

    if (N > DP4A_WIDE_N_THRESHOLD) {
        constexpr int BLOCK_N = 128;
        dim3 grid(internal::divup(M, BLOCK_M), internal::divup(N, BLOCK_N));
        int8_dp4a_gemm_kernel<128><<<grid, NUM_THREADS, 0, stream>>>(
            A, B, act_scale, w_scale, bias, output, M, K, N);
    } else {
        constexpr int BLOCK_N = 64;
        dim3 grid(internal::divup(M, BLOCK_M), internal::divup(N, BLOCK_N));
        int8_dp4a_gemm_kernel<64><<<grid, NUM_THREADS, 0, stream>>>(
            A, B, act_scale, w_scale, bias, output, M, K, N);
    }
}

void int8_dp4a_linear(
    const __half* x,
    const int8_t* w_int8,
    const __half* w_scale,
    const __half* bias,
    __half*       output,
    int           M,
    int           K,
    int           N,
    int8_t*       scratch_a_int8,
    float*        scratch_a_scale,
    cudaStream_t  stream)
{
    // Step 1: quantize activation (per-row absmax)
    forge::quant::fused_quantize_activation(
        x, scratch_a_int8, scratch_a_scale, M, K, stream);

    // Step 2: dp4a GEMM + dequant + bias
    int8_dp4a_gemm_dequant(
        scratch_a_int8, w_int8, scratch_a_scale, w_scale, bias, output,
        M, K, N, stream);
}

}} // namespace forge::gemm
