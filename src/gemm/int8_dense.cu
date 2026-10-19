// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Dense INT8 Tensor Core GEMM via nvcuda::wmma intrinsics.
// Extracted from Sparsemma (int8_gemm_tc.cu).
// Requires sm_80+ (Ampere) for 16x16x16 INT8 wmma.
#include <forge/gemm/int8_dense.cuh>
#include <forge/quant/quantize.cuh>
#include <algorithm>
#include "cuda_utils.cuh"

// wmma intrinsics require sm_80+ for INT8 16x16x16.
// Guard so this file compiles cleanly for sm_61/sm_75 targets.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#include <mma.h>
using namespace nvcuda;
#endif

namespace forge { namespace gemm {

// N threshold: use wide (128x128) tile when N > this value.
static constexpr int WIDE_N_THRESHOLD = 2048;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

template <int BLOCK_N_T>
__global__ void int8_gemm_dequant_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    const float*  __restrict__ act_scale,
    const __half* __restrict__ w_scale,
    const __half* __restrict__ bias,
    __half*       __restrict__ C,
    const int M, const int K, const int N)
{
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_K = 64;
    constexpr int WARPS_M = 4;
    constexpr int WARPS_N = 2;
    constexpr int WARP_M = BLOCK_M / WARPS_M;      // 32
    constexpr int WARP_N = BLOCK_N_T / WARPS_N;    // 32 or 64
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int TILES_M = WARP_M / WMMA_M;       // 2
    constexpr int TILES_N = WARP_N / WMMA_N;       // 2 or 4
    constexpr int K_STEPS = BLOCK_K / WMMA_K;      // 4
    constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;  // 256

    const int block_m = blockIdx.x * BLOCK_M;
    const int block_n = blockIdx.y * BLOCK_N_T;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_row = (warp_id / WARPS_N) * WARP_M;
    const int warp_col = (warp_id % WARPS_N) * WARP_N;

    __shared__ int8_t smem_A[BLOCK_M][BLOCK_K];
    __shared__ int8_t smem_B[BLOCK_N_T][BLOCK_K];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc[TILES_M][TILES_N];
    #pragma unroll
    for (int i = 0; i < TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < TILES_N; j++)
            wmma::fill_fragment(acc[i][j], 0);

    const int tid = threadIdx.x;

    // When K % 16 != 0, stride-based int4 loads cross row boundaries.
    // Use row-addressed pointers with byte-by-byte fallback for safety.
    const bool k_aligned = (K % 16 == 0);

    for (int k_outer = 0; k_outer < K; k_outer += BLOCK_K) {
        // Load A tile via 128-bit (int4) loads
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
                        const int4* src = reinterpret_cast<const int4*>(
                            A + (size_t)m_row * K + k_pos);
                        dst[i] = *src;
                    } else {
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

        // Load B tile
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

        #pragma unroll
        for (int k_inner = 0; k_inner < K_STEPS; k_inner++) {
            const int k_offset = k_inner * WMMA_K;
            #pragma unroll
            for (int wi = 0; wi < TILES_M; wi++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag,
                    &smem_A[warp_row + wi * WMMA_M][k_offset], BLOCK_K);
                #pragma unroll
                for (int wj = 0; wj < TILES_N; wj++) {
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(b_frag,
                        &smem_B[warp_col + wj * WMMA_N][k_offset], BLOCK_K);
                    wmma::mma_sync(acc[wi][wj], a_frag, b_frag, acc[wi][wj]);
                }
            }
        }

        __syncthreads();
    }

    // Epilogue: dequant + bias via per-warp staging buffer (reuses smem_A)
    int32_t (*smem_stage)[16][16] = reinterpret_cast<int32_t (*)[16][16]>(&smem_A[0][0]);
    const int half_warp = lane_id / 16;
    const int col_in_half = lane_id % 16;

    #pragma unroll
    for (int wi = 0; wi < TILES_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < TILES_N; wj++) {
            wmma::store_matrix_sync(
                &smem_stage[warp_id][0][0],
                acc[wi][wj], 16, wmma::mem_row_major);
            #pragma unroll
            for (int r = 0; r < 16; r += 2) {
                const int local_row = r + half_warp;
                const int gr = block_m + warp_row + wi * WMMA_M + local_row;
                const int gc = block_n + warp_col + wj * WMMA_N + col_in_half;
                if (gr < M && gc < N) {
                    float val = static_cast<float>(smem_stage[warp_id][local_row][col_in_half]);
                    val *= act_scale[gr] * __half2float(w_scale[gc]);
                    if (bias != nullptr) val += __half2float(bias[gc]);
                    C[(size_t)gr * N + gc] = __float2half(val);
                }
            }
        }
    }
}

// --- Launchers (also guarded: <<<>>> generates device code referencing kernel) ---

void int8_gemm_dequant(
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

    constexpr int BLOCK_M = 128;
    constexpr int NUM_THREADS = 256;

    if (N > WIDE_N_THRESHOLD) {
        constexpr int BLOCK_N = 128;
        dim3 grid(internal::divup(M, BLOCK_M), internal::divup(N, BLOCK_N));
        int8_gemm_dequant_kernel<128><<<grid, NUM_THREADS, 0, stream>>>(
            A, B, act_scale, w_scale, bias, output, M, K, N);
    } else {
        constexpr int BLOCK_N = 64;
        dim3 grid(internal::divup(M, BLOCK_M), internal::divup(N, BLOCK_N));
        int8_gemm_dequant_kernel<64><<<grid, NUM_THREADS, 0, stream>>>(
            A, B, act_scale, w_scale, bias, output, M, K, N);
    }
}

void int8_linear(
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
    // Step 1: quantize activation
    forge::quant::fused_quantize_activation(x, scratch_a_int8, scratch_a_scale, M, K, stream);

    // Step 2: INT8 TC GEMM + dequant + bias
    int8_gemm_dequant(scratch_a_int8, w_int8, scratch_a_scale, w_scale, bias, output,
                      M, K, N, stream);
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

}} // namespace forge::gemm
