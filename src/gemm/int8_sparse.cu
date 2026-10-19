// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Sparse INT8 Tensor Core GEMM via PTX mma.sp instruction.
// Extracted from Sparsemma (int8_sparse_tc.cu).
//
// The core template (sparse_gemm_impl) and kernel wrappers are extracted
// verbatim from Sparsemma. Only the PyTorch wrapper layer has been replaced
// with a standalone C++ API.

#include <forge/gemm/int8_sparse.cuh>
#include <forge/quant/quantize.cuh>
#include "cuda_utils.cuh"
#include "cp_async.cuh"
#include <algorithm>

namespace forge { namespace gemm {

// Pull internal helpers into scope for kernel code
using internal::cp_async_16;
using internal::cp_async_16_stream;
using internal::cp_async_commit;
using internal::cp_async_wait_group;
using internal::a_swiz;
using internal::b_swiz;
using internal::divup;

// Fixed geometry constants
constexpr int SP_BLOCK_K  = 64;
constexpr int SP_K_COMP   = 32;
constexpr int SP_WARPS_WR = 4;
constexpr int SP_WARPS_BC = 2;
constexpr int SP_THREADS  = SP_WARPS_WR * SP_WARPS_BC * 32;  // 256
constexpr int SP_STAGE_PAD = 9;

constexpr int SP_WIDE_M_THRESHOLD = 2048;
constexpr int SP_TALL_N_THRESHOLD = 2048;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Split-K finalization
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__global__ void splitk_reduce_kernel(
    const float*  __restrict__ workspace,
    const __half* __restrict__ bias,
    __half*       __restrict__ output,
    const int N_out, const int total_elements)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float val = workspace[idx];
    if (bias != nullptr)
        val += __half2float(bias[idx % N_out]);
    output[idx] = __float2half(val);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Tiled Sparse INT8 GEMM core implementation (3-stage pipelined)
//
//  Computes: output(M, N_out) = dequant( W_comp(N_out, K/2) x x_int8(M, K)^T )
//
//  W_comp is the sparse operand A (row-major, 2:4 compressed along K).
//  x_int8 is the dense operand B (treated as col-major K×M = row-major M×K).
//  The MMA produces C(N_out, M). Epilogue transposes to output(M, N_out).
//
//  Template parameters:
//    BLOCK_WR_T: 128 (standard) or 256 (tall tile, higher arithmetic intensity)
//    BLOCK_BC_T: 64 (narrow) or 128 (wide) batch columns
//    DB_STAGES_T: pipeline depth (2, 3, or 4)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

template <int BLOCK_WR_T, int BLOCK_BC_T, int DB_STAGES_T>
__device__ __forceinline__ void sparse_gemm_impl(
    const int8_t*   __restrict__ W_comp,     // (N_out, K/2) compressed weight
    const int8_t*   __restrict__ x_int8,     // (M, K) quantized activation
    const uint32_t* __restrict__ w_meta,     // (N_out, K/32) packed metadata
    const float*    __restrict__ act_scale,  // (M,) fp32
    const __half*   __restrict__ w_scale,    // (N_out,) fp16
    const __half*   __restrict__ bias,       // (N_out,) fp16 or nullptr
    __half*         __restrict__ output,     // (M, N_out) fp16
    float*          __restrict__ splitk_ws,  // split-K fp32 workspace, or nullptr
    const int M, const int K, const int N_out,
    const int split_k,             // 1 = normal, >1 = split-K mode
    const int k_iters_per_split    // K iterations per split (ceiling)
) {
    // ── Derived constants from template parameters ──
    constexpr int WARP_WR = BLOCK_WR_T / SP_WARPS_WR;       // 32 or 64
    constexpr int WARP_BC = BLOCK_BC_T / SP_WARPS_BC;        // 32 or 64
    constexpr int TILES_M = WARP_WR / 16;                    // 2 or 4 MMA M-tiles per warp
    constexpr int TILES_N = WARP_BC / 8;                     // 4 or 8
    constexpr int DB_STAGES = DB_STAGES_T;

    // Pipelined smem: DB_STAGES x (A_tile + B_tile)
    constexpr int SMEM_A_STAGE = BLOCK_WR_T * SP_K_COMP;             // 4096 or 8192
    constexpr int SMEM_B_STAGE = BLOCK_BC_T * SP_BLOCK_K;            // 4096 or 8192
    constexpr int SMEM_PER_STAGE = SMEM_A_STAGE + SMEM_B_STAGE;
    constexpr int SMEM_DB = DB_STAGES * SMEM_PER_STAGE;
    // Epilogue staging (reuses smem after K-loop)
    constexpr int SMEM_EPILOGUE = 8 * 16 * SP_STAGE_PAD * (int)sizeof(int32_t);
    constexpr int SMEM_BYTES = (SMEM_DB > SMEM_EPILOGUE) ? SMEM_DB : SMEM_EPILOGUE;

    // ── Thread-local constants (invariant across tiles) ──
    const int t    = threadIdx.x;
    const int warp = t / 32;
    const int lane = t % 32;
    const int groupID = lane >> 2;
    const int tid     = lane & 3;

    const int warp_wr = (warp / SP_WARPS_BC) * WARP_WR;   // 0, WARP_WR, 2*WARP_WR, ...
    const int warp_bc = (warp % SP_WARPS_BC) * WARP_BC;    // 0, WARP_BC

    const int K_comp     = K / 2;
    const int meta_cols  = K / 32;
    const int K_iters_total = K / SP_BLOCK_K;

    // ── Persistent kernel tile grid ──
    const int tiles_x = (N_out + BLOCK_WR_T - 1) / BLOCK_WR_T;
    const int tiles_y = (M     + BLOCK_BC_T - 1) / BLOCK_BC_T;
    const int spatial_tiles = tiles_x * tiles_y;
    const int total_tiles = spatial_tiles * split_k;

    // L2 swizzle: group SWIZZLE_W N-tiles, sweep M-tiles column-major
    constexpr int SWIZZLE_W = 4;

    // ── Shared memory ──
    extern __shared__ int8_t smem_raw[];

    #define SMEM_A(stage) reinterpret_cast<int8_t (*)[SP_K_COMP]>( \
        smem_raw + (stage) * SMEM_PER_STAGE)
    #define SMEM_B(stage) reinterpret_cast<int8_t (*)[SP_BLOCK_K]>( \
        smem_raw + (stage) * SMEM_PER_STAGE + SMEM_A_STAGE)

    // ── Persistent tile loop ──
    const int tiles_per_cta = (total_tiles + (int)gridDim.x - 1) / (int)gridDim.x;
    const int tile_start = blockIdx.x * tiles_per_cta;
    const int tile_end   = min(tile_start + tiles_per_cta, total_tiles);

    for (int tile_id = tile_start; tile_id < tile_end; tile_id++) {

    const int k_split_id = tile_id / spatial_tiles;
    const int spatial_id = tile_id % spatial_tiles;

    // ── L2 CTA swizzle: spatial_id -> (tile_n, tile_m) ──
    int tile_n, tile_m;
    if (tiles_x >= 2 * SWIZZLE_W) {
        const int group_tiles = SWIZZLE_W * tiles_y;
        const int group_id    = spatial_id / group_tiles;
        const int within      = spatial_id % group_tiles;
        tile_n = (within / tiles_y) + group_id * SWIZZLE_W;
        tile_m = within % tiles_y;
        if (tile_n >= tiles_x) continue;
    } else {
        tile_n = spatial_id % tiles_x;
        tile_m = spatial_id / tiles_x;
    }

    // ── K-range for this split ──
    const int k_start_iter  = k_split_id * k_iters_per_split;
    const int k_end_iter    = min(k_start_iter + k_iters_per_split, K_iters_total);
    const int local_K_iters = k_end_iter - k_start_iter;
    if (local_K_iters <= 0) { __syncthreads(); continue; }

    const int block_wr = tile_n * BLOCK_WR_T;
    const int block_bc = tile_m * BLOCK_BC_T;

    // ── Accumulators: TILES_M M-tiles x TILES_N N-tiles x 4 int32 ──
    int32_t acc[TILES_M][TILES_N][4];
    #pragma unroll
    for (int mi = 0; mi < TILES_M; mi++)
        #pragma unroll
        for (int nj = 0; nj < TILES_N; nj++) {
            acc[mi][nj][0] = 0; acc[mi][nj][1] = 0;
            acc[mi][nj][2] = 0; acc[mi][nj][3] = 0;
        }

    // ── 3-stage pipelined K-loop with cp.async ──

    constexpr int TOTAL_A_16 = BLOCK_WR_T * SP_K_COMP / 16;
    constexpr int TOTAL_B_16 = BLOCK_BC_T * SP_BLOCK_K / 16;

    #define LOAD_TILE(STAGE, K_DENSE_OFF) do {                                \
        const int _k_comp_off = (K_DENSE_OFF) / 2;                           \
        auto _sA = SMEM_A(STAGE);                                            \
        auto _sB = SMEM_B(STAGE);                                            \
        for (int _i = t; _i < TOTAL_A_16; _i += SP_THREADS) {               \
            const int _row  = _i / (SP_K_COMP / 16);                         \
            const int _col4 = _i % (SP_K_COMP / 16);                         \
            const int _gr = block_wr + _row;                                  \
            const int _sc = a_swiz(_row, _col4 * 16);                        \
            if (_gr < N_out)                                                  \
                cp_async_16(&_sA[_row][_sc],                                  \
                    W_comp + (size_t)_gr * K_comp + _k_comp_off + _col4*16);  \
            else                                                              \
                *reinterpret_cast<int4*>(&_sA[_row][_sc])                     \
                    = make_int4(0,0,0,0);                                     \
        }                                                                     \
        for (int _i = t; _i < TOTAL_B_16; _i += SP_THREADS) {               \
            const int _row  = _i / (SP_BLOCK_K / 16);                        \
            const int _col4 = _i % (SP_BLOCK_K / 16);                        \
            const int _gr = block_bc + _row;                                  \
            const int _sc = b_swiz(_row, _col4 * 16);                        \
            if (_gr < M)                                                      \
                cp_async_16_stream(&_sB[_row][_sc],                           \
                    x_int8 + (size_t)_gr * K + (K_DENSE_OFF) + _col4 * 16);  \
            else                                                              \
                *reinterpret_cast<int4*>(&_sB[_row][_sc])                     \
                    = make_int4(0,0,0,0);                                     \
        }                                                                     \
        cp_async_commit();                                                    \
    } while (0)

    // ── Prologue: async load first (DB_STAGES-1) tiles ──
    #pragma unroll
    for (int s = 0; s < DB_STAGES - 1 && s < local_K_iters; s++)
        LOAD_TILE(s, (k_start_iter + s) * SP_BLOCK_K);

    // ── Main K-loop ──
    for (int ki = 0; ki < local_K_iters; ki++) {
        const int k_abs      = k_start_iter + ki;
        const int k_dense    = k_abs * SP_BLOCK_K;
        const int buf_comp   = ki % DB_STAGES;
        const int k_ahead    = ki + (DB_STAGES - 1);
        const bool has_ahead = (k_ahead < local_K_iters);

        if (has_ahead)
            cp_async_wait_group<DB_STAGES - 2>();
        else
            cp_async_wait_group<0>();
        __syncthreads();

        if (has_ahead)
            LOAD_TILE(k_ahead % DB_STAGES, (k_start_iter + k_ahead) * SP_BLOCK_K);

        auto cA = SMEM_A(buf_comp);
        auto cB = SMEM_B(buf_comp);

        // ── Metadata prefetch ──
        const int k_meta0 = k_dense / 32;
        const int k_meta1 = k_meta0 + 1;

        uint32_t e_meta_prefetch[TILES_M];
        #pragma unroll
        for (int mi = 0; mi < TILES_M; mi++) {
            const int m_local  = warp_wr + mi * 16;
            const int m_global = block_wr + m_local;
            uint32_t e;
            if (tid == 0) {
                e = (m_global + groupID < N_out)
                    ? w_meta[(size_t)(m_global + groupID) * meta_cols + k_meta0]
                    : 0x44444444u;
            } else if (tid == 1) {
                e = (m_global + groupID + 8 < N_out)
                    ? w_meta[(size_t)(m_global + groupID + 8) * meta_cols + k_meta0]
                    : 0x44444444u;
            } else if (tid == 2) {
                e = (m_global + groupID < N_out)
                    ? w_meta[(size_t)(m_global + groupID) * meta_cols + k_meta1]
                    : 0x44444444u;
            } else {
                e = (m_global + groupID + 8 < N_out)
                    ? w_meta[(size_t)(m_global + groupID + 8) * meta_cols + k_meta1]
                    : 0x44444444u;
            }
            e_meta_prefetch[mi] = e;
        }

        // ── Per-warp MMA computation ──
        #pragma unroll
        for (int mi = 0; mi < TILES_M; mi++) {
            const int m_local = warp_wr + mi * 16;
            const uint32_t e_meta = e_meta_prefetch[mi];

            const int a_ldm_row = m_local + (lane & 15);
            uint32_t a_frag0[2], a_frag1[2];

            // First K-half (spsel=0): K_comp[0:15]
            {
                uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(
                    &cA[a_ldm_row][a_swiz(a_ldm_row, 0)]));
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(a_frag0[0]), "=r"(a_frag0[1])
                    : "r"(a_addr)
                );
            }
            // Second K-half (spsel=1): K_comp[16:31]
            {
                uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(
                    &cA[a_ldm_row][a_swiz(a_ldm_row, 16)]));
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(a_frag1[0]), "=r"(a_frag1[1])
                    : "r"(a_addr)
                );
            }

            #pragma unroll
            for (int nj = 0; nj < TILES_N; nj++) {
                const int b_ldm_n = warp_bc + nj * 8 + (lane & 7);
                const int b_k_half = ((lane >> 3) & 1) << 4;  // 0 or 16

                uint32_t b_frag0[2];
                {
                    uint32_t b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(
                        &cB[b_ldm_n][b_swiz(b_ldm_n, b_k_half)]));
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_frag0[0]), "=r"(b_frag0[1])
                        : "r"(b_addr)
                    );
                }

                // MMA 1: spsel=0, K[0:31]
                asm volatile(
                    "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5}, "
                    "{%6, %7}, "
                    "{%8, %9, %10, %11}, "
                    "%12, 0x0;\n"
                    : "=r"(acc[mi][nj][0]), "=r"(acc[mi][nj][1]),
                      "=r"(acc[mi][nj][2]), "=r"(acc[mi][nj][3])
                    : "r"(a_frag0[0]), "r"(a_frag0[1]),
                      "r"(b_frag0[0]), "r"(b_frag0[1]),
                      "r"(acc[mi][nj][0]), "r"(acc[mi][nj][1]),
                      "r"(acc[mi][nj][2]), "r"(acc[mi][nj][3]),
                      "r"(e_meta)
                );

                uint32_t b_frag1[2];
                {
                    uint32_t b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(
                        &cB[b_ldm_n][b_swiz(b_ldm_n, 32 + b_k_half)]));
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_frag1[0]), "=r"(b_frag1[1])
                        : "r"(b_addr)
                    );
                }

                // MMA 2: spsel=1, K[32:63]
                asm volatile(
                    "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5}, "
                    "{%6, %7}, "
                    "{%8, %9, %10, %11}, "
                    "%12, 0x1;\n"
                    : "=r"(acc[mi][nj][0]), "=r"(acc[mi][nj][1]),
                      "=r"(acc[mi][nj][2]), "=r"(acc[mi][nj][3])
                    : "r"(a_frag1[0]), "r"(a_frag1[1]),
                      "r"(b_frag1[0]), "r"(b_frag1[1]),
                      "r"(acc[mi][nj][0]), "r"(acc[mi][nj][1]),
                      "r"(acc[mi][nj][2]), "r"(acc[mi][nj][3]),
                      "r"(e_meta)
                );
            }
        }
    }

    __syncthreads();

    // ── Epilogue: dequant + bias + transposed write via smem staging ──
    int32_t (*smem_stage)[16][SP_STAGE_PAD] =
        reinterpret_cast<int32_t (*)[16][SP_STAGE_PAD]>(smem_raw);

    #pragma unroll
    for (int mi = 0; mi < TILES_M; mi++) {
        #pragma unroll
        for (int nj = 0; nj < TILES_N; nj++) {
            smem_stage[warp][groupID    ][tid * 2]     = acc[mi][nj][0];
            smem_stage[warp][groupID    ][tid * 2 + 1] = acc[mi][nj][1];
            smem_stage[warp][groupID + 8][tid * 2]     = acc[mi][nj][2];
            smem_stage[warp][groupID + 8][tid * 2 + 1] = acc[mi][nj][3];
            __syncwarp();

            const int tile_wr = block_wr + warp_wr + mi * 16;
            const int tile_bc = block_bc + warp_bc + nj * 8;

            #pragma unroll
            for (int i = lane; i < 128; i += 32) {
                const int bc_local = i / 16;
                const int wr_local = i % 16;
                const int wr = tile_wr + wr_local;
                const int bc = tile_bc + bc_local;

                if (wr < N_out && bc < M) {
                    float val = static_cast<float>(smem_stage[warp][wr_local][bc_local]);
                    val *= act_scale[bc] * __half2float(w_scale[wr]);
                    if (splitk_ws != nullptr) {
                        atomicAdd(&splitk_ws[(size_t)bc * N_out + wr], val);
                    } else {
                        if (bias != nullptr)
                            val += __half2float(bias[wr]);
                        output[(size_t)bc * N_out + wr] = __float2half(val);
                    }
                }
            }
            __syncwarp();
        }
    }

    __syncthreads();
    } // end persistent tile loop

    #undef LOAD_TILE
    #undef SMEM_A
    #undef SMEM_B
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Kernel wrappers with different launch_bounds
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#define SPARSE_GEMM_PARAMS                                          \
    const int8_t*   __restrict__ W_comp,                            \
    const int8_t*   __restrict__ x_int8,                            \
    const uint32_t* __restrict__ w_meta,                            \
    const float*    __restrict__ act_scale,                         \
    const __half*   __restrict__ w_scale,                           \
    const __half*   __restrict__ bias,                              \
    __half*         __restrict__ output,                            \
    float*          __restrict__ splitk_ws,                         \
    const int M, const int K, const int N_out,                     \
    const int split_k, const int k_iters_per_split

#define SPARSE_GEMM_ARGS \
    W_comp, x_int8, w_meta, act_scale, w_scale, bias, output, \
    splitk_ws, M, K, N_out, split_k, k_iters_per_split

// Occupancy-1 kernel (original behavior, deep pipeline).
template <int BLOCK_WR_T, int BLOCK_BC_T>
__global__ void sparse_gemm_kernel(SPARSE_GEMM_PARAMS) {
    constexpr int PER_STAGE = BLOCK_WR_T * SP_K_COMP + BLOCK_BC_T * SP_BLOCK_K;
    constexpr int STAGES = (PER_STAGE * 4 <= 48 * 1024) ? 4 : 3;
    sparse_gemm_impl<BLOCK_WR_T, BLOCK_BC_T, STAGES>(SPARSE_GEMM_ARGS);
}

// Occupancy-2 kernel: __launch_bounds__(256, 2) forces <=128 regs/thread.
template <int BLOCK_WR_T, int BLOCK_BC_T>
__global__ void __launch_bounds__(SP_THREADS, 2)
sparse_gemm_occ2_kernel(SPARSE_GEMM_PARAMS) {
    constexpr int PER_STAGE = BLOCK_WR_T * SP_K_COMP + BLOCK_BC_T * SP_BLOCK_K;
    constexpr int STAGES = (PER_STAGE * 4 <= 32 * 1024) ? 4
                         : (PER_STAGE * 3 <= 32 * 1024) ? 3 : 2;
    sparse_gemm_impl<BLOCK_WR_T, BLOCK_BC_T, STAGES>(SPARSE_GEMM_ARGS);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Warp-Specialized Sparse GEMM Kernel
//
//  Splits warps into 2 producer (cp.async loads) and 6 consumer (MMA compute).
//  Producer/consumer sync via shared memory flags + named barriers.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

constexpr int WS_NUM_PROD_WARPS = 2;
constexpr int WS_NUM_CONS_WARPS = 6;
constexpr int WS_PROD_THREADS = WS_NUM_PROD_WARPS * 32;   // 64
constexpr int WS_CONS_THREADS = WS_NUM_CONS_WARPS * 32;    // 192
constexpr int WS_CONS_WARPS_WR = 3;
constexpr int WS_CONS_WARPS_BC = 2;

template <int BLOCK_WR_T, int BLOCK_BC_T>
__global__ void sparse_gemm_warpspec_kernel(
    const int8_t*   __restrict__ W_comp,
    const int8_t*   __restrict__ x_int8,
    const uint32_t* __restrict__ w_meta,
    const float*    __restrict__ act_scale,
    const __half*   __restrict__ w_scale,
    const __half*   __restrict__ bias,
    __half*         __restrict__ output,
    float*          __restrict__ splitk_ws,
    const int M, const int K, const int N_out,
    const int split_k,
    const int k_iters_per_split
) {
    // ── Consumer warp-grid derived constants ──
    constexpr int WARP_WR = BLOCK_WR_T / WS_CONS_WARPS_WR;
    constexpr int WARP_BC = BLOCK_BC_T / WS_CONS_WARPS_BC;
    constexpr int TILES_M = WARP_WR / 16;
    constexpr int TILES_N = WARP_BC / 8;

    constexpr int SMEM_PER_STAGE_RAW = BLOCK_WR_T * SP_K_COMP + BLOCK_BC_T * SP_BLOCK_K;
    constexpr int DB_STAGES = (SMEM_PER_STAGE_RAW * 4 <= 48 * 1024) ? 4 : 3;
    constexpr int SMEM_A_STAGE = BLOCK_WR_T * SP_K_COMP;
    constexpr int SMEM_B_STAGE = BLOCK_BC_T * SP_BLOCK_K;
    constexpr int SMEM_PER_STAGE = SMEM_A_STAGE + SMEM_B_STAGE;
    constexpr int SMEM_DB = DB_STAGES * SMEM_PER_STAGE;
    constexpr int WS_SMEM_EPILOGUE = WS_NUM_CONS_WARPS * 16 * SP_STAGE_PAD * (int)sizeof(int32_t);
    constexpr int SMEM_BYTES = (SMEM_DB > WS_SMEM_EPILOGUE) ? SMEM_DB : WS_SMEM_EPILOGUE;

    constexpr int TOTAL_A_16 = BLOCK_WR_T * SP_K_COMP / 16;
    constexpr int TOTAL_B_16 = BLOCK_BC_T * SP_BLOCK_K / 16;

    const int t        = threadIdx.x;
    const int warp     = t / 32;
    const int lane     = t % 32;
    const int groupID  = lane >> 2;
    const int tid      = lane & 3;

    const bool is_producer = (warp < WS_NUM_PROD_WARPS);
    const int cons_warp    = warp - WS_NUM_PROD_WARPS;
    const int c_warp_wr    = (cons_warp / WS_CONS_WARPS_BC) * WARP_WR;
    const int c_warp_bc    = (cons_warp % WS_CONS_WARPS_BC) * WARP_BC;

    const int K_comp        = K / 2;
    const int meta_cols     = K / 32;
    const int K_iters_total = K / SP_BLOCK_K;

    const int tiles_x       = (N_out + BLOCK_WR_T - 1) / BLOCK_WR_T;
    const int tiles_y       = (M     + BLOCK_BC_T - 1) / BLOCK_BC_T;
    const int spatial_tiles = tiles_x * tiles_y;
    const int total_tiles   = spatial_tiles * split_k;
    constexpr int SWIZZLE_W = 4;

    extern __shared__ int8_t smem_raw[];
    __shared__ volatile int ws_data_ready[DB_STAGES];
    __shared__ volatile int ws_buf_consumed[DB_STAGES];

    #define SMEM_A(stage) reinterpret_cast<int8_t (*)[SP_K_COMP]>( \
        smem_raw + (stage) * SMEM_PER_STAGE)
    #define SMEM_B(stage) reinterpret_cast<int8_t (*)[SP_BLOCK_K]>( \
        smem_raw + (stage) * SMEM_PER_STAGE + SMEM_A_STAGE)

    #define LOAD_TILE_PROD(STAGE, K_DENSE_OFF) do {                                \
        const int _k_comp_off = (K_DENSE_OFF) / 2;                                \
        auto _sA = SMEM_A(STAGE);                                                 \
        auto _sB = SMEM_B(STAGE);                                                 \
        for (int _i = t; _i < TOTAL_A_16; _i += WS_PROD_THREADS) {               \
            const int _row  = _i / (SP_K_COMP / 16);                              \
            const int _col4 = _i % (SP_K_COMP / 16);                              \
            const int _gr = block_wr + _row;                                       \
            const int _sc = a_swiz(_row, _col4 * 16);                             \
            if (_gr < N_out)                                                       \
                cp_async_16(&_sA[_row][_sc],                                       \
                    W_comp + (size_t)_gr * K_comp + _k_comp_off + _col4*16);       \
            else                                                                   \
                *reinterpret_cast<int4*>(&_sA[_row][_sc]) = make_int4(0,0,0,0);   \
        }                                                                          \
        for (int _i = t; _i < TOTAL_B_16; _i += WS_PROD_THREADS) {               \
            const int _row  = _i / (SP_BLOCK_K / 16);                             \
            const int _col4 = _i % (SP_BLOCK_K / 16);                             \
            const int _gr = block_bc + _row;                                       \
            const int _sc = b_swiz(_row, _col4 * 16);                             \
            if (_gr < M)                                                           \
                cp_async_16_stream(&_sB[_row][_sc],                                \
                    x_int8 + (size_t)_gr * K + (K_DENSE_OFF) + _col4 * 16);       \
            else                                                                   \
                *reinterpret_cast<int4*>(&_sB[_row][_sc]) = make_int4(0,0,0,0);   \
        }                                                                          \
        cp_async_commit();                                                         \
    } while (0)

    // ── Persistent tile loop ──
    const int tiles_per_cta = (total_tiles + (int)gridDim.x - 1) / (int)gridDim.x;
    const int tile_start    = blockIdx.x * tiles_per_cta;
    const int tile_end      = min(tile_start + tiles_per_cta, total_tiles);

    for (int tile_id = tile_start; tile_id < tile_end; tile_id++) {

    const int k_split_id = tile_id / spatial_tiles;
    const int spatial_id = tile_id % spatial_tiles;

    int tile_n, tile_m;
    if (tiles_x >= 2 * SWIZZLE_W) {
        const int group_tiles = SWIZZLE_W * tiles_y;
        const int group_id    = spatial_id / group_tiles;
        const int within      = spatial_id % group_tiles;
        tile_n = (within / tiles_y) + group_id * SWIZZLE_W;
        tile_m = within % tiles_y;
        if (tile_n >= tiles_x) continue;
    } else {
        tile_n = spatial_id % tiles_x;
        tile_m = spatial_id / tiles_x;
    }

    const int k_start_iter  = k_split_id * k_iters_per_split;
    const int k_end_iter    = min(k_start_iter + k_iters_per_split, K_iters_total);
    const int local_K_iters = k_end_iter - k_start_iter;
    if (local_K_iters <= 0) { __syncthreads(); continue; }

    const int block_wr = tile_n * BLOCK_WR_T;
    const int block_bc = tile_m * BLOCK_BC_T;

    // ── Initialize producer/consumer flags ──
    if (t < DB_STAGES) {
        ws_data_ready[t]   = 0;
        ws_buf_consumed[t] = DB_STAGES;
    }
    __syncthreads();

    // ══════════════════════════════════════════════════════════════
    //  PRODUCER WARPS (warps 0-1): load tiles via cp.async
    // ══════════════════════════════════════════════════════════════
    if (is_producer) {

        for (int ki = 0; ki < local_K_iters; ki++) {
            const int stage       = ki % DB_STAGES;
            const int k_dense_off = (k_start_iter + ki) * SP_BLOCK_K;

            if (ki >= DB_STAGES) {
                if (lane == 0 && warp == 0)
                    while (ws_buf_consumed[stage] < ki) {}
                asm volatile("bar.sync 1, %0;" :: "r"(WS_PROD_THREADS));
            }

            LOAD_TILE_PROD(stage, k_dense_off);

            cp_async_wait_group<0>();
            asm volatile("bar.sync 1, %0;" :: "r"(WS_PROD_THREADS));

            __threadfence_block();
            if (t == 0) ws_data_ready[stage] = ki + 1;
        }

    }

    // ══════════════════════════════════════════════════════════════
    //  CONSUMER WARPS (warps 2-7): MMA computation + epilogue
    // ══════════════════════════════════════════════════════════════
    if (!is_producer) {

        int32_t acc[TILES_M][TILES_N][4];
        #pragma unroll
        for (int mi = 0; mi < TILES_M; mi++)
            #pragma unroll
            for (int nj = 0; nj < TILES_N; nj++) {
                acc[mi][nj][0] = 0; acc[mi][nj][1] = 0;
                acc[mi][nj][2] = 0; acc[mi][nj][3] = 0;
            }

        for (int ki = 0; ki < local_K_iters; ki++) {
            const int k_abs   = k_start_iter + ki;
            const int k_dense = k_abs * SP_BLOCK_K;
            const int stage   = ki % DB_STAGES;

            if (lane == 0 && cons_warp == 0)
                while (ws_data_ready[stage] < ki + 1) {}
            asm volatile("bar.sync 2, %0;" :: "r"(WS_CONS_THREADS));
            __threadfence_block();

            auto cA = SMEM_A(stage);
            auto cB = SMEM_B(stage);

            const int k_meta0 = k_dense / 32;
            const int k_meta1 = k_meta0 + 1;

            uint32_t e_meta_prefetch[TILES_M];
            #pragma unroll
            for (int mi = 0; mi < TILES_M; mi++) {
                const int m_local  = c_warp_wr + mi * 16;
                const int m_global = block_wr + m_local;
                uint32_t e;
                if (tid == 0) {
                    e = (m_global + groupID < N_out)
                        ? w_meta[(size_t)(m_global + groupID) * meta_cols + k_meta0]
                        : 0x44444444u;
                } else if (tid == 1) {
                    e = (m_global + groupID + 8 < N_out)
                        ? w_meta[(size_t)(m_global + groupID + 8) * meta_cols + k_meta0]
                        : 0x44444444u;
                } else if (tid == 2) {
                    e = (m_global + groupID < N_out)
                        ? w_meta[(size_t)(m_global + groupID) * meta_cols + k_meta1]
                        : 0x44444444u;
                } else {
                    e = (m_global + groupID + 8 < N_out)
                        ? w_meta[(size_t)(m_global + groupID + 8) * meta_cols + k_meta1]
                        : 0x44444444u;
                }
                e_meta_prefetch[mi] = e;
            }

            #pragma unroll
            for (int mi = 0; mi < TILES_M; mi++) {
                const int m_local = c_warp_wr + mi * 16;
                const uint32_t e_meta = e_meta_prefetch[mi];

                const int a_row0 = m_local + groupID;
                const int a_row1 = m_local + groupID + 8;
                uint32_t a_frag0[2], a_frag1[2];
                a_frag0[0] = *reinterpret_cast<const uint32_t*>(
                    &cA[a_row0][a_swiz(a_row0, tid * 4)]);
                a_frag0[1] = *reinterpret_cast<const uint32_t*>(
                    &cA[a_row1][a_swiz(a_row1, tid * 4)]);
                a_frag1[0] = *reinterpret_cast<const uint32_t*>(
                    &cA[a_row0][a_swiz(a_row0, 16 + tid * 4)]);
                a_frag1[1] = *reinterpret_cast<const uint32_t*>(
                    &cA[a_row1][a_swiz(a_row1, 16 + tid * 4)]);

                #pragma unroll
                for (int nj = 0; nj < TILES_N; nj++) {
                    const int n_local = c_warp_bc + nj * 8 + groupID;

                    uint32_t b_frag0[2];
                    b_frag0[0] = *reinterpret_cast<const uint32_t*>(
                        &cB[n_local][b_swiz(n_local, tid * 4)]);
                    b_frag0[1] = *reinterpret_cast<const uint32_t*>(
                        &cB[n_local][b_swiz(n_local, 16 + tid * 4)]);

                    // MMA 1: spsel=0, K[0:31]
                    asm volatile(
                        "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5}, "
                        "{%6, %7}, "
                        "{%8, %9, %10, %11}, "
                        "%12, 0x0;\n"
                        : "=r"(acc[mi][nj][0]), "=r"(acc[mi][nj][1]),
                          "=r"(acc[mi][nj][2]), "=r"(acc[mi][nj][3])
                        : "r"(a_frag0[0]), "r"(a_frag0[1]),
                          "r"(b_frag0[0]), "r"(b_frag0[1]),
                          "r"(acc[mi][nj][0]), "r"(acc[mi][nj][1]),
                          "r"(acc[mi][nj][2]), "r"(acc[mi][nj][3]),
                          "r"(e_meta)
                    );

                    uint32_t b_frag1[2];
                    b_frag1[0] = *reinterpret_cast<const uint32_t*>(
                        &cB[n_local][b_swiz(n_local, 32 + tid * 4)]);
                    b_frag1[1] = *reinterpret_cast<const uint32_t*>(
                        &cB[n_local][b_swiz(n_local, 48 + tid * 4)]);

                    // MMA 2: spsel=1, K[32:63]
                    asm volatile(
                        "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5}, "
                        "{%6, %7}, "
                        "{%8, %9, %10, %11}, "
                        "%12, 0x1;\n"
                        : "=r"(acc[mi][nj][0]), "=r"(acc[mi][nj][1]),
                          "=r"(acc[mi][nj][2]), "=r"(acc[mi][nj][3])
                        : "r"(a_frag1[0]), "r"(a_frag1[1]),
                          "r"(b_frag1[0]), "r"(b_frag1[1]),
                          "r"(acc[mi][nj][0]), "r"(acc[mi][nj][1]),
                          "r"(acc[mi][nj][2]), "r"(acc[mi][nj][3]),
                          "r"(e_meta)
                    );
                }
            }

            asm volatile("bar.sync 2, %0;" :: "r"(WS_CONS_THREADS));
            __threadfence_block();
            if (t == WS_PROD_THREADS)
                ws_buf_consumed[stage] = ki + DB_STAGES;
        }

        // ── Epilogue: dequant + bias + transposed write ──
        int32_t (*smem_stage)[16][SP_STAGE_PAD] =
            reinterpret_cast<int32_t (*)[16][SP_STAGE_PAD]>(smem_raw);

        #pragma unroll
        for (int mi = 0; mi < TILES_M; mi++) {
            #pragma unroll
            for (int nj = 0; nj < TILES_N; nj++) {
                smem_stage[cons_warp][groupID    ][tid * 2]     = acc[mi][nj][0];
                smem_stage[cons_warp][groupID    ][tid * 2 + 1] = acc[mi][nj][1];
                smem_stage[cons_warp][groupID + 8][tid * 2]     = acc[mi][nj][2];
                smem_stage[cons_warp][groupID + 8][tid * 2 + 1] = acc[mi][nj][3];
                __syncwarp();

                const int tile_wr = block_wr + c_warp_wr + mi * 16;
                const int tile_bc = block_bc + c_warp_bc + nj * 8;

                #pragma unroll
                for (int i = lane; i < 128; i += 32) {
                    const int bc_local = i / 16;
                    const int wr_local = i % 16;
                    const int wr = tile_wr + wr_local;
                    const int bc = tile_bc + bc_local;

                    if (wr < N_out && bc < M) {
                        float val = static_cast<float>(smem_stage[cons_warp][wr_local][bc_local]);
                        val *= act_scale[bc] * __half2float(w_scale[wr]);
                        if (splitk_ws != nullptr) {
                            atomicAdd(&splitk_ws[(size_t)bc * N_out + wr], val);
                        } else {
                            if (bias != nullptr)
                                val += __half2float(bias[wr]);
                            output[(size_t)bc * N_out + wr] = __float2half(val);
                        }
                    }
                }
                __syncwarp();
            }
        }

    } // end consumer

    __syncthreads();
    } // end persistent tile loop

    #undef LOAD_TILE_PROD
    #undef SMEM_A
    #undef SMEM_B
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Shared memory size helpers (host-side)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

template <int BW, int BC>
constexpr int sparse_gemm_smem_bytes() {
    constexpr int PER_STAGE = BW * SP_K_COMP + BC * SP_BLOCK_K;
    constexpr int STAGES = (PER_STAGE * 4 <= 48 * 1024) ? 4 : 3;
    constexpr int SMEM_DB = STAGES * PER_STAGE;
    constexpr int SMEM_EPI = 8 * 16 * SP_STAGE_PAD * (int)sizeof(int32_t);
    return (SMEM_DB > SMEM_EPI) ? SMEM_DB : SMEM_EPI;
}

template <int BW, int BC>
constexpr int warpspec_smem_bytes() {
    constexpr int PER_STAGE = BW * SP_K_COMP + BC * SP_BLOCK_K;
    constexpr int STAGES = (PER_STAGE * 4 <= 100 * 1024) ? 4 : 3;
    constexpr int SMEM_DB = STAGES * PER_STAGE;
    constexpr int SMEM_EPI = WS_NUM_CONS_WARPS * 16 * SP_STAGE_PAD * (int)sizeof(int32_t);
    return (SMEM_DB > SMEM_EPI) ? SMEM_DB : SMEM_EPI;
}

template <int BW, int BC>
constexpr int occ2_smem_bytes() {
    constexpr int PER_STAGE = BW * SP_K_COMP + BC * SP_BLOCK_K;
    constexpr int STAGES = (PER_STAGE * 4 <= 32 * 1024) ? 4
                         : (PER_STAGE * 3 <= 32 * 1024) ? 3 : 2;
    constexpr int SMEM_DB = STAGES * PER_STAGE;
    constexpr int SMEM_EPI = 8 * 16 * SP_STAGE_PAD * (int)sizeof(int32_t);
    return (SMEM_DB > SMEM_EPI) ? SMEM_DB : SMEM_EPI;
}

#undef SPARSE_GEMM_PARAMS
#undef SPARSE_GEMM_ARGS

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Launchers (standalone C++ API)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void splitk_reduce(
    const float*  workspace,
    const __half* bias,
    __half*       output,
    int           M,
    int           N_out,
    cudaStream_t  stream)
{
    if (M <= 0 || N_out <= 0) return;
    int total = M * N_out;
    int block = 256;
    int grid  = divup(total, block);
    splitk_reduce_kernel<<<grid, block, 0, stream>>>(workspace, bias, output, N_out, total);
}

void sparse_int8_gemm(
    const int8_t*   W_comp,
    const int8_t*   x_int8,
    const uint32_t* w_meta,
    const float*    act_scale,
    const __half*   w_scale,
    const __half*   bias,
    __half*         output,
    int             M,
    int             K,
    int             N_out,
    float*          splitk_workspace,
    int             split_k,
    cudaStream_t    stream)
{
    if (M <= 0 || K <= 0 || N_out <= 0) return;

    constexpr int SP_MAX_CTAS = 128;  // RTX 4090 has 128 SMs
    constexpr int SP_SPLITK_TILE_THRESHOLD = 64;
    constexpr int SP_MAX_SPLIT_K = 32;
    constexpr int SP_MIN_K_PER_SPLIT = 4;

    // Determine tile shape
    int BW, BC;
    if (M > SP_WIDE_M_THRESHOLD && N_out >= SP_TALL_N_THRESHOLD) {
        BW = 256; BC = 128;
    } else if (M > SP_WIDE_M_THRESHOLD) {
        BW = 128; BC = 128;
    } else {
        BW = 128; BC = 64;
    }

    const int spatial_tiles = ((N_out + BW - 1) / BW) * ((M + BC - 1) / BC);
    const int k_iters_total = K / SP_BLOCK_K;

    // Compute split-K factor if caller passed split_k <= 0 (auto)
    if (split_k <= 0) {
        split_k = 1;
        if (spatial_tiles < SP_SPLITK_TILE_THRESHOLD && k_iters_total >= 8) {
            split_k = (SP_MAX_CTAS + spatial_tiles - 1) / spatial_tiles;
            const int max_by_k = k_iters_total / SP_MIN_K_PER_SPLIT;
            if (split_k > max_by_k) split_k = max_by_k;
            if (split_k > SP_MAX_SPLIT_K) split_k = SP_MAX_SPLIT_K;
            if (split_k < 1) split_k = 1;
        }
    }

    const int k_iters_per_split = (k_iters_total + split_k - 1) / split_k;

    #define LAUNCH_SPARSE_GEMM(BWv, BCv) do {                                       \
        constexpr int _smem = sparse_gemm_smem_bytes<(BWv), (BCv)>();              \
        if (_smem > 48 * 1024) {                                                   \
            cudaFuncSetAttribute(sparse_gemm_kernel<(BWv), (BCv)>,                 \
                cudaFuncAttributeMaxDynamicSharedMemorySize, _smem);                \
        }                                                                          \
        const int _total = spatial_tiles * split_k;                                \
        const int _nctas = (_total < SP_MAX_CTAS) ? _total : SP_MAX_CTAS;        \
        sparse_gemm_kernel<(BWv), (BCv)><<<_nctas, SP_THREADS, _smem, stream>>>(  \
            W_comp, x_int8, w_meta, act_scale, w_scale, bias, output,             \
            splitk_workspace, M, K, N_out, split_k, k_iters_per_split             \
        );                                                                         \
    } while (0)

    if (BW == 256) {
        LAUNCH_SPARSE_GEMM(256, 128);
    } else if (BC == 128) {
        LAUNCH_SPARSE_GEMM(128, 128);
    } else {
        LAUNCH_SPARSE_GEMM(128, 64);
    }

    #undef LAUNCH_SPARSE_GEMM

    // Split-K finalization
    if (split_k > 1 && splitk_workspace != nullptr) {
        splitk_reduce(splitk_workspace, bias, output, M, N_out, stream);
    }
}

void sparse_int8_linear(
    const __half*   x,
    const int8_t*   W_comp,
    const uint32_t* w_meta,
    const __half*   w_scale,
    const __half*   bias,
    __half*         output,
    int             M,
    int             K,
    int             N_out,
    int8_t*         scratch_x_int8,
    float*          scratch_x_scale,
    float*          splitk_workspace,
    int             split_k,
    cudaStream_t    stream)
{
    // Step 1: quantize activation
    forge::quant::fused_quantize_activation(x, scratch_x_int8, scratch_x_scale, M, K, stream);

    // Step 2: sparse GEMM
    sparse_int8_gemm(W_comp, scratch_x_int8, w_meta, scratch_x_scale,
                     w_scale, bias, output, M, K, N_out,
                     splitk_workspace, split_k, stream);
}

}} // namespace forge::gemm
