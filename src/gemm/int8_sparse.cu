// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Sparse INT8 Tensor Core GEMM via PTX mma.sp instruction.
// Extracted from Sparsemma (int8_sparse_tc.cu), including the 2026-10-19 optimization
// campaign (sparsemma 333910e..338133d):
//   - one mma.sp m16n8k64 per 16x8 tile per 64-deep K step; A and B fragments loaded with
//     ldmatrix.x4, each B fragment loaded once per K step and reused across the warp's M tiles
//   - sparsity metadata staged through the cp.async pipeline into shared memory (branch-free,
//     conflict-free LDS); 4 K steps per copy for the large (L2-bound) tiles
//   - per-thread cp.async / ldmatrix addresses computed once per tile (OOB rows clamped)
//   - register epilogue (dequant + movmatrix transpose), staged through shared memory for
//     16-byte global stores
//   - one persistent CTA per SM (dynamic smem request > half the SM), 3 pipeline stages,
//     M-fastest tile order
//   - tile and split-K chosen by a wave-aware cost model
// and the 2026-10-20 narrow-N follow-up (sparsemma, fused quantize):
//   - N-fastest tile order when the int8 activation exceeds L2 (read once, not per column)
//   - fused quantize + GEMM for K <= 1280 (resident int8 activation block, L2 prefetch of
//     the next block), picked by sparse_int8_linear when it saves DRAM traffic
//
// The core template (sparse_gemm_impl) is kept identical to Sparsemma; only the PyTorch
// wrapper layer has been replaced with a standalone C++ API.

#include <forge/gemm/int8_sparse.cuh>
#include <forge/quant/quantize.cuh>
#include "cuda_utils.cuh"
#include "cp_async.cuh"
#include <algorithm>

namespace forge { namespace gemm {

// Pull internal helpers into scope for kernel code
using internal::cp_async_16_u32;
using internal::cp_async_16_stream_u32;
using internal::cp_async_8_u32;
using internal::cp_async_commit;
using internal::cp_async_wait_group;
using internal::a_swiz;
using internal::b_swiz;
using internal::divup;

// Fixed geometry constants
constexpr int SP_BLOCK_K  = 64;     // dense K elements per iteration (one mma.sp k64)
constexpr int SP_K_COMP   = 32;     // compressed K per iteration (SP_BLOCK_K / 2)
constexpr int SP_WARPS_WR = 4;
constexpr int SP_WARPS_BC = 2;
constexpr int SP_THREADS  = SP_WARPS_WR * SP_WARPS_BC * 32;  // 256

// Sparsity metadata staging (2 uint32 per weight row per 64-deep K step):
//   META_STEPS = 1: 8 bytes per row in every pipeline stage. 8 useful bytes per 32-byte
//                   L2 sector, but the cheapest loop; best for the small 128x64 tile, which
//                   is barrier/latency-bound rather than L2-bound.
//   META_STEPS = 4: 32 bytes (4 K steps) per row into a 2-slot ring every 4th K step: full
//                   sectors, ~25% less L2 traffic for the large, L2-bound tiles.
// FQ (fused quantize, see sp_fq_base_bytes) 128x64: 2-step chunks (16 B/row, 2 stages ahead
// need META_STEPS >= 2); its K loop streams only weights, so metadata sectors matter more.
template <int BW, int BC, bool FQ = false>
constexpr int sp_meta_steps() { return (BC == 64) ? (FQ ? 2 : 1) : 4; }

// Pipeline depth per tile (A+B per stage: 16 / 12 / 8 KB). 3 stages measured best for all
// tiles: deeper pipelines enlarge the smem carve-out and shrink L1, and with one CTA per SM
// and an L2-bound loop 3 stages already cover the latency.
template <int BW, int BC>
constexpr int sp_stages() { return 3; }

// Fused-quantize (FQ) path layout: [A+metadata pipeline | epilogue staging] (the larger of
// the two), then the resident int8 activation block (BC rows x full K, stored as K steps in
// the pipeline's B-stage layout), then BC fp32 per-token scales.
template <int BW, int BC>
constexpr int sp_fq_base_bytes() {
    constexpr int MS   = sp_meta_steps<BW, BC, true>();
    constexpr int PIPE = sp_stages<BW, BC>() * (BW * SP_K_COMP + (MS == 1 ? BW * 8 : 0))
                       + (MS == 1 ? 0 : 2 * BW * MS * 8);
    constexpr int STG  = BC * BW * 2;
    return PIPE > STG ? PIPE : STG;
}
// Register-staged row quantize: each lane holds FQ_VPT 16-byte vectors of a row
// (K <= FQ_VPT * 256 = 1280).
template <int BC>
constexpr int sp_fq_vpt() { return 5; }

template <int BW, int BC, int STAGES>
constexpr int sp_smem_bytes() {
    constexpr int MS = sp_meta_steps<BW, BC>();
    constexpr int PER_STAGE = BW * SP_K_COMP + BC * SP_BLOCK_K + (MS == 1 ? BW * 8 : 0);
    return STAGES * PER_STAGE + (MS == 1 ? 0 : 2 * BW * MS * 8);
}

template <int BW, int BC>
constexpr int sparse_gemm_smem_bytes() {
    return sp_smem_bytes<BW, BC, sp_stages<BW, BC>()>();
}

// ===========================================================================
//  Split-K finalization
// ===========================================================================

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

// ===========================================================================
//  Tiled Sparse INT8 GEMM core implementation (3-stage cp.async pipeline)
//
//  Computes: output(M, N_out) = dequant( W_comp(N_out, K/2) x x_int8(M, K)^T )
//
//  W_comp is the sparse operand A (row-major, 2:4 compressed along K).
//  x_int8 is the dense operand B (treated as col-major KxM = row-major MxK).
//  The MMA produces C(N_out, M). Epilogue transposes to output(M, N_out).
//
//  Template parameters:
//    BLOCK_WR_T: 128 or 256 weight rows per tile
//    BLOCK_BC_T: 64 or 128 batch columns (tokens) per tile
//    DB_STAGES_T: pipeline depth
//  Persistent CTAs (one per SM) walk tiles in M-fastest order (N-fastest once the int8
//  activation exceeds L2); tile shape and split-K come from the cost model in
//  sparse_int8_gemm_plan.
//
//  FQ = true (sparse_gemm_fq_kernel, 128x64 only): fused activation quantize. B is not
//  pipelined: each CTA quantizes its 64-token block (full K <= 1280) from fp16 into a
//  resident smem block once, runs every weight tile against it (N-fastest), and
//  prefetches its next block into L2 meanwhile. Removes the int8 activation's DRAM
//  round trip on large-M, narrow-N layers (see sp_use_fused).
// ===========================================================================

// NF: N-fastest tile order (FQ always; the two-kernel GEMM when the int8 activation
// exceeds L2). A template parameter so the M-fastest kernels compile exactly as before.
template <int BLOCK_WR_T, int BLOCK_BC_T, int DB_STAGES_T, bool FQ = false, bool NF = FQ>
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
    const int k_iters_per_split,   // K iterations per split (ceiling)
    const __half* __restrict__ x_fp16,  // FQ: (M, K) fp16 activation, quantized in-kernel
    const int x_stream                  // FQ: evict-first loads of x_fp16 (input > L2)
) {
    // -- Derived constants from template parameters --
    constexpr int WARP_WR = BLOCK_WR_T / SP_WARPS_WR;       // 32 or 64
    constexpr int WARP_BC = BLOCK_BC_T / SP_WARPS_BC;        // 32 or 64
    constexpr int TILES_M = WARP_WR / 16;                    // 2 or 4 MMA M-tiles per warp
    constexpr int TILES_N = WARP_BC / 8;                     // 4 or 8
    constexpr int DB_STAGES = DB_STAGES_T;

    // Pipelined smem: DB_STAGES x (A_tile + B_tile), then a 2-slot metadata ring
    constexpr int SMEM_A_STAGE = BLOCK_WR_T * SP_K_COMP;             // 4096 or 8192
    constexpr int SMEM_B_STAGE = FQ ? 0 : BLOCK_BC_T * SP_BLOCK_K;   // 4096 or 8192 (FQ: resident)
    // Sparsity metadata: see sp_meta_steps (per-stage slot, or a 2-slot 4-step ring)
    constexpr int META_STEPS   = sp_meta_steps<BLOCK_WR_T, BLOCK_BC_T, FQ>();
    // Ring slot swizzle: rows 16/META_STEPS apart share banks; XOR their 8-byte step slot
    constexpr int MQ_SHIFT     = (META_STEPS == 2) ? 3 : 2;
    constexpr int SMEM_M_STAGE = (META_STEPS == 1) ? BLOCK_WR_T * 8 : 0;
    constexpr int SMEM_PER_STAGE = SMEM_A_STAGE + SMEM_B_STAGE + SMEM_M_STAGE;
    constexpr int SMEM_M_SLOT  = (META_STEPS == 1) ? 0 : BLOCK_WR_T * META_STEPS * 8;
    constexpr int SMEM_M_OFF   = DB_STAGES * SMEM_PER_STAGE;
    constexpr int SMEM_DB      = SMEM_M_OFF + 2 * SMEM_M_SLOT;
    static_assert(FQ || SMEM_DB == sp_smem_bytes<BLOCK_WR_T, BLOCK_BC_T, DB_STAGES>(), "smem layout");
    // FQ: resident activation block offset (after the pipeline / epilogue staging region)
    constexpr int SMEM_FQ_B = FQ ? sp_fq_base_bytes<BLOCK_WR_T, BLOCK_BC_T>() : 0;
    static_assert(!FQ || SMEM_FQ_B >= SMEM_DB, "FQ layout");
    static_assert(META_STEPS == 1 || META_STEPS >= DB_STAGES - 1,
                  "2-slot metadata ring needs META_STEPS >= DB_STAGES - 1");

    // -- Thread-local constants (invariant across tiles) --
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

    // -- Persistent kernel tile grid --
    // 1D grid with block-partitioned assignment: contiguous tiles per CTA
    // for maximum L2 cache reuse of weight (A) tiles.
    // When split_k > 1, the tile space is (spatial_tiles x split_k) -- each
    // (spatial_tile, k_split) pair is a separate work item.
    const int tiles_x = (N_out + BLOCK_WR_T - 1) / BLOCK_WR_T;
    const int tiles_y = (M     + BLOCK_BC_T - 1) / BLOCK_BC_T;
    const int spatial_tiles = tiles_x * tiles_y;
    const int total_tiles = spatial_tiles * split_k;

    // -- Shared memory (persists across tile iterations) --
    // Dynamic smem allows > 48KB (needed for 4-stage 256x128 = 64KB)
    extern __shared__ int8_t smem_raw[];

    const uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));

    // ==================================================================
    //  Persistent tile loop: each CTA processes multiple tiles
    //  Block-partitioned: CTA i gets tiles [i*tpc, (i+1)*tpc)
    //  to keep weight tiles hot in L2 across consecutive M-tiles.
    // ==================================================================
    const int tiles_per_cta = (total_tiles + (int)gridDim.x - 1) / (int)gridDim.x;
    const int tile_start = blockIdx.x * tiles_per_cta;
    const int tile_end   = min(tile_start + tiles_per_cta, total_tiles);

    [[maybe_unused]] int res_m = -1;   // FQ: M-tile whose activation block is resident in smem

    for (int tile_id = tile_start; tile_id < tile_end; tile_id++) {

    // -- Decode tile_id -> (spatial_id, k_split_id) --
    // K-splits for the same spatial tile are grouped together so the
    // persistent kernel's block-partition keeps weight tiles in L2.
    const int k_split_id = tile_id / spatial_tiles;
    const int spatial_id = tile_id % spatial_tiles;

    // -- Tile order: spatial_id -> (tile_n, tile_m) --
    // M-fastest: consecutive tiles of a CTA (block-partitioned) share the weight tile,
    // and co-running CTAs mostly share weight rows. Measured faster than N-fastest and
    // than banded/grouped orders on the headline shapes.
    // N-fastest (FQ always; two-kernel path when the int8 activation exceeds L2): the CTA
    // runs every weight tile against one activation block before moving on, so the
    // activation is read from DRAM once instead of once per weight-tile column.
    const int tile_n = NF ? spatial_id % tiles_x : spatial_id / tiles_y;
    const int tile_m = NF ? spatial_id / tiles_x : spatial_id % tiles_y;

    // -- K-range for this split --
    const int k_start_iter  = k_split_id * k_iters_per_split;
    const int k_end_iter    = min(k_start_iter + k_iters_per_split, K_iters_total);
    const int local_K_iters = k_end_iter - k_start_iter;
    if (local_K_iters <= 0) { __syncthreads(); continue; }

    const int block_wr = tile_n * BLOCK_WR_T;
    const int block_bc = tile_m * BLOCK_BC_T;

    // ================================================================
    //  FQ: quantize this M-tile's activation rows into the resident smem block
    //  (once per M-tile; consecutive tiles of a CTA share it, N-fastest order).
    //  Each warp quantizes BLOCK_BC_T/8 rows, two at a time: a lane holds up to
    //  FQ_VPT 16-byte vectors of each row in registers, the warp reduces the row
    //  absmax with shuffles, then stores 8 int8 per vector in the B-stage layout of
    //  K step k/64. Arithmetic matches _sp_quantize_vec_kernel (bit-identical).
    //  The previous tile's trailing __syncthreads() protects the overwrite.
    // ================================================================
    if constexpr (FQ) {
        if (tile_m != res_m) {
            res_m = tile_m;
            constexpr int FQ_VPT = sp_fq_vpt<BLOCK_BC_T>();
            constexpr int ROWS_PER_WARP = BLOCK_BC_T / (SP_THREADS / 32);
            const int nvec = K / 8;
            float* s_scale = reinterpret_cast<float*>(smem_raw + SMEM_FQ_B + BLOCK_BC_T * K);
            #pragma unroll 1
            for (int rr = 0; rr < ROWS_PER_WARP; rr += 2) {
                uint4 v[2][FQ_VPT];
                float amax[2];
                #pragma unroll
                for (int u = 0; u < 2; u++) {
                    const int r = warp * ROWS_PER_WARP + rr + u;
                    const uint4* src = reinterpret_cast<const uint4*>(
                        x_fp16 + (size_t)min(block_bc + r, M - 1) * K);
                    amax[u] = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < FQ_VPT; i++) {
                        const int idx = lane + 32 * i;
                        if (idx < nvec) {
                            v[u][i] = x_stream ? __ldcs(src + idx) : src[idx];
                            const __half2* h = reinterpret_cast<const __half2*>(&v[u][i]);
                            #pragma unroll
                            for (int j = 0; j < 4; j++) {
                                const float2 f = __half22float2(h[j]);
                                amax[u] = fmaxf(amax[u], fmaxf(fabsf(f.x), fabsf(f.y)));
                            }
                        }
                    }
                }
                #pragma unroll
                for (int u = 0; u < 2; u++) {
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1)
                        amax[u] = fmaxf(amax[u], __shfl_xor_sync(0xffffffff, amax[u], o));
                    const int r = warp * ROWS_PER_WARP + rr + u;
                    const float scale = fmaxf(amax[u], 1e-8f) / 127.0f;
                    const float inv_scale = 1.0f / scale;
                    if (lane == 0) s_scale[r] = scale;
                    #pragma unroll
                    for (int i = 0; i < FQ_VPT; i++) {
                        const int idx = lane + 32 * i;
                        if (idx < nvec) {
                            const __half2* h = reinterpret_cast<const __half2*>(&v[u][i]);
                            uint32_t packed[2] = {0u, 0u};
                            #pragma unroll
                            for (int j = 0; j < 4; j++) {
                                const float2 f = __half22float2(h[j]);
                                const int q0 = max(-128, min(127, __float2int_rn(f.x * inv_scale)));
                                const int q1 = max(-128, min(127, __float2int_rn(f.y * inv_scale)));
                                packed[j >> 1] |= ((uint32_t)(q0 & 0xFF) << (16 * (j & 1)))
                                                | ((uint32_t)(q1 & 0xFF) << (16 * (j & 1) + 8));
                            }
                            const int k = idx * 8;
                            const uint32_t a = smem_base + SMEM_FQ_B
                                + (uint32_t)((k / SP_BLOCK_K) * (BLOCK_BC_T * SP_BLOCK_K)
                                             + r * SP_BLOCK_K + b_swiz(r, k % SP_BLOCK_K));
                            asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n"
                                         :: "r"(a), "r"(packed[0]), "r"(packed[1]));
                        }
                    }
                }
            }
            __syncthreads();
            // Prefetch the CTA's next M-tile (contiguous rows of x) into L2 now, so its DRAM
            // read overlaps this M-tile's K loops and its quantize phase above hits L2.
            const int nm = tile_m + 1;
            if (nm < tiles_y && nm * tiles_x < tile_end) {
                const char* pf_base = reinterpret_cast<const char*>(x_fp16 + (size_t)nm * BLOCK_BC_T * K);
                const int pf_lines = min(BLOCK_BC_T, M - nm * BLOCK_BC_T) * (K / 64);   // 128-byte lines
                for (int l = t; l < pf_lines; l += SP_THREADS)
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(pf_base + (size_t)l * 128));
                // Measured on 1280x320, M=147456: no prefetch 0.774 ms, this 0.689; issuing it
                // before the quantize phase 0.793, two M-tiles ahead 0.779, evict_last 0.721.
            }
        }
    }

    // -- Accumulators: TILES_M M-tiles x TILES_N N-tiles x 4 int32 --
    int32_t acc[TILES_M][TILES_N][4];
    #pragma unroll
    for (int mi = 0; mi < TILES_M; mi++)
        #pragma unroll
        for (int nj = 0; nj < TILES_N; nj++) {
            acc[mi][nj][0] = 0; acc[mi][nj][1] = 0;
            acc[mi][nj][2] = 0; acc[mi][nj][3] = 0;
        }

    // ================================================================
    //  3-stage pipelined K-loop with cp.async
    //
    //  Prologue: async load tiles into buf[0] and buf[1].
    //  Main loop: while computing from buf[ki%3], async load into
    //             buf[(ki+2)%3]. Two tiles always prefetched ahead.
    //  Epilogue staging reuses smem after K-loop completes.
    // ================================================================

    // -- Per-thread cp.async assignments, fixed for the whole tile --
    // Each thread copies A_ITERS 16-byte A chunks, B_ITERS 16-byte B chunks and (if
    // t < BLOCK_WR_T) one 8-byte metadata pair per K step. Source pointers and swizzled
    // smem offsets are computed once here; per K step only the K offset changes.
    // Rows past N_out / M are clamped to the last valid row instead of zero-filled:
    // they only feed output rows/columns the epilogue never stores, and clamped
    // metadata is still a valid 2:4 pattern.
    constexpr int TOTAL_A_16 = BLOCK_WR_T * SP_K_COMP / 16;
    constexpr int TOTAL_B_16 = BLOCK_BC_T * SP_BLOCK_K / 16;
    constexpr int A_CPR = SP_K_COMP / 16;                  // 16B chunks per A row (2)
    constexpr int B_CPR = SP_BLOCK_K / 16;                 // 16B chunks per B row (4)
    constexpr int A_ITERS = TOTAL_A_16 / SP_THREADS;
    constexpr int B_ITERS = TOTAL_B_16 / SP_THREADS;
    static_assert(TOTAL_A_16 % SP_THREADS == 0 && TOTAL_B_16 % SP_THREADS == 0,
                  "tile loads must divide evenly across the CTA");
    static_assert(BLOCK_WR_T <= SP_THREADS, "one metadata row per thread");

    const int8_t* a_src[A_ITERS];
    uint32_t      a_dst[A_ITERS];
    #pragma unroll
    for (int it = 0; it < A_ITERS; it++) {
        const int row = t / A_CPR + it * (SP_THREADS / A_CPR);
        const int col = (t % A_CPR) * 16;
        a_src[it] = W_comp + (size_t)min(block_wr + row, N_out - 1) * K_comp + col;
        a_dst[it] = row * SP_K_COMP + a_swiz(row, col);
    }
    [[maybe_unused]] const int8_t* b_src[B_ITERS];   // unused by FQ (B is resident)
    [[maybe_unused]] uint32_t      b_dst[B_ITERS];
    #pragma unroll
    for (int it = 0; it < B_ITERS; it++) {
        const int row = t / B_CPR + it * (SP_THREADS / B_CPR);
        const int col = (t % B_CPR) * 16;
        b_src[it] = x_int8 + (size_t)min(block_bc + row, M - 1) * K + col;
        b_dst[it] = SMEM_A_STAGE + row * SP_BLOCK_K + b_swiz(row, col);
    }
    // Metadata chunk: a row's META_STEPS x 8 bytes are copied by META_STEPS consecutive
    // threads (one K step each), so each row is one coalesced request. Within the row the
    // 8-byte step slot is XOR-swizzled by (row >> 2) & 3 so compute-side reads (16 rows x
    // 2 words per warp) are bank-conflict-free.
    constexpr int M_COPIES = (BLOCK_WR_T * META_STEPS + SP_THREADS - 1) / SP_THREADS;
    static_assert(META_STEPS == 1 || BLOCK_WR_T * META_STEPS % SP_THREADS == 0,
                  "metadata copies must divide evenly");
    const bool m_loader = (t < BLOCK_WR_T * META_STEPS);
    const int  m_j = t % META_STEPS;                      // K step within the chunk
    const uint32_t* m_src[M_COPIES];
    uint32_t        m_dst[M_COPIES];
    #pragma unroll
    for (int it = 0; it < M_COPIES; it++) {
        const int row = t / META_STEPS + it * (SP_THREADS / META_STEPS);
        m_src[it] = w_meta + (size_t)min(block_wr + row, N_out - 1) * meta_cols + 2 * m_j;
        m_dst[it] = (META_STEPS == 1)
            ? (uint32_t)(SMEM_A_STAGE + SMEM_B_STAGE + row * 8)            // stage-relative
            : (uint32_t)(row * (META_STEPS * 8)
                         + ((m_j ^ ((row >> MQ_SHIFT) & (META_STEPS - 1))) << 3));
    }

    // Async load of absolute K step K_ABS into the stage at smem byte offset STAGE_OFF.
    #define LOAD_TILE(STAGE_OFF, K_ABS) do {                                      \
        const uint32_t _so = smem_base + (STAGE_OFF);                             \
        _Pragma("unroll")                                                          \
        for (int _it = 0; _it < A_ITERS; _it++)                                   \
            cp_async_16_u32(_so + a_dst[_it], a_src[_it] + (K_ABS) * SP_K_COMP);  \
        if constexpr (!FQ) {                                                       \
            _Pragma("unroll")                                                      \
            for (int _it = 0; _it < B_ITERS; _it++)                               \
                cp_async_16_stream_u32(_so + b_dst[_it], b_src[_it] + (K_ABS) * SP_BLOCK_K); \
        }                                                                          \
        if constexpr (META_STEPS == 1) {                                           \
            if (m_loader)                                                          \
                cp_async_8_u32(_so + m_dst[0], m_src[0] + (K_ABS) * 2);            \
        } else if (((K_ABS) % META_STEPS) == 0 || (K_ABS) == k_start_iter) {       \
            /* metadata chunk: at chunk boundaries and at the tile's first K step */\
            const int _c = (K_ABS) / META_STEPS;                                   \
            if (_c * META_STEPS + m_j < K_iters_total) {                           \
                const uint32_t _ms = smem_base + SMEM_M_OFF + (_c & 1) * SMEM_M_SLOT;\
                _Pragma("unroll")                                                  \
                for (int _it = 0; _it < M_COPIES; _it++)                           \
                    cp_async_8_u32(_ms + m_dst[_it], m_src[_it] + _c * (META_STEPS * 2));\
            }                                                                      \
        }                                                                          \
        cp_async_commit();                                                         \
    } while (0)

    // -- Prologue: async load first (DB_STAGES-1) tiles --
    // Fill all but one pipeline stage before entering the K-loop.
    // Offsets are absolute (k_start_iter + s) to handle split-K.
    #pragma unroll
    for (int s = 0; s < DB_STAGES - 1; s++)
        if (s < local_K_iters)
            LOAD_TILE(s * SMEM_PER_STAGE, k_start_iter + s);

    // -- Per-lane ldmatrix / metadata offsets within a stage --
    // A (x4): lane L addresses row warp_wr + (L & 15), K_comp half (L >> 4); +16 rows per
    //   M-tile keeps the swizzle phase, so M-tile mi is +mi*16*SP_K_COMP.
    // B (x4): lane L addresses row warp_bc + (L & 7), 16-byte K chunk (L >> 3); +8 rows per
    //   N-tile keeps the swizzle phase (+nj*8*SP_BLOCK_K).
    // Metadata: lane (g, tid) reads row warp_wr + mi*16 + g + 8*(tid&1), word tid>>1 of the
    //   current step's (swizzled) 8-byte slot; the swizzle phase q is the same for every mi.
    const uint32_t a_ldm_off = [&] {
        const int row = warp_wr + (lane & 15);
        return (uint32_t)(row * SP_K_COMP + a_swiz(row, (lane >> 4) << 4));
    }();
    const uint32_t b_ldm_off = [&] {
        const int row = warp_bc + (lane & 7);
        return (uint32_t)((FQ ? SMEM_FQ_B : SMEM_A_STAGE) + row * SP_BLOCK_K
                          + b_swiz(row, (lane >> 3) << 4));
    }();
    const int      m_row     = warp_wr + groupID + ((tid & 1) << 3);
    const int      m_q       = (m_row >> MQ_SHIFT) & (META_STEPS - 1);
    const uint32_t m_ldm_off = (META_STEPS == 1 ? SMEM_A_STAGE + SMEM_B_STAGE : SMEM_M_OFF)
                             + (uint32_t)(m_row * (META_STEPS * 8) + (tid >> 1) * 4);

    // -- Main K-loop (over local K range) --
    //
    //  Single-sync pipeline: one __syncthreads() per K-iteration instead of two.
    //  The sync serves double duty: (1) ensures previous iteration's compute is
    //  done (so the ahead buffer can be safely overwritten), and (2) ensures the
    //  current buffer's cp.async data is visible to all warps.
    //
    //  LOAD_TILE is placed AFTER the sync (but before compute) so the buffer it
    //  writes to -- buf[(ki+DB_STAGES-1)%DB_STAGES] -- is guaranteed free: its
    //  last consumer was compute at iteration ki-(DB_STAGES-1), which completed
    //  before the sync at iteration ki-(DB_STAGES-2) at the latest.
    //
    uint32_t comp_off  = 0;                                   // stage being computed
    uint32_t ahead_off = (DB_STAGES - 1) * SMEM_PER_STAGE;    // stage being loaded
    for (int ki = 0; ki < local_K_iters; ki++) {
        const int  k_ahead   = ki + (DB_STAGES - 1);          // local index
        const bool has_ahead = (k_ahead < local_K_iters);

        // -- Wait for CURRENT tile --
        // At most (DB_STAGES-2) commit groups may still be in flight.
        if (has_ahead)
            cp_async_wait_group<DB_STAGES - 2>();
        else
            cp_async_wait_group<0>();
        __syncthreads();

        // -- Initiate load (DB_STAGES-1) tiles ahead --
        if (has_ahead)
            LOAD_TILE(ahead_off, k_start_iter + k_ahead);

        const uint32_t cur = smem_base + comp_off;
        // B fragments: the current pipeline stage, or (FQ) K step k of the resident block
        const uint32_t bcur = FQ ? smem_base + (uint32_t)((k_start_iter + ki) * (BLOCK_BC_T * SP_BLOCK_K))
                                 : cur;

        // FQ: warps whose weight rows are all past N_out (narrow N, last tile) skip the math
        if (!FQ || block_wr + warp_wr < N_out) {

        // -- Metadata from the shared-memory ring (staged by LOAD_TILE) --
        // One conflict-free LDS per M-tile.
        uint32_t m_cur;
        if constexpr (META_STEPS == 1) {
            m_cur = cur + m_ldm_off;
        } else {
            const int k_abs = k_start_iter + ki;
            m_cur = smem_base + m_ldm_off
                  + ((k_abs / META_STEPS) & 1) * SMEM_M_SLOT
                  + (((k_abs % META_STEPS) ^ m_q) << 3);
        }
        uint32_t e_meta[TILES_M];
        #pragma unroll
        for (int mi = 0; mi < TILES_M; mi++)
            asm volatile("ld.shared.u32 %0, [%1];\n"
                         : "=r"(e_meta[mi]) : "r"(m_cur + mi * 16 * (META_STEPS * 8)));

        // -- Per-warp MMA computation --
        // All TILES_M A fragments are loaded once per K step and held in registers; the
        // N-tile loop is outermost so each B fragment is loaded exactly once and reused
        // across the warp's TILES_M M-tiles: TILES_M + TILES_N ldmatrix.x4 per K step.
        uint32_t a_frag0[TILES_M][2], a_frag1[TILES_M][2];
        #pragma unroll
        for (int mi = 0; mi < TILES_M; mi++) {
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[mi][0]), "=r"(a_frag0[mi][1]),
                  "=r"(a_frag1[mi][0]), "=r"(a_frag1[mi][1])
                : "r"(cur + a_ldm_off + mi * 16 * SP_K_COMP)
            );
        }

        #pragma unroll
        for (int nj = 0; nj < TILES_N; nj++) {
            uint32_t b_frag0[2], b_frag1[2];
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag0[0]), "=r"(b_frag0[1]), "=r"(b_frag1[0]), "=r"(b_frag1[1])
                : "r"(bcur + b_ldm_off + nj * 8 * SP_BLOCK_K)
            );

            #pragma unroll
            for (int mi = 0; mi < TILES_M; mi++) {
                // One m16n8k64 sparse MMA covers the whole 64-deep K step: A is the
                // 16x32 compressed tile {a_frag0, a_frag1}, B the 64x8 tile {b_frag0,
                // b_frag1}. Metadata: lanes tid=0/1 hold rows g/g+8 for K[0:31], lanes
                // tid=2/3 hold rows g/g+8 for K[32:63] (same E as the two k32 MMAs).
                asm volatile(
                    "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9, %10, %11}, "
                    "{%12, %13, %14, %15}, "
                    "%16, 0x0;\n"
                    : "=r"(acc[mi][nj][0]), "=r"(acc[mi][nj][1]),
                      "=r"(acc[mi][nj][2]), "=r"(acc[mi][nj][3])
                    : "r"(a_frag0[mi][0]), "r"(a_frag0[mi][1]),
                      "r"(a_frag1[mi][0]), "r"(a_frag1[mi][1]),
                      "r"(b_frag0[0]), "r"(b_frag0[1]),
                      "r"(b_frag1[0]), "r"(b_frag1[1]),
                      "r"(acc[mi][nj][0]), "r"(acc[mi][nj][1]),
                      "r"(acc[mi][nj][2]), "r"(acc[mi][nj][3]),
                      "r"(e_meta[mi])
                );
            }
        }
        }   // warp-live

        // Rotate stage offsets
        comp_off  = (comp_off  + SMEM_PER_STAGE == DB_STAGES * SMEM_PER_STAGE) ? 0 : comp_off  + SMEM_PER_STAGE;
        ahead_off = (ahead_off + SMEM_PER_STAGE == DB_STAGES * SMEM_PER_STAGE) ? 0 : ahead_off + SMEM_PER_STAGE;
    }

    // ================================================================
    //  Epilogue: dequant + bias in registers, transposed store
    //
    //  MMA output is C(N_out, M) in fragment layout: acc[mi][nj] holds
    //  (wr = m0+g, bc = n0+2tid / n0+2tid+1) and (wr = m0+g+8, same bc).
    //  Output is (M, N_out) row-major, so consecutive wr must be adjacent.
    //  Each 8x8 block is packed to half2 (row wr, cols bc, bc+1) and transposed
    //  in registers with movmatrix: afterwards the lane holds (row bc = n0+g,
    //  cols wr = m0+2tid, m0+2tid+1) -> one aligned 4-byte store per block.
    //  No shared memory, so no extra CTA barrier before the next tile.
    // ================================================================
    {
        // Per-token activation scale: global, or (FQ) the block's smem copy
        auto act_sc = [&](int bc) -> float {
            if constexpr (FQ)
                return reinterpret_cast<const float*>(smem_raw + SMEM_FQ_B + BLOCK_BC_T * K)[bc - block_bc];
            else
                return act_scale[min(bc, M - 1)];
        };
        float ws_r[TILES_M][2], bias_r[TILES_M][2];
        #pragma unroll
        for (int mi = 0; mi < TILES_M; mi++) {
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const int wr = min(block_wr + warp_wr + mi * 16 + groupID + h * 8, N_out - 1);
                ws_r[mi][h]   = __half2float(w_scale[wr]);
                bias_r[mi][h] = (bias != nullptr && splitk_ws == nullptr)
                              ? __half2float(bias[wr]) : 0.0f;
            }
        }

        if (splitk_ws != nullptr) {
            // Split-K: fp32 partials accumulated with atomics (bias in reduce kernel)
            #pragma unroll
            for (int nj = 0; nj < TILES_N; nj++) {
                const int bc_frag = block_bc + warp_bc + nj * 8 + tid * 2;   // fragment columns
                const float as0 = act_sc(bc_frag);
                const float as1 = act_sc(bc_frag + 1);
                #pragma unroll
                for (int mi = 0; mi < TILES_M; mi++) {
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int wr = block_wr + warp_wr + mi * 16 + groupID + h * 8;
                        if (wr < N_out) {
                            if (bc_frag < M)
                                atomicAdd(&splitk_ws[(size_t)bc_frag * N_out + wr],
                                          (float)acc[mi][nj][2 * h] * (as0 * ws_r[mi][h]));
                            if (bc_frag + 1 < M)
                                atomicAdd(&splitk_ws[(size_t)(bc_frag + 1) * N_out + wr],
                                          (float)acc[mi][nj][2 * h + 1] * (as1 * ws_r[mi][h]));
                        }
                    }
                }
            }
        } else {
            // Stage the warp's WARP_BC x WARP_WR fp16 tile in shared memory as output rows
            // (bc-major), then write 16-byte vectors: a warp store covers whole 64/128-byte
            // row segments instead of 8 rows x 16 bytes. The pipeline buffers are free once
            // every warp has finished its last K step (barrier below).
            constexpr int ROW_B = WARP_WR * 2;              // staged row pitch (bytes)
            constexpr int CPR   = WARP_WR / 8;              // 16-byte chunks per row (8 or 4)
            static_assert(CPR == 8 || CPR == 4, "staging swizzle assumes 4 or 8 chunks per row");
            static_assert(SP_WARPS_WR * SP_WARPS_BC * WARP_BC * ROW_B <= (FQ ? SMEM_FQ_B : SMEM_DB),
                          "epilogue staging must fit in the pipeline buffers");
            // XOR swizzle of the 16-byte chunk: conflict-free for the 8-row fragment stores
            // and for the 16-byte row reads.
            auto swz = [](int r) { return (CPR == 8) ? (r & 7) : ((r >> 1) & 3); };
            const uint32_t stg = smem_base + (uint32_t)(warp * WARP_BC * ROW_B);
            __syncthreads();

            #pragma unroll
            for (int nj = 0; nj < TILES_N; nj++) {
                const int bc_frag = block_bc + warp_bc + nj * 8 + tid * 2;   // fragment columns
                const float as0 = act_sc(bc_frag);
                const float as1 = act_sc(bc_frag + 1);
                const int r = nj * 8 + groupID;                              // row after transpose
                #pragma unroll
                for (int mi = 0; mi < TILES_M; mi++) {
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        float v0 = (float)acc[mi][nj][2 * h]     * (as0 * ws_r[mi][h]) + bias_r[mi][h];
                        float v1 = (float)acc[mi][nj][2 * h + 1] * (as1 * ws_r[mi][h]) + bias_r[mi][h];
                        __half2 hv = __floats2half2_rn(v0, v1);
                        uint32_t packed = *reinterpret_cast<uint32_t*>(&hv);
                        uint32_t trans;
                        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                                     : "=r"(trans) : "r"(packed));
                        const int c = mi * 2 + h;                            // 16-byte chunk
                        asm volatile("st.shared.u32 [%0], %1;\n" ::
                            "r"(stg + r * ROW_B + ((c ^ swz(r)) << 4) + tid * 4), "r"(trans));
                    }
                }
            }
            __syncwarp();

            #pragma unroll
            for (int i = lane; i < WARP_BC * CPR; i += 32) {
                const int r = i / CPR, c = i % CPR;
                uint32_t v0, v1, v2, v3;
                asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
                             : "r"(stg + r * ROW_B + ((c ^ swz(r)) << 4)));
                const int bc = block_bc + warp_bc + r;
                const int wr = block_wr + warp_wr + c * 8;
                // N_out % 16 == 0, so an 8-column chunk is entirely in or out of range
                if (bc < M && wr < N_out)
                    *reinterpret_cast<uint4*>(output + (size_t)bc * N_out + wr) =
                        make_uint4(v0, v1, v2, v3);
            }
        }
    }

    // Next tile's prologue overwrites pipeline stages (and the epilogue staging) that
    // other warps may still be reading.
    __syncthreads();
    } // end persistent tile loop

    #undef LOAD_TILE
}

// ===========================================================================
//  Kernel wrapper
// ===========================================================================

template <int BLOCK_WR_T, int BLOCK_BC_T, bool NF = false>
__global__ void sparse_gemm_kernel(
    const int8_t*   __restrict__ W_comp,
    const int8_t*   __restrict__ x_int8,
    const uint32_t* __restrict__ w_meta,
    const float*    __restrict__ act_scale,
    const __half*   __restrict__ w_scale,
    const __half*   __restrict__ bias,
    __half*         __restrict__ output,
    float*          __restrict__ splitk_ws,
    const int M, const int K, const int N_out,
    const int split_k, const int k_iters_per_split)
{
    sparse_gemm_impl<BLOCK_WR_T, BLOCK_BC_T, sp_stages<BLOCK_WR_T, BLOCK_BC_T>(), false, NF>(
        W_comp, x_int8, w_meta, act_scale, w_scale, bias, output,
        splitk_ws, M, K, N_out, split_k, k_iters_per_split, nullptr, 0);
}

// Fused-quantize kernel: reads the fp16 activation, quantizes each 64-token block into a
// resident smem block, runs every weight tile against it (see sparse_int8_linear).
template <int BLOCK_WR_T, int BLOCK_BC_T>
__global__ void sparse_gemm_fq_kernel(
    const int8_t*   __restrict__ W_comp,
    const uint32_t* __restrict__ w_meta,
    const __half*   __restrict__ w_scale,
    const __half*   __restrict__ bias,
    __half*         __restrict__ output,
    const int M, const int K, const int N_out,
    const __half*   __restrict__ x_fp16, const int x_stream)
{
    sparse_gemm_impl<BLOCK_WR_T, BLOCK_BC_T, sp_stages<BLOCK_WR_T, BLOCK_BC_T>(), true>(
        W_comp, nullptr, w_meta, nullptr, w_scale, bias, output,
        nullptr, M, K, N_out, 1, K / SP_BLOCK_K, x_fp16, x_stream);
}

// ===========================================================================
//  Host-side dispatch
// ===========================================================================
namespace {

constexpr int kTileBW[kSparseNumTiles] = {256, 128, 128};
constexpr int kTileBC[kSparseNumTiles] = {128, 128, 64};

struct SpDeviceInfo {
    int sm_count = 0;
    int smem_per_sm = 0;
    int reserved_smem_per_block = 0;
    int smem_optin = 0;
    int l2_bytes = 0;
    int launch_smem[kSparseNumTiles] = {};   // dynamic smem per tile config, 0 = not set up
    int fq_smem_set = 0;                     // max dynamic smem attribute set for the FQ kernel
};

// Per-device properties, queried once (128 SMs / 100 KB smem per SM on an RTX 4090).
SpDeviceInfo& sp_device_info() {
    static SpDeviceInfo cache[64];
    int dev = 0;
    cudaGetDevice(&dev);
    SpDeviceInfo& d = cache[dev & 63];
    if (d.sm_count == 0) {
        int sms = 0;
        cudaDeviceGetAttribute(&d.smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
        cudaDeviceGetAttribute(&d.reserved_smem_per_block, cudaDevAttrReservedSharedMemoryPerBlock, dev);
        cudaDeviceGetAttribute(&d.smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&d.l2_bytes, cudaDevAttrL2CacheSize, dev);
        d.sm_count = (sms > 0) ? sms : 128;
    }
    return d;
}

// One persistent CTA per SM. When registers and smem would allow two, the block scheduler
// can place two of the #SMs CTAs on one SM and leave another idle (measured: 128x64 at 96
// regs ran 12-19% slower), and two co-resident CTAs were slower even with a 2x grid.
// Requesting just over half the SM's shared memory guarantees one CTA per SM on any GPU.
template <int BW, int BC>
int sp_launch_smem(SpDeviceInfo& d, int tile) {
    int& smem = d.launch_smem[tile];
    if (smem == 0) {
        constexpr int smem_k = sparse_gemm_smem_bytes<BW, BC>();
        const int half = d.smem_per_sm / 2 - d.reserved_smem_per_block + 128;
        int s = (smem_k > half) ? smem_k : half;
        if (s > d.smem_optin) s = smem_k;
        if (s > 48 * 1024) {
            cudaFuncSetAttribute(sparse_gemm_kernel<BW, BC, false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, s);
            cudaFuncSetAttribute(sparse_gemm_kernel<BW, BC, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, s);
        }
        smem = s;
    }
    return smem;
}

template <int BW, int BC>
void sp_launch(SpDeviceInfo& d, int tile, const int8_t* W_comp, const int8_t* x_int8,
               const uint32_t* w_meta, const float* act_scale, const __half* w_scale,
               const __half* bias, __half* output, float* ws, int M, int K, int N_out,
               int split_k, cudaStream_t stream)
{
    const int smem = sp_launch_smem<BW, BC>(d, tile);
    const long long spatial = (long long)divup(N_out, BW) * divup(M, BC);
    const long long total = spatial * split_k;
    const int nctas = (int)std::min<long long>(total, d.sm_count);
    const int k_iters_total = K / SP_BLOCK_K;
    const int k_iters_per_split = (k_iters_total + split_k - 1) / split_k;
    // Tile order: M-fastest keeps a weight tile hot across consecutive M-tiles, but every
    // weight-tile column then streams the whole int8 activation; once that exceeds L2 it is
    // re-read from DRAM per column (3x for N_out = 320 on 128-row tiles). N-fastest reads it once.
    if ((long long)M * K > (long long)d.l2_bytes)
        sparse_gemm_kernel<BW, BC, true><<<nctas, SP_THREADS, smem, stream>>>(
            W_comp, x_int8, w_meta, act_scale, w_scale, bias, output,
            ws, M, K, N_out, split_k, k_iters_per_split);
    else
        sparse_gemm_kernel<BW, BC, false><<<nctas, SP_THREADS, smem, stream>>>(
            W_comp, x_int8, w_meta, act_scale, w_scale, bias, output,
            ws, M, K, N_out, split_k, k_iters_per_split);
}

// Fused quantize + sparse GEMM: 128-row weight tiles x 64-token activation blocks.
constexpr int SP_FQ_BW = 128, SP_FQ_BC = 64;

int sp_fq_smem_bytes(int K) {
    return sp_fq_base_bytes<SP_FQ_BW, SP_FQ_BC>() + SP_FQ_BC * K + SP_FQ_BC * (int)sizeof(float);
}

bool sp_fq_fits(SpDeviceInfo& d, int K) {
    return K % SP_BLOCK_K == 0 && K <= sp_fq_vpt<SP_FQ_BC>() * 256 && sp_fq_smem_bytes(K) <= d.smem_optin;
}

void sp_fq_launch(SpDeviceInfo& d, const __half* x, const int8_t* W_comp, const uint32_t* w_meta,
                  const __half* w_scale, const __half* bias, __half* output,
                  int M, int K, int N_out, cudaStream_t stream)
{
    constexpr int BW = SP_FQ_BW, BC = SP_FQ_BC;
    const int smem = sp_fq_smem_bytes(K);
    if (smem > d.fq_smem_set) {
        cudaFuncSetAttribute(sparse_gemm_fq_kernel<BW, BC>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        d.fq_smem_set = smem;
    }
    const long long tiles = (long long)divup(N_out, BW) * divup(M, BC);
    const int nctas = (int)std::min<long long>(tiles, d.sm_count);
    // Evict-first loads when x exceeds L2: it streams, and L2 is kept for the prefetched rows
    const int x_stream = ((long long)M * K * 2 > (long long)d.l2_bytes) ? 1 : 0;
    sparse_gemm_fq_kernel<BW, BC><<<nctas, SP_THREADS, smem, stream>>>(
        W_comp, w_meta, w_scale, bias, output, M, K, N_out, x, x_stream);
}

} // namespace

// ===========================================================================
//  Launchers (standalone C++ API)
// ===========================================================================
//
// Dispatch: pick (tile, split-K) by a wave-aware cost model fitted on an RTX 4090
// sweep (K 1280..6144 x N 768..6144 x M 64..8192, every tile x split):
//   t(tile, s) = waves * (a_tile * ceil(ksteps / s) + b_tile)
//              + [s > 1] * (c0 + (c1 + c2 * s) * M*N/1e6)
//   waves      = ceil(spatial_tiles(tile) * s / SMs)   (one persistent CTA per SM)
// a_tile: time per 64-deep K step of one tile, b_tile: per-tile fixed cost (pipeline
// fill, epilogue), c0/c1/c2: split-K launch, workspace memset + reduce, atomics.
// Splits above 6 lost on every measured shape (atomic contention on the M x N
// partials outgrows the extra parallelism), so they are not auto candidates.
// Wave quantization dominates small and mid shapes, which fixed M/N thresholds miss.
// The coefficients are RTX 4090 numbers; only the SM count adapts to other GPUs.

namespace {
double sp_plan_cost(int M, int K, int N_out, int split_k, bool have_workspace, int tile,
                    SparseGemmPlan* out);
}

SparseGemmPlan sparse_int8_gemm_plan(int M, int K, int N_out, int split_k,
                                     bool have_workspace, int tile)
{
    SparseGemmPlan plan;
    sp_plan_cost(M, K, N_out, split_k, have_workspace, tile, &plan);
    return plan;
}

namespace {
// Cost-model minimum (ms) and the plan that reaches it.
double sp_plan_cost(int M, int K, int N_out, int split_k, bool have_workspace, int tile,
                    SparseGemmPlan* out)
{
    static constexpr double kA[kSparseNumTiles] = {8.1730e-04, 4.0900e-04, 3.1130e-04};  // ms per K step per tile
    static constexpr double kB[kSparseNumTiles] = {3.5480e-03, 2.2260e-03, 1.2530e-03};  // ms per tile
    constexpr double kSplitC0 = 2.9200e-03;   // ms per split-K launch
    constexpr double kSplitC1 = 1.1810e-03;   // ms per 1e6 outputs
    constexpr double kSplitC2 = 2.4990e-03;   // ms per 1e6 outputs per split
    static constexpr int kAutoSplits[] = {1, 2, 3, 4, 6};

    if (tile >= kSparseNumTiles) tile = kSparseTileAuto;
    if (split_k > 1 && !have_workspace) split_k = 1;      // split-K needs the fp32 workspace

    const int sm_count = sp_device_info().sm_count;
    const int k_iters_total = K / SP_BLOCK_K;
    const bool auto_split = (split_k <= 0);

    SparseGemmPlan plan{kSparseTile128x128, 1};
    double best_cost = 1e30;
    for (int t = 0; t < kSparseNumTiles; t++) {
        if (tile >= 0 && t != tile) continue;
        const long long tiles = (long long)divup(N_out, kTileBW[t]) * divup(M, kTileBC[t]);
        const int n_cand = auto_split ? (have_workspace ? 5 : 1) : 1;
        for (int c = 0; c < n_cand; c++) {
            const int s = auto_split ? kAutoSplits[c] : split_k;
            if (auto_split && s > 1 && s > k_iters_total / 2) continue;
            const long long waves = (tiles * s + sm_count - 1) / sm_count;
            const int ksteps = (k_iters_total + s - 1) / s;
            double cost = (double)waves * (kA[t] * ksteps + kB[t]);
            if (s > 1) cost += kSplitC0 + (kSplitC1 + kSplitC2 * s) * ((double)M * N_out * 1e-6);
            if (cost < best_cost) { best_cost = cost; plan.tile = t; plan.split_k = s; }
        }
    }
    *out = plan;
    return best_cost;
}

// Path choice for the fp16-input layer. The two-pass path writes an int8 copy of x and the
// GEMM reads it back: nearly free while x and the copy (3*M*K bytes) stay in L2, 2*M*K bytes
// of DRAM traffic once they do not (measured cut-over between 83% and 94% of the 72 MiB L2,
// which also holds weights and output; 90% used). The fused path saves that traffic but runs the small
// 128x64 tile with the weights re-streamed from L2 per 64 tokens. Take it when the DRAM
// time saved (at ~0.92 TB/s, RTX 4090 copy bandwidth) exceeds the cost-model penalty of the
// 128x64 tile vs the best two-pass plan. Narrow N_out at huge M qualifies (1280->320 at
// M=147456); wide N_out and every shape whose activation fits L2 stay on the two-pass path.
bool sp_use_fused(SpDeviceInfo& d, int M, int K, int N_out) {
    if (M <= 0 || !sp_fq_fits(d, K)) return false;
    if (3.0 * M * K <= 0.9 * (double)d.l2_bytes) return false;
    SparseGemmPlan p;
    const double fused = sp_plan_cost(M, K, N_out, 1, false, kSparseTile128x64, &p);
    const double best  = sp_plan_cost(M, K, N_out, 0, true, kSparseTileAuto, &p);
    constexpr double kDramBytesPerMs = 0.92e9;
    return fused - best < 2.0 * M * K / kDramBytesPerMs;
}
} // namespace

bool sparse_int8_linear_fused(int M, int K, int N_out) {
    return sp_use_fused(sp_device_info(), M, K, N_out);
}

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
    cudaStream_t    stream,
    int             tile)
{
    if (M <= 0 || K <= 0 || N_out <= 0) return;

    const SparseGemmPlan plan = sparse_int8_gemm_plan(
        M, K, N_out, split_k, splitk_workspace != nullptr, tile);
    float* ws = (plan.split_k > 1) ? splitk_workspace : nullptr;
    if (ws != nullptr)   // partials are accumulated with fp32 atomics
        cudaMemsetAsync(ws, 0, (size_t)M * N_out * sizeof(float), stream);

    SpDeviceInfo& d = sp_device_info();
    switch (plan.tile) {
    case kSparseTile256x128:
        sp_launch<256, 128>(d, plan.tile, W_comp, x_int8, w_meta, act_scale, w_scale, bias,
                            output, ws, M, K, N_out, plan.split_k, stream);
        break;
    case kSparseTile128x128:
        sp_launch<128, 128>(d, plan.tile, W_comp, x_int8, w_meta, act_scale, w_scale, bias,
                            output, ws, M, K, N_out, plan.split_k, stream);
        break;
    default:
        sp_launch<128, 64>(d, plan.tile, W_comp, x_int8, w_meta, act_scale, w_scale, bias,
                           output, ws, M, K, N_out, plan.split_k, stream);
        break;
    }

    // Split-K finalization: fp32 partials -> fp16 output + bias
    if (ws != nullptr)
        splitk_reduce(ws, bias, output, M, N_out, stream);
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
    cudaStream_t    stream,
    int             path)
{
    if (M <= 0 || K <= 0 || N_out <= 0) return;
    // Fused quantize + GEMM (one kernel, no int8 round trip): forced, or chosen by
    // sp_use_fused when the caller did not ask for a specific split-K. x must be 16-byte
    // aligned (16-byte vector loads).
    SpDeviceInfo& d = sp_device_info();
    const bool aligned = (reinterpret_cast<uintptr_t>(x) % 16) == 0;
    const bool fused = aligned && sp_fq_fits(d, K) &&
        (path == kSparseLinearFused ||
         (path == kSparseLinearAuto && split_k <= 1 && sp_use_fused(d, M, K, N_out)));
    if (fused) {
        sp_fq_launch(d, x, W_comp, w_meta, w_scale, bias, output, M, K, N_out, stream);
        return;
    }

    // Step 1: quantize activation
    forge::quant::fused_quantize_activation(x, scratch_x_int8, scratch_x_scale, M, K, stream);

    // Step 2: sparse GEMM
    sparse_int8_gemm(W_comp, scratch_x_int8, w_meta, scratch_x_scale,
                     w_scale, bias, output, M, K, N_out,
                     splitk_workspace, split_k, stream);
}

}} // namespace forge::gemm
