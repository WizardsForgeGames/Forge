// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Fused 3D Winograd F(2,3)^3 — k=3x3x3, stride=1, pad=1.
//
// v2: Key optimizations:
//
// 1. Fused B^T input transform in registers. Instead of 3 sequential
//    in-place shared-memory B^T passes (each requiring a __syncthreads),
//    each thread computes its V[alpha] directly from the raw input tile
//    using 8 shared memory reads. Eliminates 4 of 6 syncs per cin
//    iteration, keeps ALL threads active.
//
// 2. Cin-unrolled by 2: processes 2 cin channels per pair of syncs.
//    Load both cin values to shared memory → sync → compute both → sync.
//    This halves the effective sync count (1 per cin vs 2 in baseline)
//    without overlapping load/compute phases, avoiding register pressure.
//    Shared memory: 2× input + 2× weight ≈ 19KB per block.
//
// 3. Transposed U_t layout for backward coalescing. Forward uses
//    U[alpha][cin][cout], backward uses U_t[alpha][cout][cin].

#include <forge/nn/conv3d_winograd.cuh>
#include <forge/error.h>
#include <mma.h>

namespace forge { namespace nn {

// Spatial stride for 4x4x4 Winograd domain patches

// Alpha flip for backward data (spatial kernel flip in Winograd domain)
__device__ __constant__ int ALPHA_FLIP[4] = {3, 1, 2, 0};

// B^T non-zero indices and signs per row
__device__ __constant__ int   BT_IDX[4][2] = {{0,2}, {1,2}, {1,2}, {1,3}};
__device__ __constant__ float BT_SGN[4][2] = {{1,-1}, {1,1}, {-1,1}, {1,-1}};

// A^T inverse transform coefficients for F(2,3) Winograd
//   AT[0] = {1, 1,  1,  0}  (output position 0)
//   AT[1] = {0, 1, -1, -1}  (output position 1)
__device__ __constant__ float AT_COEFF[2][4] = {
    {1.0f, 1.0f,  1.0f, 0.0f},
    {0.0f, 1.0f, -1.0f, -1.0f}
};

// =====================================================================
// Filter Transform Kernel
// =====================================================================

__global__ void winograd_filter_transform_kernel(
    const float* __restrict__ weight,
    float*       __restrict__ U,
    float*       __restrict__ U_t,
    int C_out, int C_in)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C_out * C_in) return;

    int cout = idx / C_in;
    int cin  = idx % C_in;

    const float* g = weight + (cout * C_in + cin) * 27;
    float f[3][3][3];
    #pragma unroll
    for (int d = 0; d < 3; d++)
        #pragma unroll
        for (int h = 0; h < 3; h++)
            #pragma unroll
            for (int w = 0; w < 3; w++)
                f[d][h][w] = g[d * 9 + h * 3 + w];

    float t1[3][3][4];
    #pragma unroll
    for (int d = 0; d < 3; d++)
        #pragma unroll
        for (int h = 0; h < 3; h++) {
            float g0 = f[d][h][0], g1 = f[d][h][1], g2 = f[d][h][2];
            t1[d][h][0] = g0;
            t1[d][h][1] = (g0 + g1 + g2) * 0.5f;
            t1[d][h][2] = (g0 - g1 + g2) * 0.5f;
            t1[d][h][3] = g2;
        }

    float t2[3][4][4];
    #pragma unroll
    for (int d = 0; d < 3; d++)
        #pragma unroll
        for (int aw = 0; aw < 4; aw++) {
            float g0 = t1[d][0][aw], g1 = t1[d][1][aw], g2 = t1[d][2][aw];
            t2[d][0][aw] = g0;
            t2[d][1][aw] = (g0 + g1 + g2) * 0.5f;
            t2[d][2][aw] = (g0 - g1 + g2) * 0.5f;
            t2[d][3][aw] = g2;
        }

    const int CinCout = C_in * C_out;
    #pragma unroll
    for (int ad = 0; ad < 4; ad++)
        #pragma unroll
        for (int ah = 0; ah < 4; ah++)
            #pragma unroll
            for (int aw = 0; aw < 4; aw++) {
                float g0 = t2[0][ah][aw], g1 = t2[1][ah][aw], g2 = t2[2][ah][aw];
                float val;
                switch (ad) {
                    case 0: val = g0; break;
                    case 1: val = (g0 + g1 + g2) * 0.5f; break;
                    case 2: val = (g0 - g1 + g2) * 0.5f; break;
                    default: val = g2; break;
                }
                int alpha = ad * 16 + ah * 4 + aw;
                U[alpha * CinCout + cin * C_out + cout] = val;
                if (U_t)
                    U_t[alpha * CinCout + cout * C_in + cin] = val;
            }
}

// =====================================================================
// Fused Winograd Kernel v2 — register B^T + cin-unrolled-by-2
// =====================================================================

static constexpr int BC = 32;
static constexpr int COUT_CHUNK = 8;
static constexpr int TILE_STRIDE = 64;

template <int BT_VAL, bool IS_BACKWARD>
__device__ __forceinline__
void conv3d_winograd_v2_impl(
    const float* __restrict__ input,
    const float* __restrict__ U,
    const float* __restrict__ bias,
    float*       __restrict__ output,
    int C_loop, int C_tile,
    int D, int H, int W,
    int T_d, int T_h, int T_w, int T_total)
{
    constexpr int BLOCK = BT_VAL * 64;
    constexpr int INP_BUF = BT_VAL * TILE_STRIDE;
    constexpr int WGT_BUF = 64 * (BC + 1);

    const int tile_base = blockIdx.x * BT_VAL;
    const int cout_base = blockIdx.y * BC;
    const int n = blockIdx.z;

    const int tid = threadIdx.x;
    const int thread_tile  = tid >> 6;
    const int thread_alpha = tid & 63;

    // Decode alpha into per-dimension components
    const int alpha_d = thread_alpha >> 4;
    const int alpha_h = (thread_alpha >> 2) & 3;
    const int alpha_w = thread_alpha & 3;

    // Tile spatial origin
    int t_global = tile_base + thread_tile;
    int tw = 0, th = 0, td = 0;
    if (t_global < T_total) {
        tw = t_global % T_w;
        th = (t_global / T_w) % T_h;
        td = t_global / (T_w * T_h);
    }
    int d0 = td * 2 - 1;
    int h0 = th * 2 - 1;
    int w0 = tw * 2 - 1;

    // Precompute B^T offsets and signs for this thread's alpha.
    int bt_off[8];
    float bt_sgn[8];
    {
        int d_idx0 = BT_IDX[alpha_d][0], d_idx1 = BT_IDX[alpha_d][1];
        int h_idx0 = BT_IDX[alpha_h][0], h_idx1 = BT_IDX[alpha_h][1];
        int w_idx0 = BT_IDX[alpha_w][0], w_idx1 = BT_IDX[alpha_w][1];
        float ds0 = BT_SGN[alpha_d][0], ds1 = BT_SGN[alpha_d][1];
        float hs0 = BT_SGN[alpha_h][0], hs1 = BT_SGN[alpha_h][1];
        float ws0 = BT_SGN[alpha_w][0], ws1 = BT_SGN[alpha_w][1];

        bt_off[0] = d_idx0*16 + h_idx0*4 + w_idx0;
        bt_off[1] = d_idx0*16 + h_idx0*4 + w_idx1;
        bt_off[2] = d_idx0*16 + h_idx1*4 + w_idx0;
        bt_off[3] = d_idx0*16 + h_idx1*4 + w_idx1;
        bt_off[4] = d_idx1*16 + h_idx0*4 + w_idx0;
        bt_off[5] = d_idx1*16 + h_idx0*4 + w_idx1;
        bt_off[6] = d_idx1*16 + h_idx1*4 + w_idx0;
        bt_off[7] = d_idx1*16 + h_idx1*4 + w_idx1;

        bt_sgn[0] = ds0*hs0*ws0;  bt_sgn[1] = ds0*hs0*ws1;
        bt_sgn[2] = ds0*hs1*ws0;  bt_sgn[3] = ds0*hs1*ws1;
        bt_sgn[4] = ds1*hs0*ws0;  bt_sgn[5] = ds1*hs0*ws1;
        bt_sgn[6] = ds1*hs1*ws0;  bt_sgn[7] = ds1*hs1*ws1;
    }

    // ===== Shared memory: 2 input + 2 weight slots for cin-unroll =====
    // Layout: [inp_A | inp_B | wgt_A | wgt_B]
    extern __shared__ float smem[];
    float* s_inp_A = smem;
    float* s_inp_B = smem + INP_BUF;
    float* s_wgt_A = smem + 2 * INP_BUF;
    float* s_wgt_B = smem + 2 * INP_BUF + WGT_BUF;

    // Accumulators
    float acc[BC];
    #pragma unroll
    for (int c = 0; c < BC; c++) acc[c] = 0.0f;

    const int DHW = D * H * W;
    const int HW  = H * W;
    const float* inp_base = input  + (size_t)n * C_loop * DHW;
    float*       out_base = output + (size_t)n * C_tile * DHW;
    const int U_alpha_stride = C_loop * C_tile;

    // Pre-compute input spatial offset (constant across all cin)
    int d_in = d0 + (thread_alpha >> 4);
    int h_in = h0 + ((thread_alpha >> 2) & 3);
    int w_in = w0 + (thread_alpha & 3);
    bool inp_valid = (t_global < T_total) &&
                     (d_in >= 0 && d_in < D && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W);
    int inp_off = d_in * HW + h_in * W + w_in;
    int inp_smem = thread_tile * TILE_STRIDE + thread_alpha;

    // =================== Cin-unrolled main loop (2 cin per sync pair) ===================
    // Process 2 cin values per pair of syncs:
    //   load(A, ci) + load(B, ci+1) → sync → compute(A) + compute(B) → sync
    // This halves the sync count vs the baseline without overlapping load/compute,
    // so register pressure stays at baseline levels (~80 regs, 0 stack spill).

    for (int ci = 0; ci < C_loop; ci += 2) {
        bool has_second = (ci + 1 < C_loop);

        // --- Load phase: 2 cin values into shared memory ---

        // Input A (ci)
        {
            float val = inp_valid ? inp_base[ci * DHW + inp_off] : 0.0f;
            s_inp_A[inp_smem] = val;
        }

        // Input B (ci+1)
        if (has_second) {
            float val = inp_valid ? inp_base[(ci + 1) * DHW + inp_off] : 0.0f;
            s_inp_B[inp_smem] = val;
        }

        // Weight A (ci)
        for (int i = tid; i < 64 * BC; i += BLOCK) {
            int a = i >> 5;
            int c = i & 31;
            int cout_idx = cout_base + c;
            float w_val = 0.0f;
            if (cout_idx < C_tile) {
                int a_eff = a;
                if (IS_BACKWARD) {
                    int ad = a >> 4, ah = (a >> 2) & 3, aw = a & 3;
                    a_eff = ALPHA_FLIP[ad] * 16 + ALPHA_FLIP[ah] * 4 + ALPHA_FLIP[aw];
                }
                w_val = U[a_eff * U_alpha_stride + ci * C_tile + cout_idx];
            }
            s_wgt_A[a * (BC + 1) + c] = w_val;
        }

        // Weight B (ci+1)
        if (has_second) {
            for (int i = tid; i < 64 * BC; i += BLOCK) {
                int a = i >> 5;
                int c = i & 31;
                int cout_idx = cout_base + c;
                float w_val = 0.0f;
                if (cout_idx < C_tile) {
                    int a_eff = a;
                    if (IS_BACKWARD) {
                        int ad = a >> 4, ah = (a >> 2) & 3, aw = a & 3;
                        a_eff = ALPHA_FLIP[ad] * 16 + ALPHA_FLIP[ah] * 4 + ALPHA_FLIP[aw];
                    }
                    w_val = U[a_eff * U_alpha_stride + (ci + 1) * C_tile + cout_idx];
                }
                s_wgt_B[a * (BC + 1) + c] = w_val;
            }
        }

        __syncthreads();

        // --- Compute phase: B^T + accumulate for both cin values ---

        // Compute A (ci)
        {
            const float* s_tile = s_inp_A + thread_tile * TILE_STRIDE;
            float v = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                v += bt_sgn[i] * s_tile[bt_off[i]];

            const float* w_row = s_wgt_A + thread_alpha * (BC + 1);
            #pragma unroll
            for (int c = 0; c < BC; c++)
                acc[c] += w_row[c] * v;
        }

        // Compute B (ci+1)
        if (has_second) {
            const float* s_tile = s_inp_B + thread_tile * TILE_STRIDE;
            float v = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                v += bt_sgn[i] * s_tile[bt_off[i]];

            const float* w_row = s_wgt_B + thread_alpha * (BC + 1);
            #pragma unroll
            for (int c = 0; c < BC; c++)
                acc[c] += w_row[c] * v;
        }

        __syncthreads();
    }

    // =================== Output Transform (A^T, chunked) ===================
    float* s_acc = smem;

    for (int cc = 0; cc < BC; cc += COUT_CHUNK) {
        #pragma unroll
        for (int c = 0; c < COUT_CHUNK; c++)
            s_acc[(thread_tile * 64 + thread_alpha) * COUT_CHUNK + c] = acc[cc + c];
        __syncthreads();

        {
            int c_local = tid % COUT_CHUNK;
            int rem2 = tid / COUT_CHUNK;
            int t_local = rem2 / 8;
            int out_pos = rem2 % 8;

            int t_g = tile_base + t_local;
            int cout_idx = cout_base + cc + c_local;

            if (t_g < T_total && cout_idx < C_tile) {
                int ow = out_pos & 1;
                int oh = (out_pos >> 1) & 1;
                int od = (out_pos >> 2) & 1;

                int ad0 = od, ah0 = oh, aw0 = ow;

                float result = 0.0f;
                #pragma unroll
                for (int di = 0; di < 3; di++) {
                    float ds = (od == 0 || di == 0) ? 1.0f : -1.0f;
                    #pragma unroll
                    for (int hi = 0; hi < 3; hi++) {
                        float hs = (oh == 0 || hi == 0) ? 1.0f : -1.0f;
                        #pragma unroll
                        for (int wi = 0; wi < 3; wi++) {
                            float ws = (ow == 0 || wi == 0) ? 1.0f : -1.0f;
                            int alpha = (ad0+di) * 16 + (ah0+hi) * 4 + (aw0+wi);
                            result += ds * hs * ws *
                                s_acc[(t_local * 64 + alpha) * COUT_CHUNK + c_local];
                        }
                    }
                }

                if (!IS_BACKWARD && bias) result += bias[cout_idx];

                int out_tw = t_g % T_w;
                int out_th = (t_g / T_w) % T_h;
                int out_td = t_g / (T_w * T_h);
                int d_out = out_td * 2 + od;
                int h_out = out_th * 2 + oh;
                int w_out = out_tw * 2 + ow;

                if (d_out < D && h_out < H && w_out < W)
                    out_base[cout_idx * DHW + d_out * HW + h_out * W + w_out] = result;
            }
        }
        __syncthreads();
    }
}

// Forward: launch_bounds for 3 blocks/SM occupancy
template <int BT_VAL>
__global__ __launch_bounds__(BT_VAL * 64, 3)
void conv3d_winograd_v2_forward(
    const float* __restrict__ input,
    const float* __restrict__ U,
    const float* __restrict__ bias,
    float*       __restrict__ output,
    int C_loop, int C_tile,
    int D, int H, int W,
    int T_d, int T_h, int T_w, int T_total)
{
    conv3d_winograd_v2_impl<BT_VAL, false>(
        input, U, bias, output, C_loop, C_tile,
        D, H, W, T_d, T_h, T_w, T_total);
}

// Backward: no launch_bounds — backward needs ~96 regs for ALPHA_FLIP path
template <int BT_VAL>
__global__
void conv3d_winograd_v2_backward(
    const float* __restrict__ input,
    const float* __restrict__ U,
    const float* __restrict__ bias,
    float*       __restrict__ output,
    int C_loop, int C_tile,
    int D, int H, int W,
    int T_d, int T_h, int T_w, int T_total)
{
    conv3d_winograd_v2_impl<BT_VAL, true>(
        input, U, bias, output, C_loop, C_tile,
        D, H, W, T_d, T_h, T_w, T_total);
}

// =====================================================================
// Fused Winograd Kernel v3 — Multi-Warp WMMA TF32 Tensor Cores
// =====================================================================
//
// Multi-warp design: 128 threads (4 warps) per block.
// All 128 threads cooperate on B^T fill (the bottleneck).
// Each warp does independent WMMA for its own 16-cout block.
// Alpha-outer loop preserves WMMA cin-accumulation (1 store+scatter per alpha).
//
// Block: 128 threads (4 warps)
// Grid:  (ceil(T_total/16), ceil(C_tile/64), N)
//
// Requires: C_loop % 8 == 0 && C_tile % 16 == 0.
// Falls back to v2 otherwise.

static constexpr int V3_NW = 4;   // warps per block
static constexpr int V3_BLOCK = V3_NW * 32;  // 128 threads

template <bool IS_BACKWARD>
__global__ __launch_bounds__(V3_BLOCK, 3)
void conv3d_winograd_v3_kernel(
    const float* __restrict__ input,
    const float* __restrict__ U,
    const float* __restrict__ bias,
    float*       __restrict__ output,
    int C_loop, int C_tile,
    int D, int H, int W,
    int T_d, int T_h, int T_w, int T_total)
{
    using namespace nvcuda::wmma;

    const int tile_base = blockIdx.x * 16;
    const int cout_base = blockIdx.y * (V3_NW * 16);  // 64 cout per block
    const int n = blockIdx.z;
    const int tid = threadIdx.x;       // 0..127
    const int warp_id = tid >> 5;      // 0..3
    const int lane_id = tid & 31;      // 0..31

    // Per-warp cout offset
    const int cout_warp = cout_base + warp_id * 16;

    // Static shared memory
    __shared__ float s_A_buf[16][8];           // WMMA A staging (shared across warps)
    __shared__ float s_result[V3_NW][16][16];  // Per-warp WMMA C output
    __shared__ int   s_tile_d0[16];
    __shared__ int   s_tile_h0[16];
    __shared__ int   s_tile_w0[16];

    // Precompute tile coordinates (threads 0-15)
    if (tid < 16) {
        int t_global = tile_base + tid;
        if (t_global < T_total) {
            int tw = t_global % T_w;
            int th = (t_global / T_w) % T_h;
            int td = t_global / (T_w * T_h);
            s_tile_d0[tid] = td * 2 - 1;
            s_tile_h0[tid] = th * 2 - 1;
            s_tile_w0[tid] = tw * 2 - 1;
        } else {
            s_tile_d0[tid] = -1;
            s_tile_h0[tid] = -1;
            s_tile_w0[tid] = -1;
        }
    }
    __syncthreads();

    // Output registers: 8 cout × 8 positions = 64 per thread
    float out_reg[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) out_reg[i] = 0.0f;

    const int DHW = D * H * W;
    const int HW  = H * W;
    const float* inp_base = input + (size_t)n * C_loop * DHW;
    const int U_stride = C_loop * C_tile;

    // Thread ownership for A^T scatter (within warp)
    const int my_tile = lane_id >> 1;            // 0..15
    const int my_cout_start = (lane_id & 1) * 8; // 0 or 8

    // Does this warp have valid cout channels?
    const bool warp_valid = (cout_warp < C_tile);

    // =================== Main loop: alpha outer (64), cin inner ===================
    for (int alpha = 0; alpha < 64; alpha++) {
        const int ad = alpha >> 4;
        const int ah = (alpha >> 2) & 3;
        const int aw = alpha & 3;

        int a_eff = alpha;
        if constexpr (IS_BACKWARD) {
            a_eff = ALPHA_FLIP[ad] * 16 + ALPHA_FLIP[ah] * 4 + ALPHA_FLIP[aw];
        }

        // B^T offsets and signs for this alpha
        const int d_idx0 = BT_IDX[ad][0], d_idx1 = BT_IDX[ad][1];
        const int h_idx0 = BT_IDX[ah][0], h_idx1 = BT_IDX[ah][1];
        const int w_idx0 = BT_IDX[aw][0], w_idx1 = BT_IDX[aw][1];
        const float ds0 = BT_SGN[ad][0], ds1 = BT_SGN[ad][1];
        const float hs0 = BT_SGN[ah][0], hs1 = BT_SGN[ah][1];
        const float ws0 = BT_SGN[aw][0], ws1 = BT_SGN[aw][1];

        int bt_d[8], bt_h[8], bt_w[8];
        float bt_s[8];
        bt_d[0]=d_idx0; bt_h[0]=h_idx0; bt_w[0]=w_idx0; bt_s[0]=ds0*hs0*ws0;
        bt_d[1]=d_idx0; bt_h[1]=h_idx0; bt_w[1]=w_idx1; bt_s[1]=ds0*hs0*ws1;
        bt_d[2]=d_idx0; bt_h[2]=h_idx1; bt_w[2]=w_idx0; bt_s[2]=ds0*hs1*ws0;
        bt_d[3]=d_idx0; bt_h[3]=h_idx1; bt_w[3]=w_idx1; bt_s[3]=ds0*hs1*ws1;
        bt_d[4]=d_idx1; bt_h[4]=h_idx0; bt_w[4]=w_idx0; bt_s[4]=ds1*hs0*ws0;
        bt_d[5]=d_idx1; bt_h[5]=h_idx0; bt_w[5]=w_idx1; bt_s[5]=ds1*hs0*ws1;
        bt_d[6]=d_idx1; bt_h[6]=h_idx1; bt_w[6]=w_idx0; bt_s[6]=ds1*hs1*ws0;
        bt_d[7]=d_idx1; bt_h[7]=h_idx1; bt_w[7]=w_idx1; bt_s[7]=ds1*hs1*ws1;

        // Per-warp WMMA accumulator
        fragment<accumulator, 16, 16, 8, float> acc;
        fill_fragment(acc, 0.0f);

        // Inner loop over cin in chunks of 8
        for (int ci = 0; ci < C_loop; ci += 8) {
            // Phase 1: ALL 128 threads cooperate to fill A_buf[16][8]
            // 128 entries, 128 threads → 1 entry per thread
            if (tid < 128) {
                int tile_local = tid >> 3;   // 0..15
                int cin_off = tid & 7;       // 0..7
                int t_global = tile_base + tile_local;
                float v = 0.0f;
                if (t_global < T_total) {
                    int d0 = s_tile_d0[tile_local];
                    int h0 = s_tile_h0[tile_local];
                    int w0 = s_tile_w0[tile_local];
                    int cin_idx = ci + cin_off;
                    const float* cin_plane = inp_base + cin_idx * DHW;
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        int d = d0 + bt_d[k];
                        int h = h0 + bt_h[k];
                        int w = w0 + bt_w[k];
                        float val = (d >= 0 && d < D &&
                                     h >= 0 && h < H &&
                                     w >= 0 && w < W)
                                    ? cin_plane[d * HW + h * W + w] : 0.0f;
                        v += bt_s[k] * val;
                    }
                }
                s_A_buf[tile_local][cin_off] = v;
            }
            __syncthreads();

            // Phase 2: Each warp does independent WMMA
            if (warp_valid) {
                fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
                fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
                load_matrix_sync(a_frag, &s_A_buf[0][0], 8);
                load_matrix_sync(b_frag,
                    &U[a_eff * U_stride + ci * C_tile + cout_warp], C_tile);
                mma_sync(acc, a_frag, b_frag, acc);
            }
            __syncthreads();  // protect A_buf before next cin iteration overwrites it
        }

        // Phase 3: Store + A^T scatter (per warp)
        if (warp_valid) {
            store_matrix_sync(&s_result[warp_id][0][0], acc, 16, mem_row_major);
            __syncwarp();

            #pragma unroll
            for (int p = 0; p < 8; p++) {
                float m_val = s_result[warp_id][my_tile][my_cout_start + p];
                #pragma unroll
                for (int od = 0; od < 2; od++) {
                    float cd = AT_COEFF[od][ad];
                    if (cd == 0.0f) continue;
                    #pragma unroll
                    for (int oh = 0; oh < 2; oh++) {
                        float ch = AT_COEFF[oh][ah];
                        if (ch == 0.0f) continue;
                        float cd_ch = cd * ch;
                        #pragma unroll
                        for (int ow = 0; ow < 2; ow++) {
                            float cw = AT_COEFF[ow][aw];
                            if (cw == 0.0f) continue;
                            int pos = od * 4 + oh * 2 + ow;
                            out_reg[p * 8 + pos] += cd_ch * cw * m_val;
                        }
                    }
                }
            }
        }
        __syncthreads();  // protect s_A_buf before next alpha
    } // end alpha loop

    // =================== Write output to global memory ===================
    if (!warp_valid) return;

    float* out_base = output + (size_t)n * C_tile * DHW;
    int t_global = tile_base + my_tile;
    if (t_global < T_total) {
        int tw = t_global % T_w;
        int th = (t_global / T_w) % T_h;
        int td = t_global / (T_w * T_h);

        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int cout_idx = cout_warp + my_cout_start + p;
            if (cout_idx >= C_tile) continue;
            #pragma unroll
            for (int pos = 0; pos < 8; pos++) {
                int od = pos >> 2;
                int oh = (pos >> 1) & 1;
                int ow = pos & 1;
                int d_out = td * 2 + od;
                int h_out = th * 2 + oh;
                int w_out = tw * 2 + ow;
                if (d_out < D && h_out < H && w_out < W) {
                    float val = out_reg[p * 8 + pos];
                    if constexpr (!IS_BACKWARD) {
                        if (bias) val += bias[cout_idx];
                    }
                    out_base[cout_idx * DHW + d_out * HW + h_out * W + w_out] = val;
                }
            }
        }
    }
}

// =====================================================================
// Dispatch helpers
// =====================================================================

template <bool IS_BACKWARD>
static void launch_winograd(
    const float* input, const float* U, const float* bias,
    float* output,
    int C_loop, int C_tile,
    int N, int D, int H, int W,
    cudaStream_t stream)
{
    int T_d = (D + 1) / 2;
    int T_h = (H + 1) / 2;
    int T_w = (W + 1) / 2;
    int T_total = T_d * T_h * T_w;

    // v3 tensor core path: C_loop must be multiple of 8 (WMMA k=8),
    // C_tile must be multiple of 16 (WMMA n=16).
    if (C_loop % 8 == 0 && C_tile % 16 == 0) {
        constexpr int cout_per_block = V3_NW * 16;  // 64
        dim3 grid((T_total + 15) / 16, (C_tile + cout_per_block - 1) / cout_per_block, N);
        conv3d_winograd_v3_kernel<IS_BACKWARD><<<grid, V3_BLOCK, 0, stream>>>(
            input, U, bias, output,
            C_loop, C_tile, D, H, W,
            T_d, T_h, T_w, T_total);
    } else {
        // v2 fallback (FP32 scalar, cin-unrolled)
        constexpr int BT_VAL = 4;
        constexpr int BLOCK = BT_VAL * 64;

        int smem_cin2 = 2 * BT_VAL * TILE_STRIDE + 2 * 64 * (BC + 1);
        int smem_acc = BT_VAL * 64 * COUT_CHUNK;
        int smem_total = (smem_cin2 > smem_acc) ? smem_cin2 : smem_acc;
        int smem_bytes = smem_total * (int)sizeof(float);

        dim3 grid((T_total + BT_VAL - 1) / BT_VAL, (C_tile + BC - 1) / BC, N);
        if constexpr (!IS_BACKWARD) {
            conv3d_winograd_v2_forward<BT_VAL><<<grid, BLOCK, smem_bytes, stream>>>(
                input, U, bias, output,
                C_loop, C_tile, D, H, W,
                T_d, T_h, T_w, T_total);
        } else {
            conv3d_winograd_v2_backward<BT_VAL><<<grid, BLOCK, smem_bytes, stream>>>(
                input, U, bias, output,
                C_loop, C_tile, D, H, W,
                T_d, T_h, T_w, T_total);
        }
    }
    FORGE_CHECK(cudaGetLastError());
}

// =====================================================================
// Public API
// =====================================================================

void conv3d_winograd_filter_transform(
    const float* weight, float* U, float* U_t,
    int C_out, int C_in, cudaStream_t stream)
{
    int total = C_out * C_in;
    int block = 256;
    int grid = (total + block - 1) / block;
    winograd_filter_transform_kernel<<<grid, block, 0, stream>>>(weight, U, U_t, C_out, C_in);
    FORGE_CHECK(cudaGetLastError());
}

void conv3d_winograd_forward(
    const float* input, const float* U, const float* bias,
    float* output,
    int N, int C_in, int D, int H, int W, int C_out,
    cudaStream_t stream)
{
    launch_winograd<false>(input, U, bias, output, C_in, C_out, N, D, H, W, stream);
}

void conv3d_winograd_backward_data(
    const float* d_output, const float* U, float* d_input,
    int N, int C_in, int D, int H, int W, int C_out,
    cudaStream_t stream)
{
    launch_winograd<true>(d_output, U, nullptr, d_input, C_out, C_in, N, D, H, W, stream);
}

}} // namespace forge::nn
