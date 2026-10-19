// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Implicit GEMM Conv3D — register-tiled, structured K-loop.
//
// Key optimizations over naive implicit GEMM:
//   1. Register tiling: each thread computes [TM x TN] outputs (not just 1)
//   2. Structured K-loop: iterates (cin, kd, kh, kw) explicitly, avoiding
//      expensive integer divmod for K-index decomposition
//   3. Precomputed spatial coords in shared memory (eliminates divmod in N)
//   4. Multiple BN dispatch for different spatial sizes
//
// Memory traffic analysis (32ch k=3, 32^3):
//   Implicit GEMM reads each input element 27x (once per kernel position).
//   Total: 32 * 32768 * 4 * 27 = 115 MB. This is 3x more than Winograd.
//   cuDNN likely uses Winograd for k=3, which explains its 3-4x advantage.
//   Our FP32 scalar GEMM achieves ~50% of memory-bound limit — similar
//   efficiency to cuDNN, just higher absolute traffic.

#include <forge/nn/conv3d_implicit.cuh>
#include <forge/error.h>

namespace forge { namespace nn {

static inline __host__ __device__ int conv_out_dim(int in, int pad, int dil, int k, int stride) {
    return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
}

// =====================================================================
// Forward kernel: register-tiled implicit GEMM with structured K-loop
// =====================================================================
//
// C[M, N] = A[M, K] * B[K, N]
//   M = C_out_pg, N = spatial_out, K = C_in_pg * kD * kH * kW
//   A = weight (explicit), B = input patches (implicit vol2col)
//
// Structured K-loop: instead of flat k -> decompose(cin,kd,kh,kw),
// we iterate over (cin, kd, kh, kw) directly and buffer into BK tiles.
// This eliminates ALL integer division/modulo from the hot path.

template <int BM, int BN, int BK, int TM, int TN>
__global__ void conv3d_implicit_fwd_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float*       __restrict__ output,
    int C_in_pg, int D, int H, int W,
    int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD, int dilH, int dilW,
    int C_out_pg, int K_total,
    int D_out, int H_out, int W_out, int spatial_out)
{
    constexpr int THREADS = (BM / TM) * (BN / TN);
    const int tid = threadIdx.x;
    const int thread_row = tid / (BN / TN);
    const int thread_col = tid % (BN / TN);

    const int m_base = blockIdx.x * BM;
    const int n_base = blockIdx.y * BN;

    // Accumulators in registers
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    // Shared memory for GEMM tiles
    __shared__ float s_A[BK][BM + 1];
    __shared__ float s_B[BK][BN + 1];

    // Precompute output spatial coords (avoids divmod in K-loop)
    __shared__ int s_d_out[BN], s_h_out[BN], s_w_out[BN];
    for (int i = tid; i < BN; i += THREADS) {
        int n = n_base + i;
        if (n < spatial_out) {
            s_w_out[i] = n % W_out;
            int tmp = n / W_out;
            s_h_out[i] = tmp % H_out;
            s_d_out[i] = tmp / H_out;
        }
    }
    __syncthreads();

    const int HW = H * W;
    const int DHW = D * HW;
    const int kHW = kH * kW;

    // Structured K-loop: iterate (cin, kd, kh, kw) explicitly.
    // Buffer BK rows before syncing and computing.
    int tile_pos = 0;
    int k_linear = 0;

    for (int cin = 0; cin < C_in_pg; cin++) {
        for (int kd = 0; kd < kD; kd++) {
            const int d_offset = kd * dilD - padD;
            for (int kh = 0; kh < kH; kh++) {
                const int h_offset = kh * dilH - padH;
                for (int kw = 0; kw < kW; kw++) {
                    const int w_offset = kw * dilW - padW;

                    // Weight row: s_A[tile_pos][bm]
                    if (tid < BM) {
                        int m = m_base + tid;
                        s_A[tile_pos][tid] = (m < C_out_pg)
                            ? weight[m * K_total + k_linear]
                            : 0.0f;
                    }

                    // Input row: s_B[tile_pos][bn] (using precomputed output coords)
                    if (tid < BN) {
                        float val = 0.0f;
                        int n = n_base + tid;
                        if (n < spatial_out) {
                            int d_in = s_d_out[tid] * strideD + d_offset;
                            int h_in = s_h_out[tid] * strideH + h_offset;
                            int w_in = s_w_out[tid] * strideW + w_offset;
                            if (d_in >= 0 && d_in < D &&
                                h_in >= 0 && h_in < H &&
                                w_in >= 0 && w_in < W) {
                                val = input[cin * DHW + d_in * HW + h_in * W + w_in];
                            }
                        }
                        s_B[tile_pos][tid] = val;
                    }

                    k_linear++;
                    tile_pos++;

                    if (tile_pos == BK) {
                        __syncthreads();

                        // Register-tiled outer product accumulation
                        #pragma unroll
                        for (int bk = 0; bk < BK; bk++) {
                            float a_frag[TM], b_frag[TN];
                            #pragma unroll
                            for (int tm = 0; tm < TM; tm++)
                                a_frag[tm] = s_A[bk][thread_row * TM + tm];
                            #pragma unroll
                            for (int tn = 0; tn < TN; tn++)
                                b_frag[tn] = s_B[bk][thread_col * TN + tn];
                            #pragma unroll
                            for (int tm = 0; tm < TM; tm++)
                                #pragma unroll
                                for (int tn = 0; tn < TN; tn++)
                                    acc[tm][tn] += a_frag[tm] * b_frag[tn];
                        }
                        __syncthreads();
                        tile_pos = 0;
                    }
                }
            }
        }
    }

    // Handle remaining K rows (tile_pos < BK)
    if (tile_pos > 0) {
        // Zero out unused rows so compute is safe
        for (int bk = tile_pos; bk < BK; bk++) {
            if (tid < BM) s_A[bk][tid] = 0.0f;
            if (tid < BN) s_B[bk][tid] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int bk = 0; bk < BK; bk++) {
            float a_frag[TM], b_frag[TN];
            #pragma unroll
            for (int tm = 0; tm < TM; tm++)
                a_frag[tm] = s_A[bk][thread_row * TM + tm];
            #pragma unroll
            for (int tn = 0; tn < TN; tn++)
                b_frag[tn] = s_B[bk][thread_col * TN + tn];
            #pragma unroll
            for (int tm = 0; tm < TM; tm++)
                #pragma unroll
                for (int tn = 0; tn < TN; tn++)
                    acc[tm][tn] += a_frag[tm] * b_frag[tn];
        }
    }

    // Write output
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
        int m = m_base + thread_row * TM + tm;
        if (m < C_out_pg) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                int n = n_base + thread_col * TN + tn;
                if (n < spatial_out) {
                    output[m * spatial_out + n] = acc[tm][tn];
                }
            }
        }
    }
}

// Bias kernel
static __global__ void conv3d_implicit_bias_kernel(
    float*       __restrict__ output,
    const float* __restrict__ bias,
    int total, int C_out, int spatial)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = (idx / spatial) % C_out;
    output[idx] += bias[c];
}

// =====================================================================
// Forward dispatch
// =====================================================================

void conv3d_implicit_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    int N, int C_in, int D, int H, int W,
    int C_out, int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD, int dilH, int dilW,
    int groups,
    cudaStream_t stream)
{
    int D_out = conv_out_dim(D, padD, dilD, kD, strideD);
    int H_out = conv_out_dim(H, padH, dilH, kH, strideH);
    int W_out = conv_out_dim(W, padW, dilW, kW, strideW);
    int spatial_out = D_out * H_out * W_out;

    int C_in_pg  = C_in / groups;
    int C_out_pg = C_out / groups;
    int K_total  = C_in_pg * kD * kH * kW;

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            const float* inp = input + ((size_t)n * C_in + g * C_in_pg) * D * H * W;
            float*       out = output + ((size_t)n * C_out + g * C_out_pg) * spatial_out;
            const float* wgt = weight + (size_t)g * C_out_pg * K_total;

            // Dispatch based on spatial size for optimal block count
            if (spatial_out >= 2048) {
                // Large spatial: BN=128, more work per block
                constexpr int BM = 32, BN = 128, BK = 8, TM = 4, TN = 4;
                constexpr int THREADS = (BM / TM) * (BN / TN);  // 256
                dim3 grid((C_out_pg + BM - 1) / BM,
                          (spatial_out + BN - 1) / BN);
                conv3d_implicit_fwd_kernel<BM, BN, BK, TM, TN>
                    <<<grid, THREADS, 0, stream>>>(
                        inp, wgt, out,
                        C_in_pg, D, H, W,
                        kD, kH, kW,
                        padD, padH, padW,
                        strideD, strideH, strideW,
                        dilD, dilH, dilW,
                        C_out_pg, K_total,
                        D_out, H_out, W_out, spatial_out);
            } else {
                // Small spatial: BN=32, more blocks for better GPU utilization
                constexpr int BM = 32, BN = 32, BK = 8, TM = 4, TN = 4;
                constexpr int THREADS = (BM / TM) * (BN / TN);  // 64
                dim3 grid((C_out_pg + BM - 1) / BM,
                          (spatial_out + BN - 1) / BN);
                conv3d_implicit_fwd_kernel<BM, BN, BK, TM, TN>
                    <<<grid, THREADS, 0, stream>>>(
                        inp, wgt, out,
                        C_in_pg, D, H, W,
                        kD, kH, kW,
                        padD, padH, padW,
                        strideD, strideH, strideW,
                        dilD, dilH, dilW,
                        C_out_pg, K_total,
                        D_out, H_out, W_out, spatial_out);
            }
        }
    }

    // Bias
    if (bias) {
        int total = N * C_out * spatial_out;
        int block = 256;
        int grid = (total + block - 1) / block;
        conv3d_implicit_bias_kernel<<<grid, block, 0, stream>>>(
            output, bias, total, C_out, spatial_out);
    }
}

// =====================================================================
// Backward data kernel: register-tiled, structured K-loop
// =====================================================================
//
// d_input[c_in, d, h, w] = sum_{cout, kd, kh, kw}
//   weight[cout, c_in, kd, kh, kw] * d_output[cout, d_out, h_out, w_out]
//
// GEMM: d_input[M, N] = weight_T[M, K] * d_output_gather[K, N]
//   M = C_in_pg, N = D*H*W, K = C_out_pg * kD*kH*kW

template <int BM, int BN, int BK, int TM, int TN>
__global__ void conv3d_implicit_bwd_data_kernel(
    const float* __restrict__ d_output,
    const float* __restrict__ weight,
    float*       __restrict__ d_input,
    int C_in_pg, int D, int H, int W,
    int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD, int dilH, int dilW,
    int C_out_pg, int K_total,
    int D_out, int H_out, int W_out, int spatial_out)
{
    constexpr int THREADS = (BM / TM) * (BN / TN);
    const int tid = threadIdx.x;
    const int thread_row = tid / (BN / TN);
    const int thread_col = tid % (BN / TN);

    const int m_base = blockIdx.x * BM;
    const int n_base = blockIdx.y * BN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    const int HW = H * W;
    const int spatial_in = D * HW;
    const int kHW = kH * kW;
    const int kDHW = kD * kHW;

    __shared__ float s_A[BK][BM + 1];
    __shared__ float s_B[BK][BN + 1];

    // Precompute input spatial coords
    __shared__ int s_d_in[BN], s_h_in[BN], s_w_in[BN];
    for (int i = tid; i < BN; i += THREADS) {
        int n = n_base + i;
        if (n < spatial_in) {
            s_w_in[i] = n % W;
            int tmp = n / W;
            s_h_in[i] = tmp % H;
            s_d_in[i] = tmp / H;
        }
    }
    __syncthreads();

    // Structured K-loop: iterate (cout, kd, kh, kw) explicitly
    int tile_pos = 0;

    for (int cout = 0; cout < C_out_pg; cout++) {
        for (int kd = 0; kd < kD; kd++) {
            const int d_off_base = padD - kd * dilD;
            for (int kh = 0; kh < kH; kh++) {
                const int h_off_base = padH - kh * dilH;
                for (int kw = 0; kw < kW; kw++) {
                    const int w_off_base = padW - kw * dilW;
                    const int k_linear_wt = kd * kHW + kh * kW + kw;

                    // Weight^T row: weight[cout, c_in, kd, kh, kw]
                    if (tid < BM) {
                        int c_in = m_base + tid;
                        s_A[tile_pos][tid] = (c_in < C_in_pg)
                            ? weight[(cout * C_in_pg + c_in) * kDHW + k_linear_wt]
                            : 0.0f;
                    }

                    // d_output gather
                    if (tid < BN) {
                        float val = 0.0f;
                        int n = n_base + tid;
                        if (n < spatial_in) {
                            int d_off = s_d_in[tid] + d_off_base;
                            int h_off = s_h_in[tid] + h_off_base;
                            int w_off = s_w_in[tid] + w_off_base;

                            if (d_off >= 0 && d_off % strideD == 0 &&
                                h_off >= 0 && h_off % strideH == 0 &&
                                w_off >= 0 && w_off % strideW == 0) {
                                int d_out = d_off / strideD;
                                int h_out = h_off / strideH;
                                int w_out = w_off / strideW;
                                if (d_out < D_out && h_out < H_out && w_out < W_out) {
                                    val = d_output[cout * spatial_out +
                                                   d_out * (H_out * W_out) + h_out * W_out + w_out];
                                }
                            }
                        }
                        s_B[tile_pos][tid] = val;
                    }

                    tile_pos++;
                    if (tile_pos == BK) {
                        __syncthreads();
                        #pragma unroll
                        for (int bk = 0; bk < BK; bk++) {
                            float a_frag[TM], b_frag[TN];
                            #pragma unroll
                            for (int tm = 0; tm < TM; tm++)
                                a_frag[tm] = s_A[bk][thread_row * TM + tm];
                            #pragma unroll
                            for (int tn = 0; tn < TN; tn++)
                                b_frag[tn] = s_B[bk][thread_col * TN + tn];
                            #pragma unroll
                            for (int tm = 0; tm < TM; tm++)
                                #pragma unroll
                                for (int tn = 0; tn < TN; tn++)
                                    acc[tm][tn] += a_frag[tm] * b_frag[tn];
                        }
                        __syncthreads();
                        tile_pos = 0;
                    }
                }
            }
        }
    }

    // Handle remainder
    if (tile_pos > 0) {
        for (int bk = tile_pos; bk < BK; bk++) {
            if (tid < BM) s_A[bk][tid] = 0.0f;
            if (tid < BN) s_B[bk][tid] = 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int bk = 0; bk < BK; bk++) {
            float a_frag[TM], b_frag[TN];
            #pragma unroll
            for (int tm = 0; tm < TM; tm++)
                a_frag[tm] = s_A[bk][thread_row * TM + tm];
            #pragma unroll
            for (int tn = 0; tn < TN; tn++)
                b_frag[tn] = s_B[bk][thread_col * TN + tn];
            #pragma unroll
            for (int tm = 0; tm < TM; tm++)
                #pragma unroll
                for (int tn = 0; tn < TN; tn++)
                    acc[tm][tn] += a_frag[tm] * b_frag[tn];
        }
    }

    // Write d_input
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
        int c_in = m_base + thread_row * TM + tm;
        if (c_in < C_in_pg) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                int n = n_base + thread_col * TN + tn;
                if (n < spatial_in) {
                    d_input[c_in * spatial_in + n] = acc[tm][tn];
                }
            }
        }
    }
}

// =====================================================================
// Backward data dispatch
// =====================================================================

void conv3d_implicit_backward_data(
    const float* d_output,
    const float* weight,
    float*       d_input,
    int N, int C_in, int D, int H, int W,
    int C_out, int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD, int dilH, int dilW,
    int groups,
    cudaStream_t stream)
{
    int D_out = conv_out_dim(D, padD, dilD, kD, strideD);
    int H_out = conv_out_dim(H, padH, dilH, kH, strideH);
    int W_out = conv_out_dim(W, padW, dilW, kW, strideW);
    int spatial_out = D_out * H_out * W_out;
    int spatial_in  = D * H * W;

    int C_in_pg  = C_in / groups;
    int C_out_pg = C_out / groups;
    int K_total  = C_in_pg * kD * kH * kW;

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            const float* d_out_ptr = d_output + ((size_t)n * C_out + g * C_out_pg) * spatial_out;
            float*       d_inp_ptr = d_input + ((size_t)n * C_in + g * C_in_pg) * spatial_in;
            const float* wgt_ptr   = weight + (size_t)g * C_out_pg * K_total;

            if (spatial_in >= 2048) {
                constexpr int BM = 32, BN = 128, BK = 8, TM = 4, TN = 4;
                constexpr int THREADS = (BM / TM) * (BN / TN);
                dim3 grid((C_in_pg + BM - 1) / BM,
                          (spatial_in + BN - 1) / BN);
                conv3d_implicit_bwd_data_kernel<BM, BN, BK, TM, TN>
                    <<<grid, THREADS, 0, stream>>>(
                        d_out_ptr, wgt_ptr, d_inp_ptr,
                        C_in_pg, D, H, W,
                        kD, kH, kW,
                        padD, padH, padW,
                        strideD, strideH, strideW,
                        dilD, dilH, dilW,
                        C_out_pg, K_total,
                        D_out, H_out, W_out, spatial_out);
            } else {
                constexpr int BM = 32, BN = 32, BK = 8, TM = 4, TN = 4;
                constexpr int THREADS = (BM / TM) * (BN / TN);
                dim3 grid((C_in_pg + BM - 1) / BM,
                          (spatial_in + BN - 1) / BN);
                conv3d_implicit_bwd_data_kernel<BM, BN, BK, TM, TN>
                    <<<grid, THREADS, 0, stream>>>(
                        d_out_ptr, wgt_ptr, d_inp_ptr,
                        C_in_pg, D, H, W,
                        kD, kH, kW,
                        padD, padH, padW,
                        strideD, strideH, strideW,
                        dilD, dilH, dilW,
                        C_out_pg, K_total,
                        D_out, H_out, W_out, spatial_out);
            }
        }
    }
}

}} // namespace forge::nn
