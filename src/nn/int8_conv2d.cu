// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// INT8 Conv2d forward (inference only) via im2col + unified INT8 GEMM.
//
// Pipeline:
//   1. im2col: FP16 input[C_in, H, W] → col[H_out*W_out, C_in*kH*kW]
//      (transposed layout — each row is one spatial position's patch,
//       matching the A[M,K] layout expected by the INT8 GEMM)
//   2. Quantize: per-row absmax, col_fp16 → col_int8 + act_scale
//   3. INT8 GEMM: col_int8[M,K] @ weight[C_out,K]^T → gemm_out[M,C_out]
//      with fused dequant (act_scale * w_scale) + bias
//   4. Transpose: gemm_out[H_out*W_out, C_out] → output[C_out, H_out*W_out]
//
// Pointwise (1×1, stride=1, no padding) conv special-cases step 1 as a
// simple NCHW→NHWC transpose, avoiding im2col overhead.

#include <forge/nn/int8_conv2d.cuh>
#include <forge/gemm/int8_gemm.cuh>
#include <forge/quant/quantize.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace nn {

// =====================================================================
// Helpers
// =====================================================================

static inline int conv_out_size(int in_size, int pad, int dil, int kernel, int stride) {
    return (in_size + 2 * pad - dil * (kernel - 1) - 1) / stride + 1;
}

// =====================================================================
// im2col for FP16 → transposed layout [M, K]
// =====================================================================
//
// Standard im2col produces col[K, M] where K=C_in*kH*kW, M=H_out*W_out.
// We produce the transpose: col[M, K], so each row is one patch.
// This lets the INT8 GEMM treat it directly as A[M, K].

static __global__ void im2col_fp16_kernel(
    const __half* __restrict__ input,   // [C_in, H, W]
    __half*       __restrict__ col,     // [M, K] = [H_out*W_out, C_in*kH*kW]
    int C_in, int H, int W,
    int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW,
    int H_out, int W_out)
{
    const int K = C_in * kH * kW;
    const int M = H_out * W_out;
    const int total = M * K;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x)
    {
        const int c_kh_kw = idx % K;     // filter tap index
        const int spatial  = idx / K;    // spatial position

        const int h_out = spatial / W_out;
        const int w_out = spatial % W_out;

        const int kw = c_kh_kw % kW;
        const int tmp = c_kh_kw / kW;
        const int kh = tmp % kH;
        const int c  = tmp / kH;

        const int h_in = h_out * strideH + kh * dilH - padH;
        const int w_in = w_out * strideW + kw * dilW - padW;

        __half val = __float2half(0.0f);
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            val = input[c * H * W + h_in * W + w_in];
        }
        col[spatial * K + c_kh_kw] = val;
    }
}

// =====================================================================
// NCHW → NHWC transpose (for pointwise 1×1 conv)
// =====================================================================
//
// Transposes [C, spatial] → [spatial, C] so each row is one pixel's channels.
// Avoids full im2col when kH=kW=1, stride=1, pad=0.

static __global__ void nchw_to_nhwc_kernel(
    const __half* __restrict__ src,  // [C, spatial]
    __half*       __restrict__ dst,  // [spatial, C]
    int C, int spatial)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < C * spatial;
         idx += blockDim.x * gridDim.x)
    {
        const int c = idx / spatial;
        const int s = idx % spatial;
        dst[s * C + c] = src[idx];
    }
}

// =====================================================================
// Transpose [M, N] → [N, M] (GEMM output → NCHW)
// =====================================================================

static __global__ void transpose_2d_kernel(
    const __half* __restrict__ src,  // [M, N]
    __half*       __restrict__ dst,  // [N, M]
    int M, int N)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < M * N;
         idx += blockDim.x * gridDim.x)
    {
        const int m = idx / N;
        const int n = idx % N;
        dst[n * M + m] = src[idx];
    }
}

// =====================================================================
// Workspace size
// =====================================================================

size_t int8_conv2d_workspace_size(
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW)
{
    const int H_out = conv_out_size(H, padH, dilH, kH, strideH);
    const int W_out = conv_out_size(W, padW, dilW, kW, strideW);
    const int M = H_out * W_out;
    const int K = C_in * kH * kW;

    // Layout: [col_fp16 | col_int8 | act_scale | gemm_out]
    const size_t col_fp16_bytes  = (size_t)M * K * sizeof(__half);
    const size_t col_int8_bytes  = (size_t)M * K * sizeof(int8_t);
    const size_t act_scale_bytes = (size_t)M * sizeof(float);
    const size_t gemm_out_bytes  = (size_t)M * C_out * sizeof(__half);

    return col_fp16_bytes + col_int8_bytes + act_scale_bytes + gemm_out_bytes;
}

// =====================================================================
// int8_conv2d_forward
// =====================================================================

void int8_conv2d_forward(
    const __half* input,
    const int8_t* weight,
    const __half* w_scale,
    const __half* bias,
    __half*       output,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW,
    void*         workspace,
    cudaStream_t  stream)
{
    if (N <= 0 || C_in <= 0 || C_out <= 0) return;

    const int H_out = conv_out_size(H, padH, dilH, kH, strideH);
    const int W_out = conv_out_size(W, padW, dilW, kW, strideW);
    const int M = H_out * W_out;    // spatial positions per sample
    const int K = C_in * kH * kW;   // filter taps

    // Partition workspace
    __half*  col_fp16  = reinterpret_cast<__half*>(workspace);
    int8_t*  col_int8  = reinterpret_cast<int8_t*>(col_fp16 + (size_t)M * K);
    float*   act_scale = reinterpret_cast<float*>(col_int8 + (size_t)M * K);
    __half*  gemm_out  = reinterpret_cast<__half*>(act_scale + M);

    constexpr int BLOCK = 256;
    const bool is_pointwise = (kH == 1 && kW == 1 &&
                               strideH == 1 && strideW == 1 &&
                               padH == 0 && padW == 0);

    for (int n = 0; n < N; ++n) {
        const __half* input_n  = input  + (size_t)n * C_in * H * W;
        __half*       output_n = output + (size_t)n * C_out * H_out * W_out;

        // Step 1: Build column matrix col_fp16[M, K]
        if (is_pointwise) {
            // 1×1 conv: transpose [C_in, H*W] → [H*W, C_in]
            const int total = C_in * M;
            const int grid = internal::divup(total, BLOCK);
            nchw_to_nhwc_kernel<<<grid, BLOCK, 0, stream>>>(
                input_n, col_fp16, C_in, M);
        } else {
            // General im2col
            const int total = M * K;
            const int grid = internal::divup(total, BLOCK);
            im2col_fp16_kernel<<<grid, BLOCK, 0, stream>>>(
                input_n, col_fp16,
                C_in, H, W,
                kH, kW, padH, padW, strideH, strideW, dilH, dilW,
                H_out, W_out);
        }

        // Step 2: Quantize col_fp16[M, K] → col_int8[M, K] + act_scale[M]
        forge::quant::fused_quantize_activation(
            col_fp16, col_int8, act_scale, M, K, stream);

        // Step 3: INT8 GEMM with fused dequant + bias
        // col_int8[M, K] @ weight[C_out, K]^T → gemm_out[M, C_out]
        forge::gemm::int8_gemm(
            col_int8, weight, act_scale, w_scale, bias,
            gemm_out, M, K, C_out, stream);

        // Step 4: Transpose gemm_out[M, C_out] → output_n[C_out, M]
        {
            const int total = M * C_out;
            const int grid = internal::divup(total, BLOCK);
            transpose_2d_kernel<<<grid, BLOCK, 0, stream>>>(
                gemm_out, output_n, M, C_out);
        }
    }
}

}} // namespace forge::nn
