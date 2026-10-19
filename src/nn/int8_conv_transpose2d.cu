// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// INT8 ConvTranspose2d forward, specialized for k=2 s=2 (2× upsample).
//
// Since stride == kernel, each input pixel maps to a unique 2×2 output block.
// No overlap → clean GEMM + scatter approach:
//
//   1. Transpose input [C_in, H*W] → col[H*W, C_in]
//   2. Quantize col → INT8 + act_scale
//   3. Replicate w_scale[C_out] → w_scale_rep[4*C_out]
//   4. GEMM: col_int8[H*W, C_in] @ weight[4*C_out, C_in]^T → gemm_out[H*W, 4*C_out]
//      (bias=nullptr in GEMM — added in scatter step)
//   5. Scatter + bias: gemm_out[m, co*4+kh*2+kw] → output[co, 2*h+kh, 2*w+kw] + bias[co]
//
// Weight repacking (done once at model load time):
//   For standard ConvTranspose2d weight[C_in, C_out, kH, kW]:
//   repacked[co*4 + kh*2 + kw, ci] = weight[ci, co, kh, kw]

#include <forge/nn/int8_conv_transpose2d.cuh>
#include <forge/gemm/int8_gemm.cuh>
#include <forge/quant/quantize.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace nn {

// =====================================================================
// NCHW → NHWC transpose (input rearrangement)
// =====================================================================

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
// Replicate w_scale[C_out] → w_scale_rep[4*C_out]
// =====================================================================

static __global__ void replicate_scale_4x_kernel(
    const __half* __restrict__ src,   // [C_out]
    __half*       __restrict__ dst,   // [4*C_out]
    int C_out)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < 4 * C_out;
         idx += blockDim.x * gridDim.x)
    {
        dst[idx] = src[idx / 4];
    }
}

// =====================================================================
// Scatter + bias: gemm_out[H*W, 4*C_out] → output[C_out, 2*H, 2*W]
// =====================================================================
//
// gemm_out[m, n] where m = h_in * W + w_in, n = co * 4 + kh * 2 + kw
// → output[co, 2*h_in + kh, 2*w_in + kw]

static __global__ void scatter_bias_kernel(
    const __half* __restrict__ gemm_out,  // [H_in*W_in, 4*C_out]
    const __half* __restrict__ bias,      // [C_out] or nullptr
    __half*       __restrict__ output,    // [C_out, H_out, W_out] where H_out=2*H_in
    int H_in, int W_in, int C_out)
{
    const int H_out = 2 * H_in;
    const int W_out = 2 * W_in;
    const int M = H_in * W_in;
    const int total = M * 4 * C_out;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x)
    {
        const int n = idx % (4 * C_out);
        const int m = idx / (4 * C_out);

        const int h_in = m / W_in;
        const int w_in = m % W_in;

        const int co = n / 4;
        const int kh = (n % 4) / 2;
        const int kw = (n % 4) % 2;

        const int h_out = 2 * h_in + kh;
        const int w_out = 2 * w_in + kw;

        float val = __half2float(gemm_out[m * (4 * C_out) + n]);
        if (bias != nullptr) val += __half2float(bias[co]);
        output[co * H_out * W_out + h_out * W_out + w_out] = __float2half(val);
    }
}

// =====================================================================
// Workspace size
// =====================================================================

size_t int8_conv_transpose2d_k2s2_workspace_size(
    int C_in, int H, int W, int C_out)
{
    const int M = H * W;
    const int N_gemm = 4 * C_out;

    const size_t col_fp16_bytes     = (size_t)M * C_in * sizeof(__half);
    const size_t col_int8_bytes     = (size_t)M * C_in * sizeof(int8_t);
    const size_t act_scale_bytes    = (size_t)M * sizeof(float);
    const size_t w_scale_rep_bytes  = (size_t)N_gemm * sizeof(__half);
    const size_t gemm_out_bytes     = (size_t)M * N_gemm * sizeof(__half);

    return col_fp16_bytes + col_int8_bytes + act_scale_bytes
         + w_scale_rep_bytes + gemm_out_bytes;
}

// =====================================================================
// Forward
// =====================================================================

void int8_conv_transpose2d_k2s2_forward(
    const __half* input,
    const int8_t* weight,
    const __half* w_scale,
    const __half* bias,
    __half*       output,
    int N, int C_in, int H, int W,
    int C_out,
    void*         workspace,
    cudaStream_t  stream)
{
    if (N <= 0 || C_in <= 0 || C_out <= 0) return;

    const int M = H * W;           // spatial positions (input)
    const int K = C_in;            // GEMM K dimension
    const int N_gemm = 4 * C_out;  // GEMM N dimension (4 kernel positions × C_out)

    // Partition workspace
    __half*  col_fp16     = reinterpret_cast<__half*>(workspace);
    int8_t*  col_int8     = reinterpret_cast<int8_t*>(col_fp16 + (size_t)M * K);
    float*   act_scale    = reinterpret_cast<float*>(col_int8 + (size_t)M * K);
    __half*  w_scale_rep  = reinterpret_cast<__half*>(act_scale + M);
    __half*  gemm_out     = w_scale_rep + N_gemm;

    constexpr int BLOCK = 256;

    // Step 3 (once): replicate w_scale[C_out] → w_scale_rep[4*C_out]
    {
        const int grid = internal::divup(N_gemm, BLOCK);
        replicate_scale_4x_kernel<<<grid, BLOCK, 0, stream>>>(
            w_scale, w_scale_rep, C_out);
    }

    for (int n = 0; n < N; n++) {
        const __half* input_n  = input  + (size_t)n * C_in * H * W;
        __half*       output_n = output + (size_t)n * C_out * (2*H) * (2*W);

        // Step 1: transpose input [C_in, H*W] → col[H*W, C_in]
        {
            const int total = C_in * M;
            const int grid = internal::divup(total, BLOCK);
            nchw_to_nhwc_kernel<<<grid, BLOCK, 0, stream>>>(
                input_n, col_fp16, C_in, M);
        }

        // Step 2: quantize col → int8 + act_scale
        forge::quant::fused_quantize_activation(
            col_fp16, col_int8, act_scale, M, K, stream);

        // Step 4: GEMM (no bias — added in scatter)
        // col_int8[M, K] @ weight[N_gemm, K]^T → gemm_out[M, N_gemm]
        forge::gemm::int8_gemm(
            col_int8, weight, act_scale, w_scale_rep, nullptr,
            gemm_out, M, K, N_gemm, stream);

        // Step 5: scatter + bias → output_n[C_out, 2*H, 2*W]
        {
            const int total = M * N_gemm;
            const int grid = internal::divup(total, BLOCK);
            scatter_bias_kernel<<<grid, BLOCK, 0, stream>>>(
                gemm_out, bias, output_n, H, W, C_out);
        }
    }
}

}} // namespace forge::nn
