// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// INT8 Depthwise Conv2d forward (inference only).
//
// Direct sliding window — no im2col or GEMM.
// For 3×3 depthwise (K=9 taps), GEMM with M=1 per group is terrible.
// Instead, each thread computes one output pixel: loads the kH×kW patch,
// multiplies by INT8 weights, dequantizes, adds bias.
//
// FSR4 uses depthwise 3×3 in ConvNextBlocks with C ∈ {16, 32, 64}.

#include <forge/nn/int8_depthwise_conv2d.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace nn {

static inline __host__ __device__ int conv_out_size(int in_size, int pad, int dil, int kernel, int stride) {
    return (in_size + 2 * pad - dil * (kernel - 1) - 1) / stride + 1;
}

// One thread per output pixel (c, h_out, w_out).
// Each thread loads kH*kW input values and kH*kW weight values,
// computes a dot product, dequantizes, and adds bias.
static __global__ void int8_depthwise_conv2d_kernel(
    const __half* __restrict__ input,    // [C, H, W]
    const int8_t* __restrict__ weight,   // [C, kH*kW]
    const __half* __restrict__ w_scale,  // [C]
    const __half* __restrict__ bias,     // [C] or nullptr
    __half*       __restrict__ output,   // [C, H_out, W_out]
    int C, int H, int W,
    int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW,
    int H_out, int W_out)
{
    const int total = C * H_out * W_out;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x)
    {
        const int w_out = idx % W_out;
        const int tmp   = idx / W_out;
        const int h_out = tmp % H_out;
        const int c     = tmp / H_out;

        const float scale = __half2float(w_scale[c]);
        const int8_t* w_ptr = weight + c * kH * kW;

        float sum = 0.0f;
        for (int kh = 0; kh < kH; kh++) {
            const int h_in = h_out * strideH + kh * dilH - padH;
            if (h_in < 0 || h_in >= H) continue;

            for (int kw = 0; kw < kW; kw++) {
                const int w_in = w_out * strideW + kw * dilW - padW;
                if (w_in < 0 || w_in >= W) continue;

                float inp = __half2float(input[c * H * W + h_in * W + w_in]);
                float w   = static_cast<float>(w_ptr[kh * kW + kw]) * scale;
                sum += inp * w;
            }
        }

        if (bias != nullptr) sum += __half2float(bias[c]);
        output[c * H_out * W_out + h_out * W_out + w_out] = __float2half(sum);
    }
}

void int8_depthwise_conv2d_forward(
    const __half* input,
    const int8_t* weight,
    const __half* w_scale,
    const __half* bias,
    __half*       output,
    int N, int C, int H, int W,
    int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW,
    cudaStream_t  stream)
{
    if (N <= 0 || C <= 0) return;

    const int H_out = conv_out_size(H, padH, dilH, kH, strideH);
    const int W_out = conv_out_size(W, padW, dilW, kW, strideW);
    const int total = C * H_out * W_out;

    constexpr int BLOCK = 256;
    const int grid = internal::divup(total, BLOCK);

    for (int n = 0; n < N; n++) {
        const __half* input_n  = input  + (size_t)n * C * H * W;
        __half*       output_n = output + (size_t)n * C * H_out * W_out;

        int8_depthwise_conv2d_kernel<<<grid, BLOCK, 0, stream>>>(
            input_n, weight, w_scale, bias, output_n,
            C, H, W, kH, kW, padH, padW, strideH, strideW, dilH, dilW,
            H_out, W_out);
    }
}

}} // namespace forge::nn
