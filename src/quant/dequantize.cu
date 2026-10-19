// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/quant/dequantize.cuh>
#include <algorithm>

namespace forge { namespace quant {

__global__ void fused_dequantize_bias_kernel(
    const int32_t* __restrict__ input,
    const float* __restrict__   act_scale,
    const __half* __restrict__  w_scale,
    const __half* __restrict__  bias,
    __half* __restrict__        output,
    int                         N)
{
    const int row = blockIdx.x;
    const int32_t* in_row  = input  + (size_t)row * N;
    __half*        out_row = output + (size_t)row * N;
    const float    a_scale = act_scale[row];

    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float val = static_cast<float>(in_row[col]) * a_scale * __half2float(w_scale[col]);
        if (bias != nullptr) {
            val += __half2float(bias[col]);
        }
        out_row[col] = __float2half(val);
    }
}

void fused_dequantize_bias(
    const int32_t* input,
    const float*   act_scale,
    const __half*  w_scale,
    const __half*  bias,
    __half*        output,
    int            M,
    int            N,
    cudaStream_t   stream)
{
    if (M <= 0 || N <= 0) return;
    int threads = min(256, ((N + 31) / 32) * 32);
    if (threads < 32) threads = 32;
    fused_dequantize_bias_kernel<<<M, threads, 0, stream>>>(
        input, act_scale, w_scale, bias, output, N);
}

}} // namespace forge::quant
