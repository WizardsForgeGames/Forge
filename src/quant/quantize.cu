// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/quant/quantize.cuh>
#include <algorithm>

namespace forge { namespace quant {

__global__ void fused_quantize_activation_kernel(
    const __half* __restrict__ input,
    int8_t* __restrict__       output,
    float* __restrict__        scales,
    int                        K)
{
    const int row = blockIdx.x;
    const __half* in_row  = input  + (size_t)row * K;
    int8_t*       out_row = output + (size_t)row * K;

    // Pass 1: find max|x| via parallel reduction
    float thread_max = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float val = __half2float(in_row[k]);
        thread_max = fmaxf(thread_max, fabsf(val));
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    // Block-level reduction via shared memory
    __shared__ float warp_maxes[32];
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    if (lane_id == 0) warp_maxes[warp_id] = thread_max;
    __syncthreads();

    if (warp_id == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float val = (lane_id < num_warps) ? warp_maxes[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (lane_id == 0) warp_maxes[0] = val;
    }
    __syncthreads();

    const float row_max   = warp_maxes[0];
    const float scale     = fmaxf(row_max, 1e-8f) / 127.0f;
    if (threadIdx.x == 0) scales[row] = scale;
    const float inv_scale = 1.0f / scale;

    // Pass 2: quantize
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float val = __half2float(in_row[k]);
        int q = __float2int_rn(val * inv_scale);
        q = max(-128, min(127, q));
        out_row[k] = static_cast<int8_t>(q);
    }
}

void fused_quantize_activation(
    const __half* input,
    int8_t*       output,
    float*        scales,
    int           M,
    int           K,
    cudaStream_t  stream)
{
    if (M <= 0 || K <= 0) return;
    int threads = min(256, ((K + 31) / 32) * 32);
    if (threads < 32) threads = 32;
    fused_quantize_activation_kernel<<<M, threads, 0, stream>>>(input, output, scales, K);
}

}} // namespace forge::quant
