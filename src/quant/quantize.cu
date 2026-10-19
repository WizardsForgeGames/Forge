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

// Single-pass variant: the row stays in registers as 16-byte vectors (VPT per thread),
// so the fp16 activation is read from memory once instead of twice, with 16-byte loads
// and 8-byte stores. Same arithmetic as fused_quantize_activation_kernel (bit-identical
// output). Requires K % 8 == 0 and K <= VPT * 8 * blockDim.x. STREAM: evict-first loads,
// used when the input is larger than L2 (it cannot stay resident, and should not flush the
// int8 output the GEMM reads next); smaller inputs use plain loads so sibling layers fed
// the same activation (e.g. Q/K/V projections) can re-hit it in L2.
template <int VPT, bool STREAM>
__global__ void quantize_activation_vec_kernel(
    const __half* __restrict__ input,
    int8_t* __restrict__       output,
    float* __restrict__        scales,
    int                        K)
{
    const int row  = blockIdx.x;
    const int nvec = K / 8;
    const uint4* in_row = reinterpret_cast<const uint4*>(input + (size_t)row * K);
    uint2* out_row = reinterpret_cast<uint2*>(output + (size_t)row * K);

    uint4 v[VPT];
    float thread_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < VPT; i++) {
        const int idx = threadIdx.x + i * blockDim.x;
        if (idx < nvec) {
            v[i] = STREAM ? __ldcs(in_row + idx) : in_row[idx];
            const __half2* h = reinterpret_cast<const __half2*>(&v[i]);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 f = __half22float2(h[j]);
                thread_max = fmaxf(thread_max, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
        }
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    __shared__ float warp_maxes[32];
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) warp_maxes[warp_id] = thread_max;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? warp_maxes[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        if (lane_id == 0) warp_maxes[0] = val;
    }
    __syncthreads();

    const float scale = fmaxf(warp_maxes[0], 1e-8f) / 127.0f;
    if (threadIdx.x == 0) scales[row] = scale;
    const float inv_scale = 1.0f / scale;

    #pragma unroll
    for (int i = 0; i < VPT; i++) {
        const int idx = threadIdx.x + i * blockDim.x;
        if (idx < nvec) {
            const __half2* h = reinterpret_cast<const __half2*>(&v[i]);
            uint32_t packed[2] = {0u, 0u};
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 f = __half22float2(h[j]);
                const int q0 = max(-128, min(127, __float2int_rn(f.x * inv_scale)));
                const int q1 = max(-128, min(127, __float2int_rn(f.y * inv_scale)));
                packed[j >> 1] |= ((uint32_t)(q0 & 0xFF) << (16 * (j & 1)))
                                | ((uint32_t)(q1 & 0xFF) << (16 * (j & 1) + 8));
            }
            out_row[idx] = make_uint2(packed[0], packed[1]);
        }
    }
}

namespace {

// L2 size of the current device, queried once per device.
size_t l2_cache_bytes() {
    static int cache[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    int& l2 = cache[dev & 63];
    if (l2 == 0) {
        cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
        if (l2 <= 0) l2 = 1;
    }
    return (size_t)l2;
}

template <int VPT>
void launch_quantize_vec(const __half* input, int8_t* output, float* scales,
                         int M, int K, int threads, bool stream_loads, cudaStream_t stream)
{
    if (stream_loads)
        quantize_activation_vec_kernel<VPT, true><<<M, threads, 0, stream>>>(input, output, scales, K);
    else
        quantize_activation_vec_kernel<VPT, false><<<M, threads, 0, stream>>>(input, output, scales, K);
}

} // namespace

void fused_quantize_activation(
    const __half* input,
    int8_t*       output,
    float*        scales,
    int           M,
    int           K,
    cudaStream_t  stream)
{
    if (M <= 0 || K <= 0) return;
    constexpr int kMaxThreads = 256, kMaxVpt = 4;
    const bool vec_ok = (K % 8 == 0) && (K <= kMaxVpt * 8 * kMaxThreads)
                     && (reinterpret_cast<uintptr_t>(input)  % 16 == 0)
                     && (reinterpret_cast<uintptr_t>(output) % 8  == 0);
    if (vec_ok) {
        const int nvec    = K / 8;
        const int threads = std::min(kMaxThreads, ((nvec + 31) / 32) * 32);
        const int vpt     = (nvec + threads - 1) / threads;          // 1..4
        const bool big    = (size_t)M * K * sizeof(__half) > l2_cache_bytes();
        switch (vpt) {
        case 1:  launch_quantize_vec<1>(input, output, scales, M, K, threads, big, stream); break;
        case 2:  launch_quantize_vec<2>(input, output, scales, M, K, threads, big, stream); break;
        case 3:  launch_quantize_vec<3>(input, output, scales, M, K, threads, big, stream); break;
        default: launch_quantize_vec<4>(input, output, scales, M, K, threads, big, stream); break;
        }
        return;
    }
    int threads = std::min(256, ((K + 31) / 32) * 32);
    if (threads < 32) threads = 32;
    fused_quantize_activation_kernel<<<M, threads, 0, stream>>>(input, output, scales, K);
}

}} // namespace forge::quant
