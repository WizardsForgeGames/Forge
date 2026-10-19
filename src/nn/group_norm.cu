// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Fused GroupNorm + GELU kernel implementation.
// One block per (sample, group) pair.  Two-pass approach:
//   Pass 1: reduce sum and sum_sq → mean, rstd
//   Pass 2: normalize, apply affine, apply GELU, write output
//
// Backward: recompute GroupNorm output on-the-fly to avoid storing activations.
#include <forge/nn/group_norm.cuh>
#include <forge/primitives/activation.cuh>

namespace forge { namespace nn {

// =====================================================================
// Warp-level reduction helpers
// =====================================================================

static __device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// =====================================================================
// Forward kernel
// =====================================================================
//
// Grid: (N * G) blocks.  Each block handles M = (C/G) * spatial elements.
// Block: THREADS threads (256 or 512).

template <int THREADS>
static __global__ void group_norm_gelu_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float*       __restrict__ y,
    float*       __restrict__ mean_out,
    float*       __restrict__ rstd_out,
    int C, int spatial, int G, float eps)
{
    const int ng = blockIdx.x;            // flat (n, g) index
    const int n  = ng / G;
    const int g  = ng % G;
    const int C_g = C / G;               // channels per group
    const int M = C_g * spatial;          // elements per group

    // Base pointer for this sample's group in the input/output
    const int group_offset = (n * C + g * C_g) * spatial;
    const float* x_ptr = x + group_offset;
    float*       y_ptr = y + group_offset;

    // ---- Pass 1: compute mean and variance ----
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < M; i += THREADS) {
        float val = x_ptr[i];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Warp-level reduction
    local_sum = warp_reduce_sum(local_sum);
    local_sum_sq = warp_reduce_sum(local_sum_sq);

    // Cross-warp reduction via shared memory
    constexpr int NUM_WARPS = THREADS / 32;
    __shared__ float s_sum[NUM_WARPS];
    __shared__ float s_sum_sq[NUM_WARPS];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        s_sum[warp_id] = local_sum;
        s_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        float sum_val = (lane_id < NUM_WARPS) ? s_sum[lane_id] : 0.0f;
        float sq_val  = (lane_id < NUM_WARPS) ? s_sum_sq[lane_id] : 0.0f;
        sum_val = warp_reduce_sum(sum_val);
        sq_val  = warp_reduce_sum(sq_val);

        if (lane_id == 0) {
            float mean = sum_val / M;
            float var = sq_val / M - mean * mean;
            float rstd = rsqrtf(var + eps);
            s_sum[0] = mean;
            s_sum_sq[0] = rstd;
            mean_out[ng] = mean;
            rstd_out[ng] = rstd;
        }
    }
    __syncthreads();

    float mean = s_sum[0];
    float rstd = s_sum_sq[0];

    // ---- Pass 2: normalize + affine + GELU ----
    for (int i = threadIdx.x; i < M; i += THREADS) {
        float val = x_ptr[i];
        float x_hat = (val - mean) * rstd;

        // Per-channel affine: c = g * C_g + (i / spatial)
        int c = g * C_g + i / spatial;
        float gn_out = gamma[c] * x_hat + beta[c];
        y_ptr[i] = primitives::gelu(gn_out);
    }
}

// =====================================================================
// Backward kernel
// =====================================================================
//
// Recomputes GroupNorm output on-the-fly to compute GELU backward.
// Three passes:
//   Pass 1: compute ds, db for input gradient reduction, and accumulate
//           per-channel d_gamma/d_beta in registers (not atomicAdd per-element).
//   Pass 1b: block-reduce per-channel d_gamma/d_beta, one atomicAdd per block.
//   Pass 2: compute d_input = rstd * (dz_eff - (1/M) * (db + x_hat * ds))
//
// C_g (channels per group) is typically 4-16 for our U-Net configs.
// We keep per-channel accumulators in registers to avoid atomicAdd storms.

static constexpr int MAX_C_PER_GROUP = 32;  // max channels per group

template <int THREADS>
static __global__ void group_norm_gelu_bwd_kernel(
    const float* __restrict__ d_output,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean_in,
    const float* __restrict__ rstd_in,
    float*       __restrict__ d_input,
    float*       __restrict__ d_gamma,
    float*       __restrict__ d_beta,
    int C, int spatial, int G)
{
    const int ng = blockIdx.x;
    const int n  = ng / G;
    const int g  = ng % G;
    const int C_g = C / G;
    const int M = C_g * spatial;

    const int group_offset = (n * C + g * C_g) * spatial;
    const float* x_ptr   = x + group_offset;
    const float* dy_ptr  = d_output + group_offset;
    float*       dx_ptr  = d_input + group_offset;

    float mean = mean_in[ng];
    float rstd = rstd_in[ng];

    // ---- Pass 1: compute ds, db and local per-channel d_gamma/d_beta ----
    float local_ds = 0.0f;
    float local_db = 0.0f;

    // Per-channel accumulators in registers (C_g is small: 4-16)
    float local_dg[MAX_C_PER_GROUP];
    float local_dbt[MAX_C_PER_GROUP];
    for (int c = 0; c < C_g; c++) {
        local_dg[c] = 0.0f;
        local_dbt[c] = 0.0f;
    }

    for (int i = threadIdx.x; i < M; i += THREADS) {
        float val = x_ptr[i];
        float x_hat = (val - mean) * rstd;
        int c_local = i / spatial;
        int c_global = g * C_g + c_local;
        float gn_out = gamma[c_global] * x_hat + beta[c_global];

        float dy = dy_ptr[i];
        float dz = primitives::gelu_grad(gn_out, dy);

        // Accumulate per-channel in registers (no atomicAdd!)
        local_dg[c_local] += dz * x_hat;
        local_dbt[c_local] += dz;

        // For input gradient
        float dz_eff = dz * gamma[c_global];
        local_ds += dz_eff * x_hat;
        local_db += dz_eff;
    }

    // ---- Block reduction for ds, db ----
    local_ds = warp_reduce_sum(local_ds);
    local_db = warp_reduce_sum(local_db);

    constexpr int NUM_WARPS = THREADS / 32;
    // Shared memory for ds/db reduction + per-channel dg/db reduction
    __shared__ float s_reduce[NUM_WARPS * 2];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        s_reduce[warp_id] = local_ds;
        s_reduce[NUM_WARPS + warp_id] = local_db;
    }
    __syncthreads();

    if (warp_id == 0) {
        float ds = (lane_id < NUM_WARPS) ? s_reduce[lane_id] : 0.0f;
        float db = (lane_id < NUM_WARPS) ? s_reduce[NUM_WARPS + lane_id] : 0.0f;
        ds = warp_reduce_sum(ds);
        db = warp_reduce_sum(db);
        if (lane_id == 0) {
            s_reduce[0] = ds;
            s_reduce[1] = db;
        }
    }
    __syncthreads();

    float ds = s_reduce[0];
    float db = s_reduce[1];
    float inv_M = 1.0f / M;

    // ---- Block reduction for per-channel d_gamma/d_beta ----
    // Reduce across threads, then one atomicAdd per block per channel.
    // Use shared memory for per-channel reduction.
    __shared__ float s_dg[MAX_C_PER_GROUP];
    __shared__ float s_dbt[MAX_C_PER_GROUP];

    for (int c = 0; c < C_g; c++) {
        // Warp reduce
        float warp_dg = warp_reduce_sum(local_dg[c]);
        float warp_dbt = warp_reduce_sum(local_dbt[c]);

        if (lane_id == 0) {
            s_reduce[warp_id] = warp_dg;
            s_reduce[NUM_WARPS + warp_id] = warp_dbt;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float sum_dg = 0.0f, sum_dbt = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                sum_dg += s_reduce[w];
                sum_dbt += s_reduce[NUM_WARPS + w];
            }
            atomicAdd(&d_gamma[g * C_g + c], sum_dg);
            atomicAdd(&d_beta[g * C_g + c], sum_dbt);
        }
        __syncthreads();
    }

    // ---- Pass 2: compute d_input ----
    for (int i = threadIdx.x; i < M; i += THREADS) {
        float val = x_ptr[i];
        float x_hat = (val - mean) * rstd;
        int c_global = g * C_g + i / spatial;
        float gn_out = gamma[c_global] * x_hat + beta[c_global];

        float dy = dy_ptr[i];
        float dz = primitives::gelu_grad(gn_out, dy);
        float dz_eff = dz * gamma[c_global];

        dx_ptr[i] = rstd * (dz_eff - inv_M * (db + x_hat * ds));
    }
}

// =====================================================================
// Public API
// =====================================================================

void group_norm_gelu_forward(
    const float* x,
    const float* gamma,
    const float* beta,
    float*       y,
    float*       mean,
    float*       rstd,
    int N, int C, int spatial,
    int G,
    float eps,
    cudaStream_t stream)
{
    int blocks = N * G;
    if (blocks <= 0) return;

    int C_g = C / G;
    int M = C_g * spatial;

    // Choose block size based on elements per group
    if (M >= 4096) {
        group_norm_gelu_fwd_kernel<512><<<blocks, 512, 0, stream>>>(
            x, gamma, beta, y, mean, rstd, C, spatial, G, eps);
    } else {
        group_norm_gelu_fwd_kernel<256><<<blocks, 256, 0, stream>>>(
            x, gamma, beta, y, mean, rstd, C, spatial, G, eps);
    }
}

void group_norm_gelu_backward(
    const float* d_output,
    const float* x,
    const float* gamma,
    const float* beta,
    const float* mean,
    const float* rstd,
    float*       d_input,
    float*       d_gamma,
    float*       d_beta,
    int N, int C, int spatial,
    int G,
    cudaStream_t stream)
{
    int blocks = N * G;
    if (blocks <= 0) return;

    int C_g = C / G;
    int M = C_g * spatial;

    if (M >= 4096) {
        group_norm_gelu_bwd_kernel<512><<<blocks, 512, 0, stream>>>(
            d_output, x, gamma, beta, mean, rstd,
            d_input, d_gamma, d_beta, C, spatial, G);
    } else {
        group_norm_gelu_bwd_kernel<256><<<blocks, 256, 0, stream>>>(
            d_output, x, gamma, beta, mean, rstd,
            d_input, d_gamma, d_beta, C, spatial, G);
    }
}

// =====================================================================
// Standalone GroupNorm (no GELU fusion)
// =====================================================================

template <int THREADS>
static __global__ void group_norm_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float*       __restrict__ y,
    float*       __restrict__ mean_out,
    float*       __restrict__ rstd_out,
    int C, int spatial, int G, float eps)
{
    const int ng = blockIdx.x;
    const int n  = ng / G;
    const int g  = ng % G;
    const int C_g = C / G;
    const int M = C_g * spatial;

    const int group_offset = (n * C + g * C_g) * spatial;
    const float* x_ptr = x + group_offset;
    float*       y_ptr = y + group_offset;

    // ---- Pass 1: compute mean and variance ----
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < M; i += THREADS) {
        float val = x_ptr[i];
        local_sum += val;
        local_sum_sq += val * val;
    }

    local_sum = warp_reduce_sum(local_sum);
    local_sum_sq = warp_reduce_sum(local_sum_sq);

    constexpr int NUM_WARPS = THREADS / 32;
    __shared__ float s_sum[NUM_WARPS];
    __shared__ float s_sum_sq[NUM_WARPS];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        s_sum[warp_id] = local_sum;
        s_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum_val = (lane_id < NUM_WARPS) ? s_sum[lane_id] : 0.0f;
        float sq_val  = (lane_id < NUM_WARPS) ? s_sum_sq[lane_id] : 0.0f;
        sum_val = warp_reduce_sum(sum_val);
        sq_val  = warp_reduce_sum(sq_val);

        if (lane_id == 0) {
            float mean = sum_val / M;
            float var = sq_val / M - mean * mean;
            float rstd = rsqrtf(var + eps);
            s_sum[0] = mean;
            s_sum_sq[0] = rstd;
            mean_out[ng] = mean;
            rstd_out[ng] = rstd;
        }
    }
    __syncthreads();

    float mean = s_sum[0];
    float rstd = s_sum_sq[0];

    // ---- Pass 2: normalize + affine (no GELU) ----
    for (int i = threadIdx.x; i < M; i += THREADS) {
        float val = x_ptr[i];
        float x_hat = (val - mean) * rstd;
        int c = g * C_g + i / spatial;
        y_ptr[i] = gamma[c] * x_hat + beta[c];
    }
}

template <int THREADS>
static __global__ void group_norm_bwd_kernel(
    const float* __restrict__ d_output,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ mean_in,
    const float* __restrict__ rstd_in,
    float*       __restrict__ d_input,
    float*       __restrict__ d_gamma,
    float*       __restrict__ d_beta,
    int C, int spatial, int G)
{
    const int ng = blockIdx.x;
    const int n  = ng / G;
    const int g  = ng % G;
    const int C_g = C / G;
    const int M = C_g * spatial;

    const int group_offset = (n * C + g * C_g) * spatial;
    const float* x_ptr  = x + group_offset;
    const float* dy_ptr = d_output + group_offset;
    float*       dx_ptr = d_input + group_offset;

    float mean = mean_in[ng];
    float rstd = rstd_in[ng];

    // ---- Pass 1: compute ds, db and per-channel d_gamma/d_beta ----
    float local_ds = 0.0f;
    float local_db = 0.0f;

    float local_dg[MAX_C_PER_GROUP];
    float local_dbt[MAX_C_PER_GROUP];
    for (int c = 0; c < C_g; c++) {
        local_dg[c] = 0.0f;
        local_dbt[c] = 0.0f;
    }

    for (int i = threadIdx.x; i < M; i += THREADS) {
        float val = x_ptr[i];
        float x_hat = (val - mean) * rstd;
        int c_local = i / spatial;
        int c_global = g * C_g + c_local;

        float dy = dy_ptr[i];

        // No GELU — dy is the direct upstream gradient
        local_dg[c_local] += dy * x_hat;
        local_dbt[c_local] += dy;

        float dz_eff = dy * gamma[c_global];
        local_ds += dz_eff * x_hat;
        local_db += dz_eff;
    }

    // ---- Block reduction for ds, db ----
    local_ds = warp_reduce_sum(local_ds);
    local_db = warp_reduce_sum(local_db);

    constexpr int NUM_WARPS = THREADS / 32;
    __shared__ float s_reduce[NUM_WARPS * 2];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        s_reduce[warp_id] = local_ds;
        s_reduce[NUM_WARPS + warp_id] = local_db;
    }
    __syncthreads();

    if (warp_id == 0) {
        float ds = (lane_id < NUM_WARPS) ? s_reduce[lane_id] : 0.0f;
        float db = (lane_id < NUM_WARPS) ? s_reduce[NUM_WARPS + lane_id] : 0.0f;
        ds = warp_reduce_sum(ds);
        db = warp_reduce_sum(db);
        if (lane_id == 0) {
            s_reduce[0] = ds;
            s_reduce[1] = db;
        }
    }
    __syncthreads();

    float ds = s_reduce[0];
    float db = s_reduce[1];
    float inv_M = 1.0f / M;

    // ---- Block reduction for per-channel d_gamma/d_beta ----
    for (int c = 0; c < C_g; c++) {
        float warp_dg = warp_reduce_sum(local_dg[c]);
        float warp_dbt = warp_reduce_sum(local_dbt[c]);

        if (lane_id == 0) {
            s_reduce[warp_id] = warp_dg;
            s_reduce[NUM_WARPS + warp_id] = warp_dbt;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float sum_dg = 0.0f, sum_dbt = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                sum_dg += s_reduce[w];
                sum_dbt += s_reduce[NUM_WARPS + w];
            }
            atomicAdd(&d_gamma[g * C_g + c], sum_dg);
            atomicAdd(&d_beta[g * C_g + c], sum_dbt);
        }
        __syncthreads();
    }

    // ---- Pass 2: compute d_input ----
    for (int i = threadIdx.x; i < M; i += THREADS) {
        float val = x_ptr[i];
        float x_hat = (val - mean) * rstd;
        int c_global = g * C_g + i / spatial;

        float dy = dy_ptr[i];
        float dz_eff = dy * gamma[c_global];

        dx_ptr[i] = rstd * (dz_eff - inv_M * (db + x_hat * ds));
    }
}

void group_norm_forward(
    const float* x,
    const float* gamma,
    const float* beta,
    float*       y,
    float*       mean,
    float*       rstd,
    int N, int C, int spatial,
    int G,
    float eps,
    cudaStream_t stream)
{
    int blocks = N * G;
    if (blocks <= 0) return;

    int C_g = C / G;
    int M = C_g * spatial;

    if (M >= 4096) {
        group_norm_fwd_kernel<512><<<blocks, 512, 0, stream>>>(
            x, gamma, beta, y, mean, rstd, C, spatial, G, eps);
    } else {
        group_norm_fwd_kernel<256><<<blocks, 256, 0, stream>>>(
            x, gamma, beta, y, mean, rstd, C, spatial, G, eps);
    }
}

void group_norm_backward(
    const float* d_output,
    const float* x,
    const float* gamma,
    const float* mean,
    const float* rstd,
    float*       d_input,
    float*       d_gamma,
    float*       d_beta,
    int N, int C, int spatial,
    int G,
    cudaStream_t stream)
{
    int blocks = N * G;
    if (blocks <= 0) return;

    int C_g = C / G;
    int M = C_g * spatial;

    if (M >= 4096) {
        group_norm_bwd_kernel<512><<<blocks, 512, 0, stream>>>(
            d_output, x, gamma, mean, rstd,
            d_input, d_gamma, d_beta, C, spatial, G);
    } else {
        group_norm_bwd_kernel<256><<<blocks, 256, 0, stream>>>(
            d_output, x, gamma, mean, rstd,
            d_input, d_gamma, d_beta, C, spatial, G);
    }
}

}} // namespace forge::nn
