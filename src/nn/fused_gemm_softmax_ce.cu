// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/fused_gemm_softmax_ce.cuh>

namespace forge { namespace nn {

// ============================================================
//  Fused GEMM + online-softmax + cross-entropy kernel
// ============================================================
//
// One block per batch sample.  Each block:
//   1. Loads the input vector X[:, b] into shared memory (D floats).
//   2. Phase 1 — Online softmax:  each thread iterates over its
//      share of V vocabulary tokens, computing the logit via a
//      D-length dot product against s_x, and folding the result
//      into per-thread (running_max, running_sum_exp) state.
//      A block-wide reduction then merges these into global max
//      and sum_exp.
//   3. Phase 2 — Prob write + argmax:  each thread re-iterates
//      over the same V tokens, recomputes the logit (W rows are
//      L2-cached from phase 1), and writes the normalised
//      probability to global memory.
//
// Why recompute instead of storing logits?
//   V = 5000 at 4 B/float × 512 samples = 10 MB of logits.
//   Re-reading W rows from L2 (5 MB, shared across all blocks)
//   is far cheaper than a 20 MB DRAM round-trip.

template <int THREADS>
__global__ void fused_gemm_softmax_ce_kernel(
    const float* __restrict__ W,       // [V, D] row-major
    const float* __restrict__ X,       // [D, B] row-major
    const float* __restrict__ bias,    // [V] or nullptr
    float* __restrict__       probs,   // [V, B] row-major
    const int* __restrict__   targets, // [B]
    float* __restrict__       losses,  // [B]
    int* __restrict__         correct, // [B]
    int V, int D, int B)
{
    int b = blockIdx.x;
    if (b >= B) return;

    int target_b = targets[b];

    // ---- Shared memory layout ----
    // Static: reduction arrays + scalars
    // Dynamic: s_x[D] (input vector for this sample)
    __shared__ float s_rmax[THREADS];     // online-softmax max reduction
    __shared__ float s_rsum[THREADS];     // online-softmax sum reduction
    __shared__ float s_target_logit;      // logit of the target token
    __shared__ float s_block_max;         // final block-wide max
    __shared__ float s_block_inv_sum;     // 1 / block-wide sum_exp

    extern __shared__ float s_x[];        // [D]

    // Load X[:, b] into shared memory
    for (int d = threadIdx.x; d < D; d += THREADS)
        s_x[d] = X[d * B + b];
    if (threadIdx.x == 0)
        s_target_logit = 0.0f;
    __syncthreads();

    // ================================================================
    // Phase 1: Online softmax — compute block-wide max and sum_exp
    // ================================================================
    float tmax = -1e30f;   // per-thread running max
    float tsum = 0.0f;     // per-thread running sum_exp

    for (int v = threadIdx.x; v < V; v += THREADS) {
        float logit = bias ? bias[v] : 0.0f;
        const float* w_row = W + v * D;
        #pragma unroll 4
        for (int d = 0; d < D; d++)
            logit += w_row[d] * s_x[d];

        // Online softmax update
        if (logit > tmax) {
            tsum = tsum * expf(tmax - logit) + 1.0f;
            tmax = logit;
        } else {
            tsum += expf(logit - tmax);
        }

        // Record target logit (exactly one thread hits this)
        if (v == target_b)
            s_target_logit = logit;
    }

    // Block-wide pair reduction of (max, sum_exp)
    s_rmax[threadIdx.x] = tmax;
    s_rsum[threadIdx.x] = tsum;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float am = s_rmax[threadIdx.x], as = s_rsum[threadIdx.x];
            float bm = s_rmax[threadIdx.x + stride], bs = s_rsum[threadIdx.x + stride];
            if (am >= bm) {
                s_rsum[threadIdx.x] = as + bs * expf(bm - am);
            } else {
                s_rmax[threadIdx.x] = bm;
                s_rsum[threadIdx.x] = as * expf(am - bm) + bs;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        s_block_max     = s_rmax[0];
        s_block_inv_sum = 1.0f / s_rsum[0];
        losses[b] = logf(s_rsum[0]) + s_rmax[0] - s_target_logit;
    }
    __syncthreads();

    // ================================================================
    // Phase 2: Write probabilities + find argmax (for correct flag)
    // ================================================================
    float bmax    = s_block_max;
    float inv_sum = s_block_inv_sum;

    float t_best_val = -1e30f;
    int   t_best_idx = 0;

    for (int v = threadIdx.x; v < V; v += THREADS) {
        float logit = bias ? bias[v] : 0.0f;
        const float* w_row = W + v * D;
        #pragma unroll 4
        for (int d = 0; d < D; d++)
            logit += w_row[d] * s_x[d];

        probs[v * B + b] = expf(logit - bmax) * inv_sum;

        if (logit > t_best_val) {
            t_best_val = logit;
            t_best_idx = v;
        }
    }

    // Block-wide argmax reduction (reuse s_rmax / s_rsum as value/index)
    s_rmax[threadIdx.x] = t_best_val;
    ((int*)s_rsum)[threadIdx.x] = t_best_idx;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_rmax[threadIdx.x + stride] > s_rmax[threadIdx.x]) {
                s_rmax[threadIdx.x] = s_rmax[threadIdx.x + stride];
                ((int*)s_rsum)[threadIdx.x] = ((int*)s_rsum)[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        correct[b] = (((int*)s_rsum)[0] == target_b) ? 1 : 0;
}

// ============================================================
//  Public API
// ============================================================

void fused_gemm_softmax_ce_forward(
    const float* W, const float* X, const float* bias,
    float* probs, const int* targets, float* losses, int* correct,
    int V, int D, int B,
    cudaStream_t stream)
{
    if (V <= 0 || D <= 0 || B <= 0) return;

    constexpr int THREADS = 256;
    int smem_bytes = D * sizeof(float);

    fused_gemm_softmax_ce_kernel<THREADS>
        <<<B, THREADS, smem_bytes, stream>>>(
            W, X, bias, probs, targets, losses, correct, V, D, B);
}

}} // namespace forge::nn
