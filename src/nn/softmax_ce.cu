// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/softmax_ce.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace nn {

// ============================================================
//  Online-softmax forward (2-pass instead of 3)
// ============================================================
//
// One block per sample. Two passes over logits:
//
//   Phase 1 (online max + sum_exp):
//     Read each logit once. Maintain per-thread running (max, sum_exp)
//     using the online softmax algorithm, then merge across threads
//     with a tree reduction of (max, sum) pairs.
//
//   Phase 2 (normalize + write + argmax):
//     Read logits again, compute prob = exp(logit - max) / sum,
//     write to probs, track argmax for accuracy.
//
// Traffic: 2 reads + 1 write = 3 V*B passes (was 5 in the 3-phase version).

__global__ void softmax_ce_forward_kernel(
    float* __restrict__       logits,
    float* __restrict__       probs,
    const int* __restrict__   targets,
    float* __restrict__       losses,
    int* __restrict__         correct,
    int                       V,
    int                       B)
{
    extern __shared__ char smem[];
    float* s_max = (float*)smem;              // [nthreads]
    float* s_sum = s_max + blockDim.x;        // [nthreads]

    int b   = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    if (b >= B) return;

    int target_b = targets[b];

    // ---- Phase 1: Online softmax (single pass over logits) ----
    // Each thread maintains running (max, sum_exp) for its strided slice of V.
    float local_max = -1e30f;
    float local_sum = 0.0f;

    __shared__ float s_target_logit;
    if (tid == 0) s_target_logit = 0.0f;
    __syncthreads();

    for (int v = tid; v < V; v += nthreads) {
        float l = logits[v * B + b];
        if (l > local_max) {
            local_sum = local_sum * expf(local_max - l) + 1.0f;
            local_max = l;
        } else {
            local_sum += expf(l - local_max);
        }
        if (v == target_b) s_target_logit = l;
    }

    // Block-wide merge of (max, sum) pairs
    s_max[tid] = local_max;
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float am = s_max[tid], as = s_sum[tid];
            float bm = s_max[tid + stride], bs = s_sum[tid + stride];
            if (am >= bm) {
                s_sum[tid] = as + bs * expf(bm - am);
            } else {
                s_max[tid] = bm;
                s_sum[tid] = as * expf(am - bm) + bs;
            }
        }
        __syncthreads();
    }

    float final_max = s_max[0];
    float final_sum = s_sum[0];
    float inv_sum   = 1.0f / final_sum;

    // CE loss = log(sum_exp) + max - target_logit  (numerically stable)
    if (tid == 0)
        losses[b] = logf(final_sum) + final_max - s_target_logit;
    __syncthreads();

    // ---- Phase 2: Write probs + argmax ----
    float best_p = -1.0f;
    int   best_v = 0;

    for (int v = tid; v < V; v += nthreads) {
        float p = expf(logits[v * B + b] - final_max) * inv_sum;
        probs[v * B + b] = p;
        if (p > best_p) { best_p = p; best_v = v; }
    }

    // Argmax reduction (reuse s_max for values, s_sum for indices)
    s_max[tid] = best_p;
    ((int*)s_sum)[tid] = best_v;
    __syncthreads();

    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_max[tid + stride] > s_max[tid]) {
            s_max[tid] = s_max[tid + stride];
            ((int*)s_sum)[tid] = ((int*)s_sum)[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        correct[b] = (((int*)s_sum)[0] == target_b) ? 1 : 0;
}

void softmax_ce_forward(
    float*       logits,
    float*       probs,
    const int*   targets,
    float*       losses,
    int*         correct,
    int          V,
    int          B,
    int          threads,
    cudaStream_t stream)
{
    if (B <= 0 || V <= 0) return;
    size_t smem_bytes = threads * 2 * sizeof(float);
    softmax_ce_forward_kernel<<<B, threads, smem_bytes, stream>>>(
        logits, probs, targets, losses, correct, V, B);
}

__global__ void softmax_ce_backward_kernel(
    const float* __restrict__ probs,
    const int* __restrict__   targets,
    float* __restrict__       d_logits,
    int                       V,
    int                       B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int v = idx / B;
    int b = idx % B;
    if (v >= V) return;

    d_logits[v * B + b] = probs[v * B + b] - ((v == targets[b]) ? 1.0f : 0.0f);
}

void softmax_ce_backward(
    const float* probs,
    const int*   targets,
    float*       d_logits,
    int          V,
    int          B,
    cudaStream_t stream)
{
    if (V <= 0 || B <= 0) return;
    int total = V * B;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(total, block);
    softmax_ce_backward_kernel<<<grid, block, 0, stream>>>(
        probs, targets, d_logits, V, B);
}

// ============================================================
//  Fused backward + bias gradient
// ============================================================
//
// One block per V-row.  Threads cooperatively process B samples:
//   1. Read probs[v, b], compute d_logit = prob - one_hot(target)
//   2. Write d_logits[v, b]  (coalesced: consecutive b in stride-1)
//   3. Tree-reduce per-thread partial sums → d_bias[v]
//
// Replaces the separate backward (elementwise) + bias_grad (column-sum)
// kernels, eliminating one full V*B read of d_logits.

__global__ void softmax_ce_backward_bias_kernel(
    const float* __restrict__ probs,
    const int* __restrict__   targets,
    float* __restrict__       d_logits,
    float* __restrict__       d_bias,
    int                       V,
    int                       B)
{
    extern __shared__ float s_sum[];

    int v   = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    if (v >= V) return;

    float local_sum = 0.0f;
    const float* row_probs   = probs    + v * B;
    float*       row_dlogits = d_logits + v * B;

    for (int b = tid; b < B; b += nthreads) {
        float dl = row_probs[b] - ((v == targets[b]) ? 1.0f : 0.0f);
        row_dlogits[b] = dl;
        local_sum += dl;
    }

    // Tree reduction for bias gradient
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        d_bias[v] += s_sum[0];
}

void softmax_ce_backward_bias(
    const float* probs,
    const int*   targets,
    float*       d_logits,
    float*       d_bias,
    int          V,
    int          B,
    int          threads,
    cudaStream_t stream)
{
    if (V <= 0 || B <= 0) return;
    size_t smem_bytes = threads * sizeof(float);
    softmax_ce_backward_bias_kernel<<<V, threads, smem_bytes, stream>>>(
        probs, targets, d_logits, d_bias, V, B);
}

}} // namespace forge::nn
