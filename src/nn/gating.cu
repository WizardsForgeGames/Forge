// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/gating.cuh>
#include <forge/primitives/activation.cuh>
#include <forge/primitives/reduce.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace nn {

// ============================================================
//  Forward: 1 thread per element (unchanged)
// ============================================================

static __global__ void sigmoid_gate_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__       gated,
    int                       N,
    int                       B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * B;
    if (idx >= total) return;

    int unit = idx / B;
    float xi = x[idx];
    float gate = primitives::sigmoid(w[unit] * xi + b[unit]);
    gated[idx] = gate * xi;
}

void sigmoid_gate_forward(
    const float* x,
    const float* w,
    const float* b,
    float*       gated,
    int          N,
    int          B,
    cudaStream_t stream)
{
    int total = N * B;
    if (total <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(total, block);
    sigmoid_gate_forward_kernel<<<grid, block, 0, stream>>>(x, w, b, gated, N, B);
}

// ============================================================
//  Backward: 1 block per unit, cooperative reduction over B
// ============================================================
// Previous kernel: 1 thread per (unit, batch) element, all B threads
// atomicAdd to same dw[unit] → O(B) contention per unit.
//
// New kernel: 1 block per unit. Each thread loops over batch samples
// with stride blockDim.x, accumulates local_dw/local_db in registers,
// then block_reduce<SumOp> → single write per unit. Contention: O(1).

static __global__ void sigmoid_gate_backward_kernel(
    const float* __restrict__ d_gated,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__       dx,
    float* __restrict__       dw,
    float* __restrict__       db,
    int                       N,
    int                       B)
{
    extern __shared__ float smem[];

    int unit = blockIdx.x;
    if (unit >= N) return;

    int tid = threadIdx.x;
    float wi = w[unit];
    float bi = b[unit];

    // Each thread accumulates partial dw, db over its strided batch elements
    float local_dw = 0.0f;
    float local_db = 0.0f;

    int base = unit * B;
    for (int s = tid; s < B; s += blockDim.x) {
        int idx = base + s;
        float xi   = x[idx];
        float gate = primitives::sigmoid(wi * xi + bi);
        float gd   = primitives::sigmoid_grad(gate);
        float dg   = d_gated[idx];

        // dx is per-element, no reduction needed
        dx[idx] = dg * (gate + xi * gd * wi);

        local_dw += dg * xi * xi * gd;
        local_db += dg * xi * gd;
    }

    // Block-level reduction of dw
    float reduced_dw = primitives::block_reduce<primitives::SumOp>(local_dw, smem);
    if (tid == 0) dw[unit] = reduced_dw;

    // Block-level reduction of db (reuse smem after syncthreads in block_reduce)
    float reduced_db = primitives::block_reduce<primitives::SumOp>(local_db, smem);
    if (tid == 0) db[unit] = reduced_db;
}

void sigmoid_gate_backward(
    const float* d_gated,
    const float* x,
    const float* w,
    const float* b,
    float*       dx,
    float*       dw,
    float*       db,
    int          N,
    int          B,
    cudaStream_t stream)
{
    if (N <= 0 || B <= 0) return;

    // One block per unit; block size = min(256, round up B to warp)
    int block = min(256, ((B + 31) / 32) * 32);
    block = max(block, 32);  // at least one warp
    size_t smem = (block / 32) * sizeof(float);

    sigmoid_gate_backward_kernel<<<N, block, smem, stream>>>(
        d_gated, x, w, b, dx, dw, db, N, B);
}

}} // namespace forge::nn
