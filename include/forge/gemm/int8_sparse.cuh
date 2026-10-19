// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Sparse INT8 Tensor Core GEMM via PTX mma.sp instruction.
// 2:4 structured sparsity for 2x throughput over dense INT8.
// Up to ~570 dense-equivalent TOPS on an RTX 4090 (GEMM + dequant epilogue).
//
// W_comp: (N_out, K/2) compressed weight (2:4 structured sparsity)
// x_int8: (M, K) quantized activation
// w_meta: (N_out, K/32) packed uint32 metadata
//
// Requires: K % 64 == 0, N_out % 16 == 0, sm_80+.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Version 2: cost-model dispatch, sparse_int8_gemm_plan, forced tile configs.
// Version 3: fused quantize + GEMM path for sparse_int8_linear (path argument,
//            sparse_int8_linear_fused).
#define FORGE_SPARSE_GEMM_API_VERSION 3

namespace forge { namespace gemm {

/// CTA tile configurations (weight rows x tokens per tile).
constexpr int kSparseTileAuto    = -1;
constexpr int kSparseTile256x128 = 0;
constexpr int kSparseTile128x128 = 1;
constexpr int kSparseTile128x64  = 2;
constexpr int kSparseNumTiles    = 3;

/// The configuration sparse_int8_gemm runs for a problem.
struct SparseGemmPlan {
    int tile;      // kSparseTile256x128 / kSparseTile128x128 / kSparseTile128x64
    int split_k;   // >= 1
};

/// Returns the (tile, split-K) that sparse_int8_gemm chooses, from a wave-aware cost model
/// (tuned on an RTX 4090; the SM count is queried from the current device).
///   split_k <= 0: choose split-K too (candidates 1, 2, 3, 4, 6; split-K only when
///                 have_workspace). split_k >= 1: use that split (1 if !have_workspace).
///   tile < 0: choose the tile; 0..2: force that tile config.
SparseGemmPlan sparse_int8_gemm_plan(
    int  M,
    int  K,
    int  N_out,
    int  split_k        = 0,
    bool have_workspace = true,
    int  tile           = kSparseTileAuto);

/// Sparse INT8 TC GEMM with fused dequant + bias.
/// split_k <= 0 selects split-K automatically (see sparse_int8_gemm_plan); split-K
/// needs splitk_workspace, an (M, N_out) fp32 device buffer that is zeroed and
/// overwritten by the call (its prior contents do not matter). Without a workspace
/// the GEMM runs unsplit. tile forces a tile config (testing / tuning).
void sparse_int8_gemm(
    const int8_t*   W_comp,     // (N_out, K/2) device
    const int8_t*   x_int8,     // (M, K) device
    const uint32_t* w_meta,     // (N_out, K/32) device
    const float*    act_scale,  // (M,) device
    const __half*   w_scale,    // (N_out,) device
    const __half*   bias,       // (N_out,) device or nullptr
    __half*         output,     // (M, N_out) device
    int             M,
    int             K,
    int             N_out,
    float*          splitk_workspace = nullptr,
    int             split_k = 1,
    cudaStream_t    stream  = nullptr,
    int             tile    = kSparseTileAuto);

/// sparse_int8_linear paths.
constexpr int kSparseLinearAuto    = -1;  // fused when it saves DRAM traffic (see below)
constexpr int kSparseLinearTwoPass = 0;   // quantize kernel -> int8 scratch -> sparse GEMM
constexpr int kSparseLinearFused   = 1;   // one kernel: quantize into smem + sparse GEMM

/// Whether sparse_int8_linear's automatic path fuses the activation quantize into the GEMM
/// for this shape: K <= 1280 (the 64-token int8 block stays resident in shared memory),
/// the activation and its int8 copy do not fit L2 (3*M*K > 0.9 * L2), and the DRAM traffic
/// saved outweighs the smaller tile (cost model). Typical: narrow N_out at large M
/// (1280 -> 320 at M = 147456: 1.51 ms -> 0.70 ms on an RTX 4090).
bool sparse_int8_linear_fused(int M, int K, int N_out);

/// Convenience: quant + sparse GEMM + dequant.
/// path = kSparseLinearAuto picks the fused kernel when sparse_int8_linear_fused(M, K, N_out)
/// and split_k <= 1; it then does not touch the scratch buffers or the workspace.
/// kSparseLinearFused forces it (needs K <= 1280 and a 16-byte aligned x, else two-pass).
/// Outputs are bit-identical across paths.
void sparse_int8_linear(
    const __half*   x,           // (M, K) fp16 device
    const int8_t*   W_comp,      // (N_out, K/2) int8 device
    const uint32_t* w_meta,      // (N_out, K/32) device
    const __half*   w_scale,     // (N_out,) fp16 device
    const __half*   bias,        // (N_out,) fp16 device or nullptr
    __half*         output,      // (M, N_out) fp16 device
    int             M,
    int             K,
    int             N_out,
    int8_t*         scratch_x_int8,
    float*          scratch_x_scale,
    float*          splitk_workspace = nullptr,
    int             split_k = 1,
    cudaStream_t    stream  = nullptr,
    int             path    = kSparseLinearAuto);

/// Split-K reduce: sum fp32 workspace partials -> fp16 output + bias.
void splitk_reduce(
    const float*  workspace,    // (M, N_out) fp32 device
    const __half* bias,         // (N_out,) device or nullptr
    __half*       output,       // (M, N_out) fp16 device
    int           M,
    int           N_out,
    cudaStream_t  stream = nullptr);

}} // namespace forge::gemm
