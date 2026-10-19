// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Fused MLP forward and backward kernels.
//
// Architecture:
//   One block processes a tile of TILE_B batch samples through ALL layers.
//   For TILE_B > 2: weights loaded layer-by-layer into shared memory.
//   For TILE_B <= 2: weights read directly from L2-cached global memory,
//   eliminating 66KB shared memory overhead per block.
//   Activations flow through shared memory — no global memory between layers.
//
// TILE_B is a template parameter, auto-selected per call based on batch size
// to maximize SM occupancy. Smaller TILE_B = more blocks = more parallelism.
// All threads cooperatively distribute (sample, dim) work pairs — no assumption
// that TILE_B equals warp size.
//
// For H<=128: weight tile fits in shared memory (66KB at H=128).
//   TILE_B=4: +4KB per tile buffer → ~73KB total backward, 1-3 blocks/SM.
//   TILE_B=32: +33KB per tile buffer → ~99KB total backward, 1 block/SM.
//   TILE_B=1 (L2 mode): ~1KB total → many blocks/SM, limited by registers.
//
// For H>128 (up to 256): weight tile exceeds shared memory (263KB at H=256).
//   L2 mode forced (TILE_B<=2). Activations still flow through shared memory.
#include <forge/nn/fused_mlp.cuh>
#include <forge/nn/rng.cuh>
#include <forge/primitives/activation.cuh>
#include "cuda_utils.cuh"
#include <mma.h>
#include <cmath>
#include <cstdio>

namespace forge { namespace nn {

// ============================================================
//  Weight packing helpers
// ============================================================

int mlp_weight_count(const MlpDesc& desc) {
    int count = 0;
    count += desc.hidden * desc.n_inputs + desc.hidden;
    for (int l = 1; l < desc.n_layers; l++)
        count += desc.hidden * desc.hidden + desc.hidden;
    count += desc.n_outputs * desc.hidden + desc.n_outputs;
    if (desc.normalize)
        count += desc.n_inputs * 2;
    return count;
}

int mlp_acts_save_count(const MlpDesc& desc, int B) {
    return B * (desc.hidden * desc.n_layers + desc.n_outputs);
}

// ============================================================
//  Device helpers: apply activation
// ============================================================

__device__ __forceinline__ float apply_activation(
    float x, Activation act, float param)
{
    switch (act) {
        case Activation::CELU:    return primitives::celu(x, param);
        case Activation::GELU:    return primitives::gelu(x);
        case Activation::ReLU:    return (x > 0.0f) ? x : 0.0f;
        case Activation::Sigmoid: return primitives::sigmoid(x);
        case Activation::SiLU:    return primitives::silu(x);
        case Activation::Tanh:    return tanhf(x);
        default:                  return x;
    }
}

__device__ __forceinline__ float apply_activation_grad(
    float z, float dy, Activation act, float param)
{
    switch (act) {
        case Activation::CELU:    return primitives::celu_grad(z, dy, param);
        case Activation::GELU:    return primitives::gelu_grad(z, dy);
        case Activation::ReLU:    return (z > 0.0f) ? dy : 0.0f;
        case Activation::Sigmoid: {
            float s = primitives::sigmoid(z);
            return dy * s * (1.0f - s);
        }
        case Activation::SiLU:    return primitives::silu_grad(z, dy);
        case Activation::Tanh: {
            float t = tanhf(z);
            return dy * (1.0f - t * t);
        }
        default: return dy;
    }
}

// ============================================================
//  Tuning constants
// ============================================================

constexpr int MAX_H     = 256;  // maximum hidden width supported
                                // H <= 128: shared memory path (fastest)
                                // H > 128:  L2 cache path (TILE_B <= 2)
constexpr int FWD_BLOCK = 256;  // forward kernel block size (8 warps)
constexpr int BWD_BLOCK = 256;  // backward kernel block size (8 warps)

// Maximum register staging slots per thread.
// For forward/backward, each thread processes ceil(TILE_B * dim / BLOCK) elements.
// Maximum is at (TILE_B=32, dim=128, BLOCK=256) = 16.
// For H>128, TILE_B is forced to <=2, so ceil(2*256/256)=2 — well within limit.
constexpr int MAX_REGS = 16;

// Shared-memory row stride (floats) of the TF32 backward kernel: max(hidden,
// n_inputs) rounded up to the 16-wide wmma N tile used for gradient
// propagation. Shared by the kernel and the host smem-size computation.
__host__ __device__ __forceinline__ int tc_bwd_k_max(int hidden, int n_inputs) {
    int k = (n_inputs > hidden) ? n_inputs : hidden;
    return (k + 15) & ~15;
}

// Read-only weight access: __ldg() for L2 mode, direct for shared memory mode.
// __ldg() routes through the read-only texture cache with better eviction behavior
// for data that is read by many threads but never written during the kernel.
template<bool USE_LDG>
__device__ __forceinline__ float w_load(const float* ptr) {
    if constexpr (USE_LDG) return __ldg(ptr);
    else return *ptr;
}

// ============================================================
//  Forward kernel — templated on TILE_B
// ============================================================
//
// Grid:  (divup(B, TILE_B), 1)
// Block: (FWD_BLOCK, 1)
//
// All threads cooperatively distribute (sample, dim) work pairs.
// Register staging avoids WAR hazards on shared memory between reads and writes.
//
// For TILE_B <= 2: weights read directly from L2 cache (no shared memory).
// This eliminates the dominant 66KB shared memory allocation, allowing many
// more blocks per SM for better occupancy with small batch tiles.

template<int TILE_B>
__global__ void fused_mlp_forward_kernel(
    int            n_inputs,
    int            n_outputs,
    int            hidden,
    int            n_layers,
    Activation     act,
    float          act_param,
    float          output_scale,
    bool           do_normalize,
    int            norm_offset,
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__       output,
    float* __restrict__       acts_save,
    int                       B)
{
    constexpr bool USE_SMEM_W = (TILE_B > 2);

    extern __shared__ char smem_raw[];
    int w_cols   = (hidden > n_inputs) ? hidden : n_inputs;
    int w_stride = w_cols + 1;
    int a_stride = w_cols + 1;

    float* s_W;
    float* s_bias;
    float* s_act;

    if constexpr (USE_SMEM_W) {
        s_W    = (float*)smem_raw;
        s_bias = s_W + hidden * w_stride;
        s_act  = s_bias + hidden;
    } else {
        s_W    = nullptr;
        s_bias = nullptr;
        s_act  = (float*)smem_raw;
    }

    int tid    = threadIdx.x;
    int b_base = blockIdx.x * TILE_B;
    int n_active = (b_base + TILE_B <= B) ? TILE_B : (B - b_base);

    // ---- Pointer into packed weights ----
    int w_offset = 0;

    // ---- Normalization bounds ----
    const float* lb = nullptr;
    const float* ub = nullptr;
    if (do_normalize) {
        lb = weights + norm_offset;
        ub = lb + n_inputs;
    }

    // ---- Cooperatively load input into s_act ----
    {
        int elems = TILE_B * n_inputs;
        for (int i = tid; i < elems; i += blockDim.x) {
            int s = i / n_inputs;
            int d = i % n_inputs;
            int b_i = b_base + s;
            float val = (b_i < B) ? input[b_i * n_inputs + d] : 0.0f;
            if (do_normalize && b_i < B)
                val = 2.0f * (val - lb[d]) / (ub[d] - lb[d]) - 1.0f;
            s_act[s * a_stride + d] = val;
        }
    }
    __syncthreads();

    int acts_stride = hidden * n_layers + n_outputs;

    // ---- Process each hidden layer ----
    for (int l = 0; l < n_layers; l++) {
        int in_dim = (l == 0) ? n_inputs : hidden;

        const float* W_ptr = weights + w_offset;
        const float* b_ptr = W_ptr + hidden * in_dim;

        // Weight read pointers: shared memory (padded) or L2 global (compact)
        const float* W_rd;
        const float* b_rd;
        int W_rd_stride;

        if constexpr (USE_SMEM_W) {
            // Cooperatively load W[hidden, in_dim] and bias[hidden]
            int w_elements = hidden * in_dim;
            for (int i = tid; i < w_elements; i += blockDim.x) {
                int row = i / in_dim;
                int col = i % in_dim;
                s_W[row * w_stride + col] = W_ptr[i];
            }
            for (int i = tid; i < hidden; i += blockDim.x)
                s_bias[i] = b_ptr[i];
            __syncthreads();
            W_rd = s_W;
            b_rd = s_bias;
            W_rd_stride = w_stride;
        } else {
            W_rd = W_ptr;
            b_rd = b_ptr;
            W_rd_stride = in_dim;
        }

        int save_offset = l * hidden;

        // Distribute (sample, hidden_unit) pairs across all threads.
        // Stage results in registers to avoid WAR hazard on s_act.
        int total = n_active * hidden;
        float regs[MAX_REGS];
        int reg_count = 0;
        for (int idx = tid; idx < total; idx += blockDim.x) {
            int s = idx / hidden;
            int j = idx % hidden;
            float sum = w_load<!USE_SMEM_W>(&b_rd[j]);
            for (int k = 0; k < in_dim; k++)
                sum += w_load<!USE_SMEM_W>(&W_rd[j * W_rd_stride + k]) * s_act[s * a_stride + k];
            regs[reg_count++] = sum;
        }
        __syncthreads();  // all reads of s_act complete before writes

        // Write back: save pre-activation, apply activation, update s_act
        reg_count = 0;
        for (int idx = tid; idx < total; idx += blockDim.x) {
            int s = idx / hidden;
            int j = idx % hidden;
            int b_i = b_base + s;
            if (acts_save && b_i < B)
                acts_save[b_i * acts_stride + save_offset + j] = regs[reg_count];
            s_act[s * a_stride + j] = apply_activation(regs[reg_count], act, act_param);
            reg_count++;
        }
        __syncthreads();

        w_offset += hidden * in_dim + hidden;
    }

    // ---- Output layer (no activation) ----
    {
        const float* W_ptr = weights + w_offset;
        const float* b_ptr = W_ptr + n_outputs * hidden;

        const float* W_rd;
        const float* b_rd;
        int W_rd_stride;

        if constexpr (USE_SMEM_W) {
            int w_elements = n_outputs * hidden;
            for (int i = tid; i < w_elements; i += blockDim.x) {
                int row = i / hidden;
                int col = i % hidden;
                s_W[row * w_stride + col] = W_ptr[i];
            }
            for (int i = tid; i < n_outputs; i += blockDim.x)
                s_bias[i] = b_ptr[i];
            __syncthreads();
            W_rd = s_W;
            b_rd = s_bias;
            W_rd_stride = w_stride;
        } else {
            W_rd = W_ptr;
            b_rd = b_ptr;
            W_rd_stride = hidden;
        }

        int save_offset = n_layers * hidden;
        int total_out = n_active * n_outputs;
        for (int idx = tid; idx < total_out; idx += blockDim.x) {
            int s = idx / n_outputs;
            int j = idx % n_outputs;
            int b_i = b_base + s;
            float sum = w_load<!USE_SMEM_W>(&b_rd[j]);
            for (int k = 0; k < hidden; k++)
                sum += w_load<!USE_SMEM_W>(&W_rd[j * W_rd_stride + k]) * s_act[s * a_stride + k];
            if (acts_save)
                acts_save[b_i * acts_stride + save_offset + j] = sum;
            output[b_i * n_outputs + j] = sum * output_scale;
        }
    }
}

// ============================================================
//  TC Forward kernel — TF32 Tensor Core path
// ============================================================
//
// Uses wmma 16x16x8 TF32 fragments for hidden-layer matmuls.
// Double-buffered activations (no register staging needed).
// Weights always in shared memory (only supports H <= 128).
// TILE_B must be a multiple of 16.
//
// Requires sm_80+ (Ampere).  Guarded with __CUDA_ARCH__ >= 800.
//
// Grid:  (divup(B, TILE_B), 1)
// Block: (FWD_BLOCK, 1)  — 256 threads = 8 warps

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// Only compile TC kernels for sm_80+
#define FORGE_TC_AVAILABLE 1
#else
#define FORGE_TC_AVAILABLE 0
#endif

template<int TILE_B>
__global__ void
#if FORGE_TC_AVAILABLE
__launch_bounds__(FWD_BLOCK)
#endif
fused_mlp_forward_tc_kernel(
    int            n_inputs,
    int            n_outputs,
    int            hidden,
    int            n_layers,
    Activation     act,
    float          act_param,
    float          output_scale,
    bool           do_normalize,
    int            norm_offset,
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__       output,
    float* __restrict__       acts_save,
    int                       B)
{
#if FORGE_TC_AVAILABLE
    using namespace nvcuda;

    constexpr int WM = 16, WN = 16, WK = 8;
    constexpr int N_WARPS = FWD_BLOCK / 32;  // 8

    extern __shared__ char smem_raw[];

    // Shared memory strides: must be multiples of 4 for wmma alignment.
    // Use the hidden dimension directly (already multiple of 16).
    int k_max = hidden;
    if (n_inputs > hidden) k_max = n_inputs;
    k_max = (k_max + 7) & ~7;  // round up to multiple of 8

    int w_stride = k_max;   // weight tile: rows = hidden, cols = k_max
    int a_stride = k_max;   // activation buffers: cols = k_max

    // Shared memory layout:
    //   s_W:     [hidden, w_stride]   — weight tile
    //   s_bias:  [hidden]             — bias vector
    //   s_act_A: [TILE_B, a_stride]   — double buffer A
    //   s_act_B: [TILE_B, a_stride]   — double buffer B
    float* s_W     = (float*)smem_raw;
    float* s_bias  = s_W + hidden * w_stride;
    float* s_act_A = s_bias + hidden;
    float* s_act_B = s_act_A + TILE_B * a_stride;

    int tid    = threadIdx.x;
    int warp_id = tid / 32;

    // Normalization bounds
    const float* lb = nullptr;
    const float* ub = nullptr;
    if (do_normalize) {
        lb = weights + norm_offset;
        ub = lb + n_inputs;
    }

    int acts_stride = hidden * n_layers + n_outputs;

    // Persistent grid-stride loop: each block processes multiple tiles
    int n_tiles = (B + TILE_B - 1) / TILE_B;
    for (int tile_idx = blockIdx.x; tile_idx < n_tiles;
         tile_idx += gridDim.x) {

    int b_base = tile_idx * TILE_B;
    int n_active = (b_base + TILE_B <= B) ? TILE_B : (B - b_base);

    int w_offset = 0;

    // ---- Load input into s_act_A, zero-pad extra columns ----
    {
        int total_buf = TILE_B * a_stride;
        for (int i = tid; i < total_buf; i += blockDim.x)
            s_act_A[i] = 0.0f;
        __syncthreads();

        int elems = TILE_B * n_inputs;
        for (int i = tid; i < elems; i += blockDim.x) {
            int s = i / n_inputs;
            int d = i % n_inputs;
            int b_i = b_base + s;
            float val = (b_i < B) ? input[b_i * n_inputs + d] : 0.0f;
            if (do_normalize && b_i < B)
                val = 2.0f * (val - lb[d]) / (ub[d] - lb[d]) - 1.0f;
            s_act_A[s * a_stride + d] = val;
        }
    }
    __syncthreads();

    // Double-buffer pointers (swap each layer, reset per tile)
    float* s_act_in  = s_act_A;
    float* s_act_out = s_act_B;

    // ---- Process each hidden layer with Tensor Cores ----
    for (int l = 0; l < n_layers; l++) {
        int in_dim = (l == 0) ? n_inputs : hidden;
        int in_dim_padded = (in_dim + WK - 1) & ~(WK - 1);  // round up to 8

        const float* W_ptr = weights + w_offset;
        const float* b_ptr = W_ptr + hidden * in_dim;

        // ---- Cooperatively load W into s_W, zero-padding ----
        {
            int w_total = hidden * w_stride;
            for (int i = tid; i < w_total; i += blockDim.x)
                s_W[i] = 0.0f;
            __syncthreads();

            int w_elements = hidden * in_dim;
            for (int i = tid; i < w_elements; i += blockDim.x) {
                int row = i / in_dim;
                int col = i % in_dim;
                s_W[row * w_stride + col] = W_ptr[i];
            }
            for (int i = tid; i < hidden; i += blockDim.x)
                s_bias[i] = b_ptr[i];
        }
        __syncthreads();

        // ---- wmma matmul: s_act_out = s_act_in * W^T ----
        //
        // Y[s, j] = sum_k X[s, k] * W[j, k]
        // A = X [TILE_B, in_dim_padded] row-major  (matrix_a, row_major)
        // B = W [hidden, in_dim_padded] col-major   (matrix_b, col_major = W transposed)
        // C = Y [TILE_B, hidden]
        int tiles_m = TILE_B / WM;
        int tiles_n = hidden / WN;
        int k_steps = in_dim_padded / WK;
        int total_tiles = tiles_m * tiles_n;

        // Zero the output buffer (tiles that don't get written by any warp)
        {
            int out_total = TILE_B * a_stride;
            for (int i = tid; i < out_total; i += blockDim.x)
                s_act_out[i] = 0.0f;
        }
        __syncthreads();

        // Each warp computes tiles in round-robin order
        for (int t = warp_id; t < total_tiles; t += N_WARPS) {
            int mt = t / tiles_n;
            int nt = t % tiles_n;

            wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
            wmma::fill_fragment(acc, 0.0f);

            #pragma unroll 4
            for (int ks = 0; ks < k_steps; ks++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK,
                    wmma::precision::tf32, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK,
                    wmma::precision::tf32, wmma::col_major> b_frag;

                wmma::load_matrix_sync(a_frag,
                    &s_act_in[mt * WM * a_stride + ks * WK], a_stride);
                wmma::load_matrix_sync(b_frag,
                    &s_W[(nt * WN) * w_stride + ks * WK], w_stride);

                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }

            wmma::store_matrix_sync(
                &s_act_out[mt * WM * a_stride + nt * WN], acc,
                a_stride, wmma::mem_row_major);
        }
        __syncthreads();

        // ---- Element-wise: add bias, save pre-activation, apply activation ----
        {
            int save_offset = l * hidden;
            int total = n_active * hidden;
            for (int idx = tid; idx < total; idx += blockDim.x) {
                int s = idx / hidden;
                int j = idx % hidden;
                int b_i = b_base + s;
                float z = s_act_out[s * a_stride + j] + s_bias[j];
                if (acts_save && b_i < B)
                    acts_save[b_i * acts_stride + save_offset + j] = z;
                s_act_out[s * a_stride + j] = apply_activation(z, act, act_param);
            }
        }
        __syncthreads();

        // Swap double-buffer pointers
        float* tmp = s_act_in;
        s_act_in  = s_act_out;
        s_act_out = tmp;

        w_offset += hidden * in_dim + hidden;
    }

    // ---- Output layer (scalar, no activation) ----
    // n_outputs is typically small (2-10), not worth TC overhead.
    {
        const float* W_ptr = weights + w_offset;
        const float* b_ptr = W_ptr + n_outputs * hidden;

        // Load output weights into s_W
        int w_elements = n_outputs * hidden;
        for (int i = tid; i < w_elements; i += blockDim.x) {
            int row = i / hidden;
            int col = i % hidden;
            s_W[row * w_stride + col] = W_ptr[i];
        }
        for (int i = tid; i < n_outputs; i += blockDim.x)
            s_bias[i] = b_ptr[i];
        __syncthreads();

        int save_offset = n_layers * hidden;
        int total_out = n_active * n_outputs;
        for (int idx = tid; idx < total_out; idx += blockDim.x) {
            int s = idx / n_outputs;
            int j = idx % n_outputs;
            int b_i = b_base + s;
            float sum = s_bias[j];
            for (int k = 0; k < hidden; k++)
                sum += s_W[j * w_stride + k] * s_act_in[s * a_stride + k];
            if (acts_save)
                acts_save[b_i * acts_stride + save_offset + j] = sum;
            output[b_i * n_outputs + j] = sum * output_scale;
        }
    }
    __syncthreads();  // Ensure all reads complete before next tile overwrites smem

    }  // end grid-stride loop
#endif  // FORGE_TC_AVAILABLE
}

// ============================================================
//  Backward kernel — templated on TILE_B
// ============================================================
//
// All phases use cooperative (sample, dim) distribution.
// Register staging avoids WAR hazards on s_dact.
//
// For TILE_B <= 2: weights read directly from L2 cache.
// Shared memory reduced from ~67KB to ~1KB per block.

template<int TILE_B>
__global__ void fused_mlp_backward_kernel(
    int            n_inputs,
    int            n_outputs,
    int            hidden,
    int            n_layers,
    Activation     act,
    float          act_param,
    float          output_scale,
    bool           do_normalize,
    int            norm_offset,
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ acts_save,
    const float* __restrict__ d_output,
    float* __restrict__       d_input,
    float* __restrict__       d_weights,
    int                       B,
    float                     beta,
    float* __restrict__       grad_scratch,
    int                       scratch_stride)
{
    constexpr bool USE_SMEM_W = (TILE_B > 2);

    extern __shared__ char smem_raw[];
    int w_cols   = (hidden > n_inputs) ? hidden : n_inputs;
    int w_stride = w_cols + 1;
    int a_stride = w_cols + 1;

    float* s_W;
    float* s_dact;
    float* s_act_in;

    if constexpr (USE_SMEM_W) {
        s_W      = (float*)smem_raw;
        float* s_bias_pad = s_W + hidden * w_stride;
        s_dact   = s_bias_pad + hidden;
        s_act_in = s_dact + TILE_B * a_stride;
    } else {
        s_W      = nullptr;
        s_dact   = (float*)smem_raw;
        s_act_in = s_dact + TILE_B * a_stride;
    }

    int tid    = threadIdx.x;

    // Per-block gradient scratch: each block writes to its own slice (no atomics)
    const bool use_scratch = (grad_scratch != nullptr);
    float* grad_base = use_scratch
        ? (grad_scratch + blockIdx.x * scratch_stride)
        : d_weights;

    int acts_per_sample = hidden * n_layers + n_outputs;

    // ---- Compute output layer weight offset ----
    int out_w_offset = 0;
    for (int l = 0; l < n_layers; l++) {
        int in_dim = (l == 0) ? n_inputs : hidden;
        out_w_offset += hidden * in_dim + hidden;
    }

    // Persistent grid-stride loop: each block processes multiple tiles
    int n_tiles = (B + TILE_B - 1) / TILE_B;
    for (int tile_idx = blockIdx.x; tile_idx < n_tiles;
         tile_idx += gridDim.x) {

    int b_base = tile_idx * TILE_B;
    int n_active = (b_base + TILE_B <= B) ? TILE_B : (B - b_base);

    // ============================================================
    //  Output layer backward
    // ============================================================
    {
        const float* W_ptr = weights + out_w_offset;

        const float* W_rd;
        int W_rd_stride;

        if constexpr (USE_SMEM_W) {
            // Cooperatively load output weights
            int w_elements = n_outputs * hidden;
            for (int i = tid; i < w_elements; i += blockDim.x) {
                int row = i / hidden;
                int col = i % hidden;
                s_W[row * w_stride + col] = W_ptr[i];
            }
            W_rd = s_W;
            W_rd_stride = w_stride;
        } else {
            W_rd = W_ptr;
            W_rd_stride = hidden;
        }

        // Cooperatively load last hidden layer activations into s_act_in
        int last_hidden_save_offset = (n_layers - 1) * hidden;
        {
            int elems = TILE_B * hidden;
            for (int i = tid; i < elems; i += blockDim.x) {
                int s = i / hidden;
                int k = i % hidden;
                int b_i = b_base + s;
                float pre_act = (b_i < B) ?
                    acts_save[b_i * acts_per_sample + last_hidden_save_offset + k] : 0.0f;
                s_act_in[s * a_stride + k] = apply_activation(pre_act, act, act_param);
            }
        }

        // Cooperatively load d_output into s_dact
        {
            int elems = n_active * n_outputs;
            for (int i = tid; i < elems; i += blockDim.x) {
                int s = i / n_outputs;
                int j = i % n_outputs;
                int b_i = b_base + s;
                s_dact[s * a_stride + j] = d_output[b_i * n_outputs + j] * output_scale;
            }
        }
        __syncthreads();

        // ---- Propagate gradient to last hidden layer FIRST (L1 clean) ----
        // Read W before atomicAdd trashes L1 cache.
        float out_prop_regs[MAX_REGS];
        int out_prop_count = 0;
        {
            int total = n_active * hidden;
            for (int idx = tid; idx < total; idx += blockDim.x) {
                int s = idx / hidden;
                int k = idx % hidden;
                float sum = 0.0f;
                for (int j = 0; j < n_outputs; j++)
                    sum += s_dact[s * a_stride + j] * w_load<!USE_SMEM_W>(&W_rd[j * W_rd_stride + k]);
                out_prop_regs[out_prop_count++] = sum;
            }
        }
        __syncthreads();

        // ---- Weight gradient accumulation (output layer) ----
        float* dW_ptr = grad_base + out_w_offset;
        float* db_ptr = dW_ptr + n_outputs * hidden;

        for (int idx = tid; idx < n_outputs * hidden; idx += blockDim.x) {
            int j = idx / hidden;
            int k = idx % hidden;
            float sum = 0.0f;
            for (int s = 0; s < n_active; s++)
                sum += s_dact[s * a_stride + j] * s_act_in[s * a_stride + k];
            if (use_scratch) dW_ptr[idx] += sum;
            else atomicAdd(&dW_ptr[idx], sum);
        }
        for (int j = tid; j < n_outputs; j += blockDim.x) {
            float sum = 0.0f;
            for (int s = 0; s < n_active; s++)
                sum += s_dact[s * a_stride + j];
            if (use_scratch) db_ptr[j] += sum;
            else atomicAdd(&db_ptr[j], sum);
        }
        __syncthreads();

        // ---- Write propagated gradients to s_dact ----
        {
            out_prop_count = 0;
            int total = n_active * hidden;
            for (int idx = tid; idx < total; idx += blockDim.x) {
                int s = idx / hidden;
                int k = idx % hidden;
                s_dact[s * a_stride + k] = out_prop_regs[out_prop_count++];
            }
            __syncthreads();
        }
    }

    // ============================================================
    //  Hidden layers backward (reverse order)
    // ============================================================
    int w_offset_track = out_w_offset;

    for (int l = n_layers - 1; l >= 0; l--) {
        int in_dim = (l == 0) ? n_inputs : hidden;
        w_offset_track -= hidden * in_dim + hidden;

        const float* W_ptr = weights + w_offset_track;

        const float* W_rd;
        int W_rd_stride;

        if constexpr (USE_SMEM_W) {
            // Cooperatively load weights
            int w_elements = hidden * in_dim;
            for (int i = tid; i < w_elements; i += blockDim.x) {
                int row = i / in_dim;
                int col = i % in_dim;
                s_W[row * w_stride + col] = W_ptr[i];
            }
            W_rd = s_W;
            W_rd_stride = w_stride;
        } else {
            W_rd = W_ptr;
            W_rd_stride = in_dim;
        }

        // ---- Cooperative activation gradient ----
        int save_offset = l * hidden;
        {
            int elems = TILE_B * hidden;
            for (int i = tid; i < elems; i += blockDim.x) {
                int s = i / hidden;
                int j = i % hidden;
                int b_i = b_base + s;
                float pre_act = (b_i < B) ?
                    acts_save[b_i * acts_per_sample + save_offset + j] : 0.0f;
                float dy = s_dact[s * a_stride + j];
                s_dact[s * a_stride + j] = apply_activation_grad(
                    pre_act, dy, act, act_param);
            }
        }

        // ---- Cooperative input loading ----
        if (l == 0) {
            const float* lb = nullptr;
            const float* ub = nullptr;
            if (do_normalize) {
                lb = weights + norm_offset;
                ub = lb + n_inputs;
            }
            int elems = TILE_B * n_inputs;
            for (int i = tid; i < elems; i += blockDim.x) {
                int s = i / n_inputs;
                int k = i % n_inputs;
                int b_i = b_base + s;
                float val = (b_i < B) ? input[b_i * n_inputs + k] : 0.0f;
                if (do_normalize)
                    val = 2.0f * (val - lb[k]) / (ub[k] - lb[k]) - 1.0f;
                s_act_in[s * a_stride + k] = val;
            }
        } else {
            int prev_save = (l - 1) * hidden;
            int elems = TILE_B * hidden;
            for (int i = tid; i < elems; i += blockDim.x) {
                int s = i / hidden;
                int k = i % hidden;
                int b_i = b_base + s;
                float pre_act = (b_i < B) ?
                    acts_save[b_i * acts_per_sample + prev_save + k] : 0.0f;
                s_act_in[s * a_stride + k] = apply_activation(pre_act, act, act_param);
            }
        }
        __syncthreads();

        // ---- Gradient propagation FIRST (while L1 is clean) ----
        // atomicAdd to dW trashes L1 cache, so read weights W for gradient
        // propagation before the weight gradient accumulation phase.
        // prop_regs hold results in registers — s_dact is still unmodified.
        float prop_regs[MAX_REGS];
        int prop_count = 0;
        bool has_prop = false;

        if (l > 0) {
            has_prop = true;
            int total = n_active * in_dim;
            for (int idx = tid; idx < total; idx += blockDim.x) {
                int s = idx / in_dim;
                int k = idx % in_dim;
                float sum = 0.0f;
                for (int j = 0; j < hidden; j++)
                    sum += s_dact[s * a_stride + j] * w_load<!USE_SMEM_W>(&W_rd[j * W_rd_stride + k]);
                prop_regs[prop_count++] = sum;
            }
        } else if (d_input) {
            // l == 0: compute d_input from W^T * d_pre (write directly to global)
            has_prop = true;
            int total = n_active * n_inputs;
            for (int idx = tid; idx < total; idx += blockDim.x) {
                int s = idx / n_inputs;
                int k = idx % n_inputs;
                int b_i = b_base + s;
                float sum = 0.0f;
                for (int j = 0; j < hidden; j++)
                    sum += s_dact[s * a_stride + j] * w_load<!USE_SMEM_W>(&W_rd[j * W_rd_stride + k]);
                prop_regs[prop_count++] = sum;
            }
        }
        __syncthreads();

        // ---- Weight gradient accumulation ----
        float* dW_ptr = grad_base + w_offset_track;
        float* db_ptr = dW_ptr + hidden * in_dim;

        // Preload optimization: when blockDim.x % in_dim == 0, each thread
        // handles a fixed k column across all outer iterations. Cache it.
        bool preload = (blockDim.x % in_dim == 0);
        float a_col[TILE_B];
        if (preload) {
            int k_fixed = tid % in_dim;
            for (int s = 0; s < n_active; s++)
                a_col[s] = s_act_in[s * a_stride + k_fixed];
        }

        for (int idx = tid; idx < hidden * in_dim; idx += blockDim.x) {
            int j = idx / in_dim;
            float sum = 0.0f;
            if (preload) {
                for (int s = 0; s < n_active; s++)
                    sum += s_dact[s * a_stride + j] * a_col[s];
            } else {
                int k = idx % in_dim;
                for (int s = 0; s < n_active; s++)
                    sum += s_dact[s * a_stride + j] * s_act_in[s * a_stride + k];
            }
            if (use_scratch) dW_ptr[idx] += sum;
            else atomicAdd(&dW_ptr[idx], sum);
        }
        for (int j = tid; j < hidden; j += blockDim.x) {
            float sum = 0.0f;
            for (int s = 0; s < n_active; s++)
                sum += s_dact[s * a_stride + j];
            if (use_scratch) db_ptr[j] += sum;
            else atomicAdd(&db_ptr[j], sum);
        }
        __syncthreads();

        // ---- Write propagated gradients to s_dact for next layer ----
        if (l > 0 && has_prop) {
            prop_count = 0;
            int total = n_active * in_dim;
            for (int idx = tid; idx < total; idx += blockDim.x) {
                int s = idx / in_dim;
                int k = idx % in_dim;
                s_dact[s * a_stride + k] = prop_regs[prop_count++];
            }
            __syncthreads();
        } else if (l == 0 && d_input) {
            prop_count = 0;
            int total = n_active * n_inputs;
            for (int idx = tid; idx < total; idx += blockDim.x) {
                int s = idx / n_inputs;
                int k = idx % n_inputs;
                int b_i = b_base + s;
                if (b_i < B)
                    d_input[b_i * n_inputs + k] = prop_regs[prop_count++];
                else
                    prop_count++;
            }
        }
    }  // end layer loop

    }  // end grid-stride loop
}

// ============================================================
//  TC Backward kernel — TF32 Tensor Core path
// ============================================================
//
// TC-accelerated gradient propagation and weight gradient matmuls.
// Activation gradient is element-wise (stays scalar).
// Output layer backward is scalar (n_outputs typically small).
//
// Shared memory layout:
//   s_W:      [hidden, w_stride]   — weight tile
//   s_dact:   [TILE_B, a_stride]   — gradient buffer
//   s_act_in: [TILE_B, a_stride]   — input activations (for dW)
//   s_prop:   [TILE_B, a_stride]   — propagated gradient (temp)
//
// Requires sm_80+.  TILE_B must be a multiple of 16.

template<int TILE_B>
__global__ void
#if FORGE_TC_AVAILABLE
__launch_bounds__(BWD_BLOCK)
#endif
fused_mlp_backward_tc_kernel(
    int            n_inputs,
    int            n_outputs,
    int            hidden,
    int            n_layers,
    Activation     act,
    float          act_param,
    float          output_scale,
    bool           do_normalize,
    int            norm_offset,
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ acts_save,
    const float* __restrict__ d_output,
    float* __restrict__       d_input,
    float* __restrict__       d_weights,
    int                       B,
    float                     beta,
    float* __restrict__       grad_scratch,
    int                       scratch_stride)
{
#if FORGE_TC_AVAILABLE
    using namespace nvcuda;

    constexpr int WM = 16, WN = 16, WK = 8;
    constexpr int N_WARPS = BWD_BLOCK / 32;  // 8

    extern __shared__ char smem_raw[];

    // Gradient propagation writes 16-wide N tiles over in_dim (n_inputs at
    // layer 0), so strides are padded to a multiple of 16, not 8.
    const int k_max = tc_bwd_k_max(hidden, n_inputs);

    int w_stride = k_max;
    int a_stride = k_max;

    // Shared memory layout:
    //   s_W:      [hidden, w_stride]
    //   s_dact:   [TILE_B, a_stride]
    //   s_act_in: [TILE_B, a_stride]
    //   s_prop:   [TILE_B, a_stride]   (gradient propagation output)
    float* s_W      = (float*)smem_raw;
    float* s_dact   = s_W + hidden * w_stride;
    float* s_act_in = s_dact + TILE_B * a_stride;
    float* s_prop   = s_act_in + TILE_B * a_stride;

    int tid     = threadIdx.x;
    int warp_id = tid / 32;

    const bool use_scratch = (grad_scratch != nullptr);
    float* grad_base = use_scratch
        ? (grad_scratch + blockIdx.x * scratch_stride)
        : d_weights;

    int acts_per_sample = hidden * n_layers + n_outputs;

    // ---- Compute output layer weight offset ----
    int out_w_offset = 0;
    for (int l = 0; l < n_layers; l++) {
        int in_dim = (l == 0) ? n_inputs : hidden;
        out_w_offset += hidden * in_dim + hidden;
    }

    // ============================================================
    //  Persistent grid-stride loop: each block processes multiple
    //  tiles to reduce workspace and atomicAdd contention.
    // ============================================================
    int n_tiles = (B + TILE_B - 1) / TILE_B;
    for (int tile_idx = blockIdx.x; tile_idx < n_tiles;
         tile_idx += gridDim.x) {

    int b_base  = tile_idx * TILE_B;
    int n_active = (b_base + TILE_B <= B) ? TILE_B : (B - b_base);

    // ============================================================
    //  Output layer backward (scalar — n_outputs is small)
    // ============================================================
    {
        const float* W_ptr = weights + out_w_offset;

        // Load output weights into s_W
        int w_elements = n_outputs * hidden;
        for (int i = tid; i < w_elements; i += blockDim.x) {
            int row = i / hidden;
            int col = i % hidden;
            s_W[row * w_stride + col] = W_ptr[i];
        }

        // Load last hidden layer activations into s_act_in
        int last_hidden_save_offset = (n_layers - 1) * hidden;
        {
            int elems = TILE_B * hidden;
            for (int i = tid; i < elems; i += blockDim.x) {
                int s = i / hidden;
                int k = i % hidden;
                int b_i = b_base + s;
                float pre_act = (b_i < B) ?
                    acts_save[b_i * acts_per_sample + last_hidden_save_offset + k] : 0.0f;
                s_act_in[s * a_stride + k] = apply_activation(pre_act, act, act_param);
            }
        }

        // Load d_output into s_dact
        {
            // Zero the buffer first (for padding beyond n_outputs)
            int total_buf = TILE_B * a_stride;
            for (int i = tid; i < total_buf; i += blockDim.x)
                s_dact[i] = 0.0f;
            __syncthreads();

            int elems = n_active * n_outputs;
            for (int i = tid; i < elems; i += blockDim.x) {
                int s = i / n_outputs;
                int j = i % n_outputs;
                int b_i = b_base + s;
                s_dact[s * a_stride + j] = d_output[b_i * n_outputs + j] * output_scale;
            }
        }
        __syncthreads();

        // Gradient propagation to last hidden layer (scalar — n_outputs is small)
        {
            int total = n_active * hidden;
            for (int idx = tid; idx < total; idx += blockDim.x) {
                int s = idx / hidden;
                int k = idx % hidden;
                float sum = 0.0f;
                for (int j = 0; j < n_outputs; j++)
                    sum += s_dact[s * a_stride + j] * s_W[j * w_stride + k];
                s_prop[s * a_stride + k] = sum;
            }
        }
        __syncthreads();

        // Weight gradient (output layer)
        float* dW_ptr = grad_base + out_w_offset;
        float* db_ptr = dW_ptr + n_outputs * hidden;

        for (int idx = tid; idx < n_outputs * hidden; idx += blockDim.x) {
            int j = idx / hidden;
            int k = idx % hidden;
            float sum = 0.0f;
            for (int s = 0; s < n_active; s++)
                sum += s_dact[s * a_stride + j] * s_act_in[s * a_stride + k];
            if (use_scratch) dW_ptr[idx] += sum;
            else atomicAdd(&dW_ptr[idx], sum);
        }
        for (int j = tid; j < n_outputs; j += blockDim.x) {
            float sum = 0.0f;
            for (int s = 0; s < n_active; s++)
                sum += s_dact[s * a_stride + j];
            if (use_scratch) db_ptr[j] += sum;
            else atomicAdd(&db_ptr[j], sum);
        }
        __syncthreads();

        // Copy propagated gradients to s_dact for hidden layer processing.
        // Both buffers are [TILE_B, a_stride]; a_stride exceeds hidden when
        // n_inputs > hidden, so index by (s, d) -- a flat copy of
        // n_active*hidden elements would misplace every row after the first.
        // Zero the padding columns and the inactive rows (s_prop rows past
        // n_active are not written this tile) in the same pass.
        {
            int total_buf = TILE_B * a_stride;
            for (int i = tid; i < total_buf; i += blockDim.x) {
                int s = i / a_stride;
                int d = i % a_stride;
                s_dact[i] = (s < n_active && d < hidden) ? s_prop[i] : 0.0f;
            }
        }
        __syncthreads();
    }

    // ============================================================
    //  Hidden layers backward (reverse order) — TC path
    // ============================================================
    int w_offset_track = out_w_offset;

    for (int l = n_layers - 1; l >= 0; l--) {
        int in_dim = (l == 0) ? n_inputs : hidden;
        // N dimension of the propagation GEMM: round up to the 16-wide tile
        // (<= k_max). Rounding to 8 dropped the last 8 columns of d_input
        // whenever n_inputs % 16 was in 1..8.
        int in_dim_padded = (in_dim + WN - 1) & ~(WN - 1);
        w_offset_track -= hidden * in_dim + hidden;

        const float* W_ptr = weights + w_offset_track;

        // ---- Load weights into s_W, zero-padded ----
        {
            int w_total = hidden * w_stride;
            for (int i = tid; i < w_total; i += blockDim.x)
                s_W[i] = 0.0f;
            __syncthreads();

            int w_elements = hidden * in_dim;
            for (int i = tid; i < w_elements; i += blockDim.x) {
                int row = i / in_dim;
                int col = i % in_dim;
                s_W[row * w_stride + col] = W_ptr[i];
            }
        }

        // ---- Activation gradient (element-wise, scalar) ----
        {
            int save_offset = l * hidden;
            int elems = TILE_B * hidden;
            for (int i = tid; i < elems; i += blockDim.x) {
                int s = i / hidden;
                int j = i % hidden;
                int b_i = b_base + s;
                float pre_act = (b_i < B) ?
                    acts_save[b_i * acts_per_sample + save_offset + j] : 0.0f;
                float dy = s_dact[s * a_stride + j];
                s_dact[s * a_stride + j] = apply_activation_grad(
                    pre_act, dy, act, act_param);
            }
        }

        // ---- Load input activations into s_act_in, zero-padded ----
        if (l == 0) {
            const float* lb_ptr = nullptr;
            const float* ub_ptr = nullptr;
            if (do_normalize) {
                lb_ptr = weights + norm_offset;
                ub_ptr = lb_ptr + n_inputs;
            }
            // Zero pad
            int total_buf = TILE_B * a_stride;
            for (int i = tid; i < total_buf; i += blockDim.x)
                s_act_in[i] = 0.0f;
            __syncthreads();

            int elems = TILE_B * n_inputs;
            for (int i = tid; i < elems; i += blockDim.x) {
                int s = i / n_inputs;
                int k = i % n_inputs;
                int b_i = b_base + s;
                float val = (b_i < B) ? input[b_i * n_inputs + k] : 0.0f;
                if (do_normalize)
                    val = 2.0f * (val - lb_ptr[k]) / (ub_ptr[k] - lb_ptr[k]) - 1.0f;
                s_act_in[s * a_stride + k] = val;
            }
        } else {
            int prev_save = (l - 1) * hidden;
            int elems = TILE_B * hidden;
            for (int i = tid; i < elems; i += blockDim.x) {
                int s = i / hidden;
                int k = i % hidden;
                int b_i = b_base + s;
                float pre_act = (b_i < B) ?
                    acts_save[b_i * acts_per_sample + prev_save + k] : 0.0f;
                s_act_in[s * a_stride + k] = apply_activation(pre_act, act, act_param);
            }
        }
        __syncthreads();

        // ---- TC gradient propagation: s_prop = s_dact * W ----
        // dX[s,k] = sum_j dY[s,j] * W[j,k]
        // C = A * B where:
        //   A = dY [TILE_B, hidden]  row-major  (matrix_a, row_major)
        //   B = W  [hidden, in_dim]  row-major  (matrix_b, row_major)
        //   C = dX [TILE_B, in_dim]
        if (l > 0 || d_input) {
            // Zero output buffer
            int out_total = TILE_B * a_stride;
            for (int i = tid; i < out_total; i += blockDim.x)
                s_prop[i] = 0.0f;
            __syncthreads();

            int tiles_m = TILE_B / WM;
            int tiles_n_prop = in_dim_padded / WN;
            int k_steps_prop = hidden / WK;  // hidden is always multiple of 16
            int total_tiles_prop = tiles_m * tiles_n_prop;

            for (int t = warp_id; t < total_tiles_prop; t += N_WARPS) {
                int mt = t / tiles_n_prop;
                int nt = t % tiles_n_prop;

                wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
                wmma::fill_fragment(acc, 0.0f);

                #pragma unroll 4
                for (int ks = 0; ks < k_steps_prop; ks++) {
                    wmma::fragment<wmma::matrix_a, WM, WN, WK,
                        wmma::precision::tf32, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WM, WN, WK,
                        wmma::precision::tf32, wmma::row_major> b_frag;

                    // A = dY: row mt*16..+16, col ks*8..+8
                    wmma::load_matrix_sync(a_frag,
                        &s_dact[mt * WM * a_stride + ks * WK], a_stride);
                    // B = W: row ks*8..+8, col nt*16..+16
                    // W is [hidden, in_dim], row-major. We want B[K,N] row-major.
                    // B[k,n] = W[k,n] at offset k * w_stride + n
                    wmma::load_matrix_sync(b_frag,
                        &s_W[ks * WK * w_stride + nt * WN], w_stride);

                    wmma::mma_sync(acc, a_frag, b_frag, acc);
                }

                wmma::store_matrix_sync(
                    &s_prop[mt * WM * a_stride + nt * WN], acc,
                    a_stride, wmma::mem_row_major);
            }
            __syncthreads();
        }

        // ---- Weight gradient: dW = dY^T * X (TC for hidden-to-hidden) ----
        // dW[j,k] = sum_s dY[s,j] * X[s,k]
        float* dW_ptr = grad_base + w_offset_track;
        float* db_ptr = dW_ptr + hidden * in_dim;

        if (l > 0 && in_dim == hidden) {
            // TC weight gradient: dW = dY^T * X
            // C = A * B where:
            //   A = dY^T [hidden, TILE_B] — load dY col-major (transpose)
            //   B = X    [TILE_B, hidden] — row-major
            //   C = dW   [hidden, hidden]
            //   M=hidden, N=hidden, K=TILE_B
            int tiles_m_dw = hidden / WM;
            int tiles_n_dw = hidden / WN;
            int k_steps_dw = TILE_B / WK;
            int total_tiles_dw = tiles_m_dw * tiles_n_dw;

            for (int t = warp_id; t < total_tiles_dw; t += N_WARPS) {
                int mt = t / tiles_n_dw;
                int nt = t % tiles_n_dw;

                wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
                wmma::fill_fragment(acc, 0.0f);

                for (int ks = 0; ks < k_steps_dw; ks++) {
                    wmma::fragment<wmma::matrix_a, WM, WN, WK,
                        wmma::precision::tf32, wmma::col_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WM, WN, WK,
                        wmma::precision::tf32, wmma::row_major> b_frag;

                    // A = dY^T: load dY as col-major (transposed)
                    // dY[s, j] at offset s * a_stride + j
                    // col-major A[j, s]: A fragment tile [mt*16..+16, ks*8..+8]
                    // corresponds to dY rows ks*8..+8, cols mt*16..+16
                    wmma::load_matrix_sync(a_frag,
                        &s_dact[ks * WK * a_stride + mt * WM], a_stride);
                    // B = X: row-major [TILE_B, hidden]
                    // B fragment tile [ks*8..+8, nt*16..+16]
                    wmma::load_matrix_sync(b_frag,
                        &s_act_in[ks * WK * a_stride + nt * WN], a_stride);

                    wmma::mma_sync(acc, a_frag, b_frag, acc);
                }

                // Store dW tile to global memory
                // dW is [hidden, in_dim] packed contiguously. Tile (mt, nt) covers
                // rows [mt*16..+16], cols [nt*16..+16].
                // Stage through s_W (done with it for this layer). Each warp
                // needs its OWN 16x16 slot: the warps run concurrently, so a
                // shared slot lets one warp's store overwrite another's tile
                // before it is read back. Capacity: s_W holds hidden*w_stride
                // >= hidden^2 floats = (hidden/16)^2 slots, and a warp only
                // reaches this point if warp_id < total_tiles_dw = (hidden/16)^2.
                float* s_stage = s_W + warp_id * (WM * WN);
                wmma::store_matrix_sync(
                    s_stage, acc, WN, wmma::mem_row_major);
                __syncwarp();

                // Write from staging to global with atomicAdd or scratch.
                // 32 lanes cover 2 rows of 16 per step.
                int warp_lane = tid % 32;
                for (int e = warp_lane; e < WM * WN; e += 32) {
                    int gj = mt * WM + e / WN;
                    int gk = nt * WN + e % WN;
                    float val = s_stage[e];
                    if (use_scratch)
                        dW_ptr[gj * in_dim + gk] += val;
                    else
                        atomicAdd(&dW_ptr[gj * in_dim + gk], val);
                }
                __syncwarp();  // slot reads done before this warp's next store
            }
            __syncthreads();
        } else {
            // Scalar weight gradient (layer 0 with non-aligned in_dim, or fallback)
            for (int idx = tid; idx < hidden * in_dim; idx += blockDim.x) {
                int j = idx / in_dim;
                int k = idx % in_dim;
                float sum = 0.0f;
                for (int s = 0; s < n_active; s++)
                    sum += s_dact[s * a_stride + j] * s_act_in[s * a_stride + k];
                if (use_scratch) dW_ptr[idx] += sum;
                else atomicAdd(&dW_ptr[idx], sum);
            }
        }

        // Bias gradient (always scalar — just sum over batch)
        for (int j = tid; j < hidden; j += blockDim.x) {
            float sum = 0.0f;
            for (int s = 0; s < n_active; s++)
                sum += s_dact[s * a_stride + j];
            if (use_scratch) db_ptr[j] += sum;
            else atomicAdd(&db_ptr[j], sum);
        }
        __syncthreads();

        // ---- Write propagated gradients ----
        if (l > 0) {
            int total = n_active * hidden;
            for (int i = tid; i < total; i += blockDim.x) {
                int s = i / hidden;
                int k = i % hidden;
                s_dact[s * a_stride + k] = s_prop[s * a_stride + k];
            }
            // Zero padding beyond in_dim for next layer
            int total_buf = TILE_B * a_stride;
            for (int i = tid; i < total_buf; i += blockDim.x) {
                int s = i / a_stride;
                int d = i % a_stride;
                if (d >= hidden)
                    s_dact[i] = 0.0f;
            }
            __syncthreads();
        } else if (d_input) {
            int total = n_active * n_inputs;
            for (int idx = tid; idx < total; idx += blockDim.x) {
                int s = idx / n_inputs;
                int k = idx % n_inputs;
                int b_i = b_base + s;
                if (b_i < B)
                    d_input[b_i * n_inputs + k] = s_prop[s * a_stride + k];
            }
        }
    }  // end layer loop

    }  // end grid-stride loop
#endif  // FORGE_TC_AVAILABLE
}

// ============================================================
//  Gradient scratch reduction kernel
// ============================================================

__global__ void reduce_block_grads_kernel(
    const float* __restrict__ scratch,
    float* __restrict__       grad,
    int n_w, int n_blocks, float beta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_w) return;
    float sum = 0.0f;
    for (int b = 0; b < n_blocks; b++)
        sum += scratch[b * n_w + i];
    if (beta == 0.0f)
        grad[i] = sum;
    else
        grad[i] = beta * grad[i] + sum;
}

// ============================================================
//  Runtime compute capability check (for Tensor Core dispatch)
// ============================================================

/// Returns true if the current GPU supports TF32 Tensor Cores (sm_80+).
/// Cached after first call.
static bool gpu_supports_tf32(int device = 0) {
    static int cached_major = -1;
    if (cached_major < 0)
        cudaDeviceGetAttribute(&cached_major, cudaDevAttrComputeCapabilityMajor, device);
    return cached_major >= 8;
}

/// Check if TC path can be used for the given descriptor and batch size.
static bool can_use_tc(const MlpDesc& desc, int B) {
    return desc.use_tf32
        && (desc.hidden % 16 == 0)
        && (desc.hidden <= 128)   // TC kernel needs weights in shared memory
        && (B >= 16)              // TILE_B=16 minimum
        && gpu_supports_tf32();
}

// ============================================================
//  TILE_B auto-selection
// ============================================================
//
// Choose the largest TILE_B that produces enough blocks for good SM occupancy.
// Target: at least 128 blocks (~1 per SM on RTX 4090).

// Backward: maximize block count to hide atomicAdd latency with concurrency.
// For H > 128, force L2 mode (TILE_B <= 2) since the weight tile exceeds
// shared memory capacity (256*257*4 = 263KB vs 96-100KB hardware limit).
static int select_tile_b(int B, int hidden) {
    if (hidden > 128) return (B >= 2) ? 2 : 1;  // L2 mode required
    int target_blocks = internal::get_sm_count();
    if (internal::divup(B, 32) >= target_blocks) return 32;
    if (internal::divup(B, 16) >= target_blocks) return 16;
    if (internal::divup(B,  8) >= target_blocks) return  8;
    if (internal::divup(B,  4) >= target_blocks) return  4;
    if (internal::divup(B,  2) >= target_blocks) return  2;
    return 1;
}

// Forward: prefer SMEM mode (TILE_B >= 4) for weight reuse even with fewer blocks.
// SMEM avoids repeated L2 reads of the weight matrix across samples.
// The floor at TILE_B=4 skips TILE_B=2 and TILE_B=1, since forward has no atomicAdd
// and benefits from shared memory weight reuse more than from extra blocks.
// For H > 128, force L2 mode — weight tile doesn't fit in shared memory.
static int select_tile_b_forward(int B, int hidden) {
    if (hidden > 128) return (B >= 2) ? 2 : 1;  // L2 mode required
    int target_blocks = internal::get_sm_count();
    if (internal::divup(B, 32) >= target_blocks) return 32;
    if (internal::divup(B, 16) >= target_blocks) return 16;
    if (internal::divup(B,  8) >= target_blocks) return  8;
    if (internal::divup(B,  4) >= target_blocks) return  4;
    // Hard floor: SMEM mode (TILE_B >= 4) always for forward
    return (B >= 4) ? 4 : (B >= 2) ? 2 : 1;
}

// ============================================================
//  Host launch functions
// ============================================================

void fused_mlp_forward(
    const MlpDesc& desc,
    const float*   input,
    const float*   weights,
    float*         output,
    int            B,
    float*         acts_save,
    cudaStream_t   stream)
{
    if (B <= 0) return;

    int norm_offset = 0;
    if (desc.normalize) {
        MlpDesc tmp = desc;
        tmp.normalize = false;
        norm_offset = mlp_weight_count(tmp);
    }

    // ---- TC path (TF32 Tensor Cores) ----
    if (can_use_tc(desc, B)) {
        // Tile size 32: more work per block, better amortization of weight loads.
        // Shared memory: s_W[H*H] + s_bias[H] + 2*s_act[32*H] = 64KB + 0.5KB + 32KB = 96.5KB
        // Fits within 100KB SM limit for H<=128.
        constexpr int tc_tile = 32;
        int grid  = internal::divup(B, tc_tile);
        int block = FWD_BLOCK;

        // Cap grid for persistent kernel: each block loops over multiple
        // tiles via grid-stride. Reduces weight reload count.
        int max_grid = internal::get_sm_count();
        if (grid > max_grid) grid = max_grid;

        int k_max = desc.hidden;
        if (desc.n_inputs > desc.hidden) k_max = desc.n_inputs;
        k_max = (k_max + 7) & ~7;

        int w_stride = k_max;
        int a_stride = k_max;

        // s_W + s_bias + s_act_A + s_act_B
        int smem = (desc.hidden * w_stride + desc.hidden
                  + 2 * tc_tile * a_stride) * (int)sizeof(float);

        auto kernel_fn = fused_mlp_forward_tc_kernel<tc_tile>;
        if (smem > 48 * 1024)
            cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kernel_fn<<<grid, block, smem, stream>>>(
            desc.n_inputs, desc.n_outputs, desc.hidden, desc.n_layers,
            desc.act, desc.act_param, desc.output_scale, desc.normalize, norm_offset,
            input, weights, output, acts_save, B);
        return;
    }

    // ---- Scalar path ----
    int tile_b = select_tile_b_forward(B, desc.hidden);
    int grid   = internal::divup(B, tile_b);
    int block  = FWD_BLOCK;

    int w_cols   = (desc.hidden > desc.n_inputs) ? desc.hidden : desc.n_inputs;
    int w_stride = w_cols + 1;
    int a_stride = w_cols + 1;

    int smem;
    if (tile_b <= 2) {
        smem = (tile_b * a_stride) * (int)sizeof(float);
    } else {
        smem = (desc.hidden * w_stride + desc.hidden + tile_b * a_stride) * (int)sizeof(float);
    }

    auto launch = [&](auto kernel_fn) {
        if (smem > 48 * 1024)
            cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kernel_fn<<<grid, block, smem, stream>>>(
            desc.n_inputs, desc.n_outputs, desc.hidden, desc.n_layers,
            desc.act, desc.act_param, desc.output_scale, desc.normalize, norm_offset,
            input, weights, output, acts_save, B);
    };

    switch (tile_b) {
        case 1:  launch(fused_mlp_forward_kernel<1>);  break;
        case 2:  launch(fused_mlp_forward_kernel<2>);  break;
        case 4:  launch(fused_mlp_forward_kernel<4>);  break;
        case 8:  launch(fused_mlp_forward_kernel<8>);  break;
        case 16: launch(fused_mlp_forward_kernel<16>); break;
        default: launch(fused_mlp_forward_kernel<32>); break;
    }
}

void fused_mlp_backward(
    const MlpDesc& desc,
    const float*   input,
    const float*   weights,
    const float*   acts_save,
    const float*   d_output,
    float*         d_input,
    float*         d_weights,
    int            B,
    float          beta,
    cudaStream_t   stream,
    float*         workspace)
{
    if (B <= 0) return;

    int n_w = mlp_weight_count(desc);

    int norm_offset = 0;
    if (desc.normalize) {
        MlpDesc tmp = desc;
        tmp.normalize = false;
        norm_offset = mlp_weight_count(tmp);
    }

    // ---- TC path (TF32 Tensor Cores) ----
    if (can_use_tc(desc, B)) {
        constexpr int tc_tile = 16;
        int grid  = internal::divup(B, tc_tile);
        int block = BWD_BLOCK;

        // Cap grid for persistent kernel: each block loops over multiple
        // tiles via grid-stride. This limits workspace size and eliminates
        // massive atomicAdd contention (e.g., B=500K → 31K blocks → 128).
        int max_grid = internal::get_sm_count();
        if (grid > max_grid) grid = max_grid;

        float* grad_scratch   = nullptr;
        int    scratch_stride = 0;
        if (workspace && grid > 1) {
            grad_scratch   = workspace;
            scratch_stride = n_w;
        } else if (beta == 0.0f) {
            cudaMemsetAsync(d_weights, 0, n_w * sizeof(float), stream);
        }

        int k_max = tc_bwd_k_max(desc.hidden, desc.n_inputs);

        int w_stride = k_max;
        int a_stride = k_max;

        // s_W + s_dact + s_act_in + s_prop
        int smem = (desc.hidden * w_stride
                  + 3 * tc_tile * a_stride) * (int)sizeof(float);

        auto kernel_fn = fused_mlp_backward_tc_kernel<tc_tile>;
        if (smem > 48 * 1024)
            cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kernel_fn<<<grid, block, smem, stream>>>(
            desc.n_inputs, desc.n_outputs, desc.hidden, desc.n_layers,
            desc.act, desc.act_param, desc.output_scale, desc.normalize, norm_offset,
            input, weights, acts_save, d_output,
            d_input, d_weights, B, beta,
            grad_scratch, scratch_stride);

        if (grad_scratch) {
            int r_block = 256;
            int r_grid  = internal::divup(n_w, r_block);
            reduce_block_grads_kernel<<<r_grid, r_block, 0, stream>>>(
                grad_scratch, d_weights, n_w, grid, beta);
        }
        return;
    }

    // ---- Scalar path ----
    int tile_b = select_tile_b(B, desc.hidden);
    int grid   = internal::divup(B, tile_b);
    int block  = BWD_BLOCK;

    // Cap grid for persistent kernel (prevents workspace overflow at large B)
    int max_grid_scalar = internal::get_sm_count();
    if (grid > max_grid_scalar) grid = max_grid_scalar;

    float* grad_scratch    = nullptr;
    int    scratch_stride  = 0;
    if (workspace && grid > 1) {
        grad_scratch  = workspace;
        scratch_stride = n_w;
    } else if (beta == 0.0f) {
        cudaMemsetAsync(d_weights, 0, n_w * sizeof(float), stream);
    }

    int w_cols   = (desc.hidden > desc.n_inputs) ? desc.hidden : desc.n_inputs;
    int w_stride = w_cols + 1;
    int a_stride = w_cols + 1;

    int smem;
    if (tile_b <= 2) {
        smem = (tile_b * a_stride * 2) * (int)sizeof(float);
    } else {
        smem = (desc.hidden * w_stride + desc.hidden
              + tile_b * a_stride * 2) * (int)sizeof(float);
    }

    auto launch = [&](auto kernel_fn) {
        if (smem > 48 * 1024)
            cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kernel_fn<<<grid, block, smem, stream>>>(
            desc.n_inputs, desc.n_outputs, desc.hidden, desc.n_layers,
            desc.act, desc.act_param, desc.output_scale, desc.normalize, norm_offset,
            input, weights, acts_save, d_output,
            d_input, d_weights, B, beta,
            grad_scratch, scratch_stride);
    };

    switch (tile_b) {
        case 1:  launch(fused_mlp_backward_kernel<1>);  break;
        case 2:  launch(fused_mlp_backward_kernel<2>);  break;
        case 4:  launch(fused_mlp_backward_kernel<4>);  break;
        case 8:  launch(fused_mlp_backward_kernel<8>);  break;
        case 16: launch(fused_mlp_backward_kernel<16>); break;
        default: launch(fused_mlp_backward_kernel<32>); break;
    }

    if (grad_scratch) {
        int r_block = 256;
        int r_grid  = internal::divup(n_w, r_block);
        reduce_block_grads_kernel<<<r_grid, r_block, 0, stream>>>(
            grad_scratch, d_weights, n_w, grid, beta);
    }
}

int fused_mlp_backward_workspace_count(const MlpDesc& desc, int B) {
    int tile_b;
    if (can_use_tc(desc, B))
        tile_b = 16;  // TC tile size
    else
        tile_b = select_tile_b(B, desc.hidden);
    int n_blocks = internal::divup(B, tile_b);
    if (n_blocks <= 1) return 0;
    // Cap to match persistent kernel grid (prevents int overflow at large B)
    int max_grid = internal::get_sm_count();
    if (n_blocks > max_grid) n_blocks = max_grid;
    return n_blocks * mlp_weight_count(desc);
}

// ============================================================
//  Weight initialization
// ============================================================

void mlp_init_weights(
    const MlpDesc& desc,
    float*         d_weights,
    uint64_t       seed,
    InitStyle      style,
    const float*   norm_lb,
    const float*   norm_ub)
{
    int offset = 0;

    // Layer 0: fan_in = n_inputs
    float bound = 1.0f / sqrtf((float)desc.n_inputs);
    int wn = desc.hidden * desc.n_inputs;
    fill_uniform(d_weights + offset, wn, -bound, bound, seed, offset);
    offset += wn;
    fill_uniform(d_weights + offset, desc.hidden, -bound, bound, seed, offset);
    offset += desc.hidden;

    // Hidden layers 1..n_layers-1: fan_in = hidden
    bound = 1.0f / sqrtf((float)desc.hidden);
    for (int l = 1; l < desc.n_layers; l++) {
        wn = desc.hidden * desc.hidden;
        fill_uniform(d_weights + offset, wn, -bound, bound, seed, offset);
        offset += wn;
        fill_uniform(d_weights + offset, desc.hidden, -bound, bound, seed, offset);
        offset += desc.hidden;
    }

    // Output layer: fan_in = hidden
    wn = desc.n_outputs * desc.hidden;
    fill_uniform(d_weights + offset, wn, -bound, bound, seed, offset);
    offset += wn;
    fill_uniform(d_weights + offset, desc.n_outputs, -bound, bound, seed, offset);
    offset += desc.n_outputs;

    // Normalization bounds (host → device)
    if (desc.normalize && norm_lb && norm_ub) {
        cudaMemcpy(d_weights + offset, norm_lb,
                   desc.n_inputs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights + offset + desc.n_inputs, norm_ub,
                   desc.n_inputs * sizeof(float), cudaMemcpyHostToDevice);
    }
}

}} // namespace forge::nn
