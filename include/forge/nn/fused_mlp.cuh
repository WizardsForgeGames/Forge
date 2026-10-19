// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Fused MLP: entire multi-layer perceptron forward and backward in a single
// CUDA kernel per pass.  No intermediate global memory between layers —
// activations flow through shared memory.
//
// Designed for small-to-medium hidden widths where framework overhead and
// memory bandwidth dominate actual compute in PyTorch.  A fused kernel
// eliminates all kernel launch overhead and intermediate DRAM round-trips,
// yielding 10-50x speedup for small MLPs.
//
// For H <= 128: weights cached in shared memory (fastest path).
// For H > 128 (up to 256): weights read from L2 cache, activations still
//   flow through shared memory.  Slightly lower throughput but still fully
//   fused -- no intermediate global memory between layers.
//
// TF32 Tensor Core path (use_tf32=true, requires sm_80+ and hidden % 16 == 0):
//   Uses wmma 16x16x8 TF32 fragments for hidden-layer matmuls.
//   Activations double-buffered in shared memory (no register staging).
//   TF32 truncates FP32 inputs to 10-bit mantissa; accumulation is full FP32.
//   Falls back to scalar path when dimensions don't align or GPU < sm_80.
//
// Weight packing layout (contiguous float buffer):
//   Layer 0: W0[hidden, n_inputs] + b0[hidden]
//   Layer 1: W1[hidden, hidden]   + b1[hidden]
//   ...
//   Layer L: WL[hidden, hidden]   + bL[hidden]     (L = n_layers - 1)
//   Output:  Wout[n_outputs, hidden] + bout[n_outputs]
//   (Optional, if normalize=true):
//     lb[n_inputs]  — lower bounds for input normalization
//     ub[n_inputs]  — upper bounds for input normalization
//
// All weights are row-major: W[out_features, in_features].
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

/// Activation function selector for fused MLP.
enum class Activation { CELU, GELU, ReLU, Sigmoid, SiLU, Tanh };

/// Descriptor for a fused MLP architecture.
/// Fill once, pass to forward/backward.
///
/// Every field has a default, so `MlpDesc desc;` followed by assignments is
/// safe: an unset `use_tf32` means the FP32 path, never stack garbage.
struct MlpDesc {
    int        n_inputs     = 0;      // input dimension  (e.g. 9)
    int        n_outputs    = 0;      // output dimension (e.g. 2)
    int        hidden       = 0;      // hidden width     (e.g. 64, max 256)
    int        n_layers     = 0;      // number of hidden layers (e.g. 5)
    Activation act          = Activation::CELU;  // activation for hidden layers
    float      act_param    = 1.0f;   // activation parameter (alpha for CELU)
    float      output_scale = 1.0f;   // scale applied to final output (e.g. 0.01 for /100)
    bool       normalize    = false;  // apply min-max normalization to inputs
    bool       use_tf32     = false;  // use TF32 Tensor Cores (sm_80+, hidden % 16 == 0)
};

/// Returns total number of floats in the packed weight buffer.
int mlp_weight_count(const MlpDesc& desc);

/// Returns number of floats needed for acts_save buffer (for backward).
/// acts_save stores pre-activation values: [B, hidden * n_layers + n_outputs].
int mlp_acts_save_count(const MlpDesc& desc, int B);

/// Fused MLP forward pass — entire network in one kernel.
///
/// input:     [B, n_inputs]   row-major
/// weights:   packed weight buffer (see layout above)
/// output:    [B, n_outputs]  row-major
/// acts_save: [B, hidden * n_layers + n_outputs] — pass nullptr for inference-only
///
/// When acts_save is provided, pre-activation values (before activation fn)
/// are saved for each layer.  Required for backward pass.
void fused_mlp_forward(
    const MlpDesc& desc,
    const float*   input,
    const float*   weights,
    float*         output,
    int            B,
    float*         acts_save = nullptr,
    cudaStream_t   stream    = nullptr);

/// Fused MLP backward pass — chain rule through all layers in one kernel.
///
/// input:     [B, n_inputs]  (same as forward, must be preserved)
/// weights:   packed weights (same as forward)
/// acts_save: pre-activation saves from forward (required)
/// d_output:  [B, n_outputs] upstream gradient
/// d_input:   [B, n_inputs]  gradient w.r.t. the raw input (nullptr to skip);
///            with normalize=true this includes the 2/(ub - lb) chain factor
/// d_weights: packed gradient buffer (same layout as weights)
/// beta:      0.0 = overwrite d_weights, 1.0 = accumulate
/// workspace: optional per-block gradient scratch buffer. When provided,
///            each block writes gradients to a private slice (no atomicAdd),
///            then a reduction kernel sums across blocks into d_weights.
///            Allocate with fused_mlp_backward_workspace_count() floats;
///            contents on entry are ignored (the call zeroes what it uses).
void fused_mlp_backward(
    const MlpDesc& desc,
    const float*   input,
    const float*   weights,
    const float*   acts_save,
    const float*   d_output,
    float*         d_input,
    float*         d_weights,
    int            B,
    float          beta      = 0.0f,
    cudaStream_t   stream    = nullptr,
    float*         workspace = nullptr);

/// Returns number of floats needed for backward workspace (per-block gradient scratch).
/// Returns 0 if workspace would not help (single block).
int fused_mlp_backward_workspace_count(const MlpDesc& desc, int B);

/// Weight initialization style.
enum class InitStyle { KaimingUniform };

/// Initialize MLP weights with the given style.
///
/// KaimingUniform: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) per layer.
///   Matches PyTorch nn.Linear default initialization.
///
/// If desc.normalize is true and norm_lb/norm_ub are non-null,
/// copies normalization bounds [n_inputs] into the tail of the weight buffer.
/// Pass nullptr for norm_lb/norm_ub to leave normalization bounds uninitialized.
void mlp_init_weights(
    const MlpDesc& desc,
    float*         d_weights,
    uint64_t       seed,
    InitStyle      style   = InitStyle::KaimingUniform,
    const float*   norm_lb = nullptr,   // host [n_inputs], lower bounds
    const float*   norm_ub = nullptr);  // host [n_inputs], upper bounds

}} // namespace forge::nn
