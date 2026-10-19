# Forge GPU Compute Library — Architecture

## Overview

Forge is a C++17/CUDA static library for GPU-accelerated neural network training and physics simulation. It targets NVIDIA Ampere+ (sm_80, sm_86, sm_89) and is designed for scenarios where framework overhead dominates compute — small-to-medium MLPs, real-time simulation data generation, and quantized inference. All kernel APIs accept raw pointers. An optional `DeviceBuffer<T>` RAII wrapper exists for convenience but is never required.

**Build:** CMake (Visual Studio 2022 generator on Windows), CUDA Toolkit
**Targets:** sm_80 (A100), sm_86 (RTX 3090), sm_89 (RTX 4090)
**Flags:** `-O3`, `--use_fast_math`, `--ptxas-options=-v`, `--expt-relaxed-constexpr`

---

## Directory Structure

```
Forge/
  include/forge/
    forge.h                   # Master include (pulls everything)
    error.h                   # FORGE_CHECK macro
    types.h                   # half_t typedef
    buffer.h                  # DeviceBuffer<T> RAII wrapper
    primitives/               # Composable __device__ building blocks
      activation.cuh            Activation fwd/bwd (LeakyReLU, Tanh, GELU, Sigmoid, CELU)
      reduce.cuh                Warp/block reductions (sum, max, min, broadcast)
      atomic.cuh                Float atomic min/max via CAS
      distance.cuh              Euclidean, Poincare distance
      geometry.cuh              Poincare ball projection, conformal factor
      sparse.cuh                CSR accessor helpers
    optim/
      adam.cuh                  Adam optimizer kernel
      grad_clip.cuh             Gradient norm + clipping (fully on-device)
      lr_schedule.h             LR scheduler (warmup + cosine + SGDR restarts)
    nn/
      activations.cuh           Activation backward kernels (standalone)
      softmax_ce.cuh            Fused softmax + cross-entropy
      fused_gemm_softmax_ce.cuh Fused GEMM + softmax + CE (online softmax, no logit materialization)
      embedding.cuh             Embedding gather/scatter
      bias.cuh                  Bias gradient accumulation
      gating.cuh                Per-unit sigmoid gating
      trilinear.cuh             3D trilinear sample/scatter with position gradients
      mse.cuh                   MSE loss (forward + backward)
      linear.cuh                FP32 linear layer (forward, backward-data, backward-weight)
      reduce.cuh                Reduction kernels (sum-to-scalar, mean/sum over dim0)
      rng.cuh                   cuRAND Philox RNG (normal, uniform)
      fused_mlp.cuh             Fused MLP (entire network in one kernel)
    quant/
      quantize.cuh              FP16->INT8 per-token activation quantization
      dequantize.cuh            INT32->FP16 dequantization + bias
    gemm/
      fp32_tiled.cuh            FP32 tiled GEMM (all transpose combos, split-K)
      int8_dense.cuh            Dense INT8 Tensor Core GEMM + dequant
      int8_sparse.cuh           Sparse INT8 TC GEMM (2:4 structured, PTX mma.sp)
    sparsity/
      prune_compress.cuh        2:4 structured pruning + compression
      pack_metadata.cuh         Pack sparse metadata nibbles to hardware format
    sim/
      thermal.cuh               3D thermal conduction simulation (explicit Euler FD)
      thermal_datagen.cuh       GPU-accelerated training data generation
  src/
    internal/
      cuda_utils.cuh            divup(), kDefaultBlockSize (256)
      reduction.cuh             Internal wrappers delegating to forge::primitives
      cp_async.cuh              SM80+ cp.async infrastructure (16-byte global->smem)
    nn/                         Kernel implementations for nn/ headers
    optim/                      Kernel implementations for optim/ headers
    quant/                      Kernel implementations for quant/ headers
    gemm/                       Kernel implementations for gemm/ headers
    sparsity/                   Kernel implementations for sparsity/ headers
    sim/                        Kernel implementations for sim/ headers
  python/
    setup.py                    PyTorch extension build (forge-gpu package)
    forge/
      csrc/bindings.cpp         C++/Python bindings
      nn/                       PyTorch autograd wrappers
      optim/                    Adam optimizer PyTorch wrapper
      quant/                    Quantization pipeline (observers, fake_quantize, convert)
      compile/                  torch.compile backend + pattern matching
      training/
        trainer.py              GPU-native training loop with CUDA Graph capture
        convergence.py          Convergence predictor (harmonic decay model, LR recommendations)
  docs/
    training.md                 Python training utilities guide
  tests/
    CMakeLists.txt              30 test executables
```

---

## Layer 1: Core Infrastructure

### `error.h`
`FORGE_CHECK(call)` — CUDA error checking macro, prints file/line on failure, aborts.

### `types.h`
`half_t` = `__half` (CUDA FP16).

### `buffer.h`
`DeviceBuffer<T>` — move-only RAII wrapper. `alloc(n)` (owning, zero-init), `wrap(ptr, n)` (non-owning). Implicit conversion to `T*`.

### `src/internal/cuda_utils.cuh`
`divup(a, b)` (ceiling division), `kDefaultBlockSize = 256`.

### `src/internal/reduction.cuh`
Thin forwarding: `internal::warp_reduce_sum` → `primitives::warp_reduce_sum`, etc.

### `src/internal/cp_async.cuh`
SM80+ (Ampere) asynchronous global→shared memory copy:
- `cp_async_16(dst_smem, src_gmem)` — 16-byte copy, L1+L2 cache
- `cp_async_16_stream(dst_smem, src_gmem)` — L2 evict-first hint (one-shot data)
- `cp_async_commit()` / `cp_async_wait_group<N>()` — pipeline control
- `a_swiz(row, col)` / `b_swiz(row, col)` — XOR bank-conflict-free smem swizzle

---

## Layer 2: Primitives (`forge::primitives`)

All functions `__device__ __forceinline__` — zero overhead, composable. No `__global__` kernels.

### `activation.cuh`
| Function | Notes |
|----------|-------|
| `leaky_relu(z, slope=0.01)` / `leaky_relu_grad(z, dy, slope)` | |
| `tanh_grad(tanh_out, dy)` | Backward given *output*, not input |
| `gelu(z)` / `gelu_grad(z, dy)` | Tanh approximation |
| `sigmoid(x)` / `sigmoid_grad(sig_out)` | Backward given *output* |
| `celu(x, alpha=1.0)` / `celu_grad(z, dy, alpha)` | |

### `reduce.cuh`

**Reduction op structs:** `SumOp`, `MaxOp`, `MinOp` — each provides `identity()`, `combine(a,b)`, `shfl_combine(val, offset)`.

**Generic templates:**
| Template | Level | Result |
|----------|-------|--------|
| `warp_reduce<Op>(val)` | Warp (32 threads) | Lane 0 via `__shfl_down_sync` |
| `block_reduce<Op>(val, smem)` | Block | Thread 0 only |
| `block_broadcast<Op>(val, smem)` | Block | All threads (reduce + broadcast) |

**Monomorphic wrappers** (backward-compatible, delegate to templates):
| Function | Level | Result |
|----------|-------|--------|
| `warp_reduce_sum/max/min(val)` | Warp | Lane 0 |
| `block_reduce_sum/max/min(val, smem)` | Block | Thread 0 only |
| `block_broadcast_sum/max/min(val, smem)` | Block | All threads |

**Shared memory requirements:** `blockDim.x / 32` floats for reduce, `+1` for broadcast.

**Usage pattern for SumSq / AbsMax:** Caller applies transform before reducing:
```cpp
float sq = x * x;
float sumsq = block_reduce<SumOp>(sq, smem);   // sum of squares
float ab = fabsf(x);
float amax = block_reduce<MaxOp>(ab, smem);     // absolute max
```

### `atomic.cuh`
`atomic_max_float(addr, val)` / `atomic_min_float(addr, val)` — CAS loop with `__float_as_int`.

### `distance.cuh`
`dist_sq(a, b, D)`, `norm_sq(x, D)`, `dist_poincare(a, b, D)`.

### `geometry.cuh`
`poincare_project(x, D, max_norm)`, `conformal_factor(norm_sq, power)`, `boundary_repulsion_grad(grad, x, D, lambda)`.

### `sparse.cuh`
`csr_row_begin/end/nnz(row_ptr, row)` — CSR/CSC accessor helpers.

---

## Layer 3: Neural Network Ops (`forge::nn`)

### Activations (`nn/activations`)
Standalone activation backward kernels (elementwise, `divup(n,256), 256`):
- `leaky_relu_backward_kernel(z, da, dz, n, slope)`
- `tanh_backward_kernel(da, a_out, dz, n)`
- `celu_backward_kernel(z, da, dz, n, alpha)`

### Fused MLP (`nn/fused_mlp`)
Entire multi-layer perceptron in a single CUDA kernel per pass. Activations flow through shared memory — no intermediate global memory between layers. Max hidden width H ≤ 128.

**Data structures:**
- `enum class Activation { CELU, GELU, ReLU, Sigmoid, Tanh }`
- `struct MlpDesc { n_inputs, n_outputs, hidden, n_layers, act, act_param, output_scale, normalize }`

**Weight packing:** `W0[H, n_in] + b0[H]`, then `WL[H, H] + bL[H]` per hidden layer, then `Wout[n_out, H] + bout[n_out]`, optional normalization bounds.

**Kernels:**
| Kernel | Grid | Block | Shared Memory |
|--------|------|-------|---------------|
| `fused_mlp_forward_kernel` | `divup(B, 32)` | 32 (1 warp) | `s_W[H][w_stride] + s_bias[H] + s_act[32][a_stride]` |
| `fused_mlp_backward_kernel` | `divup(B, 32)` | 128 (4 warps) | `s_W + s_bias + s_dact + s_act_in` |

**Key optimizations:**
- Bank-conflict avoidance: stride = cols + 1
- Weights loaded layer-by-layer into smem (cooperative)
- Multi-warp backward (BWD_BLOCK=128) for latency hiding
- Flat inner-loop accumulation beats `__shfl_down` by 3x for smem-resident data
- `atomicAdd` for weight gradient accumulation across batch tiles
- `cudaFuncSetAttribute` for extended smem when H ≥ 96

### Linear (`nn/linear`)
FP32 linear layer — thin wrappers around `forge::gemm::fp32_gemm`. Weight layout: `W[N_out, N_in]` row-major.
- `linear_forward(input, weight, bias, output, B, N, K)` — `Y = X @ W^T + bias`
- `linear_backward_data(d_output, weight, d_input, B, N, K)` — `dX = dY @ W`
- `linear_backward_weight(d_output, input, d_weight, B, N, K, beta)` — `dW = dY^T @ X`
- Internal `add_bias_kernel` (elementwise).

### MSE Loss (`nn/mse`)
Mean Squared Error forward and backward.
- **Small path** (n ≤ 65536): fused single-block `mse_forward_small_kernel` with `block_reduce<SumOp>`
- **Large path**: `squared_diff_kernel` → `reduce_sum` → `scale_scalar_kernel`
- Backward: `mse_backward_kernel` — `d_pred[i] = coeff * (pred[i] - target[i])`

### Reduce (`nn/reduce`)
Templated two-pass scalar reduction using `block_reduce<Op>` primitives internally.
- `reduce_sum(x, out, n)` — Two-pass: `reduce_partial_kernel<SumOp>` → `reduce_final_kernel<SumOp>`. Single-pass when grid==1. Uses `cudaMalloc/FreeAsync` for intermediate buffer.
- `reduce_sumsq(x, out, n)` — Sum of squares via `SumOp` + `SquareTransform`
- `reduce_absmax(x, out, n)` — Absolute maximum via `MaxOp` + `AbsTransform`
- `reduce_sum_dim0(x, out, D0, D1)` — `[D0, D1]` → `[D1]`
- `reduce_mean_dim0(x, out, D0, D1)` — same with `1/D0` scaling

### RNG (`nn/rng`)
cuRAND Philox 4x32_10, counter-based (no per-thread state):
- `fill_normal(data, n, mean, std, seed, offset)`
- `fill_uniform(data, n, lo, hi, seed, offset)`

### Trilinear (`nn/trilinear`)
3D grid sampling with full position gradients. Grid layout: `[C, G, G, G, B]` channels-first, batch-last.
- `trilinear_sample_kernel` — Gather C channels from 3D grid (8-point weighted average)
- `trilinear_scatter_kernel` — Splat via `atomicAdd`
- `trilinear_sample_pos_grad_kernel` — Position gradients
- `trilinear_scatter_backward_kernel` — Scatter backward + position gradients

### Softmax + Cross-Entropy (`nn/softmax_ce`)
Fused forward and backward. Logits/probs layout: `[V, B]` col-major.
- `softmax_ce_forward_kernel` — 1 block per sample, online max+sum reduction. Smem: `threads * 8` bytes.
- `softmax_ce_backward_kernel` — Elementwise `probs - one_hot(target)`
- `softmax_ce_backward_bias_kernel` — Fused backward + bias gradient (1 block per V-row)

### Fused GEMM + Softmax + CE (`nn/fused_gemm_softmax_ce`)
FlashAttention-style online softmax — logits tensor never materialized in global memory, saving one full `V*B` DRAM round-trip.

### Embedding (`nn/embedding`)
- `embedding_gather_kernel` — Forward: context → output col-major
- `embedding_scatter_kernel` — Backward: `atomicAdd` to `d_embeddings[V, E]`

### Bias (`nn/bias`)
- `bias_grad_accumulate_kernel` — Column-sum over batch: `d_bias[i] += sum_b d_upstream[i*B+b]`

### Gating (`nn/gating`)
Per-unit sigmoid gating. `x` is `[N, B]`, `w` and `b` are `[N]`.
- `sigmoid_gate_forward` — `gated = sigmoid(w*x+b) * x` (1 thread per element)
- `sigmoid_gate_backward` — 1 block per unit, cooperative `block_reduce<SumOp>` over batch for `dw`/`db`. Eliminated O(B) atomicAdd contention from the original per-element kernel.

---

## Layer 4: Optimizer (`forge::optim`)

### Adam (`optim/adam`)
Generic Adam kernel. `step_scalars[3] = {lr_scale, bc1, bc2}` live on device for CUDA Graph compatibility. Optional `grad_scale[1]` for combined gradient normalization + clipping (also device-resident).

### Gradient Clipping (`optim/grad_clip`)
Fully on-device — eliminates D→H→D roundtrip:
1. `grad_norm_sq_kernel` — `block_reduce<SumOp>` per block, `atomicAdd` partial to global scalar
2. `compute_clip_scale_kernel` — single-thread: `scale = min(1/B, max_norm / (sqrt(norm_sq) / B))`
3. `grad_scale_kernel` — elementwise `g *= scale`

### LR Schedule (`optim/lr_schedule`)
`struct LRScheduler` (CPU-side). Linear warmup + cosine annealing + SGDR warm restarts.
`update_step_scalars_kernel` — single-thread device kernel computing `step_scalars[3]` to avoid H2D copy per step.

---

## Layer 5: GEMM (`forge::gemm`)

### FP32 Tiled (`gemm/fp32_tiled`)
All transpose combinations (NN, NT, TN, TT). Auto-selects tile config:
- Large K, small M: TM=16, TN=32, TK=64
- Large M: TM=64, TN=32, TK=32, 2x1 register tiling
- Default: TM=32, TN=32, TK=32

Split-K variant `fp32_gemm_splitk()` for better SM utilization when M is small.

### INT8 Dense Tensor Core (`gemm/int8_dense`)
`nvcuda::wmma` intrinsics. A: `(M,K)` row-major int8. B: `(N,K)` row-major int8. Output: `(M,N)` fp16.
Auto-selects 128x64 (narrow) or 128x128 (wide) tile. `int8_linear()` convenience: quantize + GEMM + dequant.

### INT8 Sparse Tensor Core (`gemm/int8_sparse`)
PTX `mma.sp` instruction. 2:4 structured sparsity (the hardware's peak sparse INT8 rate is 2x dense). Derived from [sparsemma](https://github.com/WizardsForgeGames/sparsemma).
Standard vs. warp-specialized variant. Split-K support.

---

## Layer 6: Quantization (`forge::quant`)

### Quantize (`quant/quantize`)
`fused_quantize_activation_kernel` — FP16 `(M,K)` → INT8 `(M,K)` + scales `(M,)`. One block per row. Two-pass: parallel max reduction → quantize.

### Dequantize (`quant/dequantize`)
`fused_dequantize_bias_kernel` — INT32 `(M,N)` → FP16 `(M,N)` with act_scale, w_scale, bias. One block per row.

---

## Layer 7: Sparsity (`forge::sparsity`)

### Prune + Compress (`sparsity/prune_compress`)
`prune_compress_kernel` — For each group of 4 along K, keeps 2 with largest magnitude. One thread per group.

### Pack Metadata (`sparsity/pack_metadata`)
`pack_metadata_kernel` — Raw metadata nibbles → hardware format uint32 for `mma.sp`.

---

## Layer 8: Physics Simulation (`forge::sim`)

**Note:** `sim/` headers are NOT pulled in by `forge.h` — must be included explicitly.

### Thermal Simulation (`sim/thermal`)
3D thermal conduction via explicit Euler finite-difference. Physics matches the reference HLSL conduction shader it was ported from.
Memory layout: `[B, G, G, G]` row-major. Precompute buffer: 9 arrays of `B*G^3` floats.
- `thermal_precompute_kernel` — Harmonic-mean conductivity, stability clamps
- `thermal_step_kernel` — One Euler step with 6-neighbor stencil
- `thermal_simulate()` — High-level: precompute once, then double-buffer loop

### Data Generation (`sim/thermal_datagen`)
GPU-accelerated training data generation replacing Python BatchGenerator.
6 material layout strategies × 6 temperature field strategies, all GPU-resident.
- `assign_layout_kernel` → `lookup_properties_kernel` → `generate_temperature_kernel` → thermal simulation
- `generate_training_batch()` — High-level end-to-end pipeline

---

## Python Integration (`forge-gpu` package)

PyTorch `CUDAExtension` via `setup.py`. Compiles all `src/**/*.cu` directly.

| Module | Contents |
|--------|----------|
| `forge.nn` | Autograd wrappers: linear, embedding, activation, softmax_ce |
| `forge.optim` | Adam optimizer wrapper |
| `forge.quant` | Observers, fake_quantize, convert (quantization pipeline) |
| `forge.compile` | torch.compile backend + pattern matching |

---

## Dependency Graph

```
forge.h (master include)
  ├── error.h, types.h, buffer.h
  │
  ├── primitives/  (zero-dependency leaf layer)
  │     activation, reduce, atomic, distance, geometry, sparse
  │
  ├── optim/
  │     adam        → cuda_utils
  │     grad_clip   → primitives/reduce, cuda_utils
  │     lr_schedule → (CPU + 1-thread GPU kernel)
  │
  ├── nn/
  │     fused_mlp             → primitives/activation, cuda_utils
  │     linear                → gemm/fp32_tiled
  │     mse                   → primitives/reduce, nn/reduce, cuda_utils
  │     softmax_ce            → (standalone)
  │     fused_gemm_softmax_ce → (standalone)
  │     reduce                → primitives/reduce, cuda_utils
  │     rng, trilinear, activations → cuda_utils
  │     embedding, bias       → (standalone)
  │     gating                → primitives/activation, primitives/reduce, cuda_utils
  │
  ├── quant/  (standalone)
  ├── gemm/
  │     fp32_tiled  → (standalone)
  │     int8_dense  → wmma intrinsics
  │     int8_sparse → PTX mma.sp, cp_async
  │
  ├── sparsity/  (standalone)
  │
  └── sim/  (NOT in forge.h — explicit include required)
        thermal         → cuda_utils
        thermal_datagen → sim/thermal, cuda_utils, <random>
```

---

## Complete Kernel Inventory (~42 kernels)

| Module | Kernel | Type |
|--------|--------|------|
| **nn/activations** | `leaky_relu_backward_kernel` | Elementwise |
| | `tanh_backward_kernel` | Elementwise |
| | `celu_backward_kernel` | Elementwise |
| **nn/fused_mlp** | `fused_mlp_forward_kernel` | Tiled (TILE_B=32) |
| | `fused_mlp_backward_kernel` | Tiled (BWD_BLOCK=128) |
| **nn/linear** | `add_bias_kernel` | Elementwise |
| **nn/mse** | `mse_forward_small_kernel` | `block_reduce<SumOp>` |
| | `squared_diff_kernel` | Elementwise |
| | `scale_scalar_kernel` | Single-thread |
| | `mse_backward_kernel` | Elementwise |
| **nn/reduce** | `reduce_partial_kernel<Op, Transform>` | `block_reduce<Op>` (templated) |
| | `reduce_final_kernel<Op>` | `block_reduce<Op>` (templated) |
| | `reduce_sum_dim0_kernel` | Per-column |
| | `reduce_mean_dim0_kernel` | Per-column |
| **nn/rng** | `fill_normal_kernel` | Elementwise |
| | `fill_uniform_kernel` | Elementwise |
| **nn/trilinear** | `trilinear_sample_kernel` | Per-element + interp |
| | `trilinear_scatter_kernel` | Per-element + atomicAdd |
| | `trilinear_sample_pos_grad_kernel` | Per-element + atomicAdd |
| | `trilinear_scatter_backward_kernel` | Per-element + atomicAdd |
| **nn/softmax_ce** | `softmax_ce_forward_kernel` | 1 block/sample |
| | `softmax_ce_backward_kernel` | Elementwise |
| | `softmax_ce_backward_bias_kernel` | 1 block/V-row |
| **nn/fused_gemm_softmax_ce** | (kernel in .cu) | Tiled GEMM + online softmax |
| **nn/embedding** | `embedding_gather_kernel` | Elementwise |
| | `embedding_scatter_kernel` | Elementwise + atomicAdd |
| **nn/bias** | `bias_grad_accumulate_kernel` | Per-row sum |
| **nn/gating** | `sigmoid_gate_forward_kernel` | Elementwise |
| | `sigmoid_gate_backward_kernel` | 1 block/unit, `block_reduce<SumOp>` |
| **optim/adam** | `adam_step_kernel` | Elementwise |
| **optim/grad_clip** | `grad_norm_sq_kernel` | `block_reduce<SumOp>` + atomicAdd |
| | `grad_scale_kernel` | Elementwise |
| | `compute_clip_scale_kernel` | Single-thread |
| **optim/lr_schedule** | `update_step_scalars_kernel` | Single-thread |
| **quant/quantize** | `fused_quantize_activation_kernel` | 1 block/row |
| **quant/dequantize** | `fused_dequantize_bias_kernel` | 1 block/row |
| **gemm/** | (templated tiled kernels) | Tiled GEMM |
| **sparsity/prune** | `prune_compress_kernel` | 1 thread/4 elements |
| **sparsity/pack** | `pack_metadata_kernel` | Elementwise |
| **sim/thermal** | `temperature_to_energy_kernel` | Elementwise |
| | `energy_to_temperature_kernel` | Elementwise |
| | `thermal_precompute_kernel` | Per-voxel |
| | `thermal_step_kernel` | Per-voxel (6-neighbor stencil) |
| **sim/datagen** | `assign_layout_kernel` | Per-voxel |
| | `lookup_properties_kernel` | Per-voxel |
| | `generate_temperature_kernel` | Per-voxel |
