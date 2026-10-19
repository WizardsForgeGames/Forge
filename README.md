# Forge

A C++17/CUDA compute library for GPU training and inference, with an optional PyTorch extension.

## Status

Forge is a set of personal tools I built for my own projects (a game engine and assorted ML experiments), published for anyone who finds them useful. It is not regularly maintained: it gets updated when I pick it up again for another project. There are no guarantees of API stability or support; issues and pull requests are welcome, but responses may be slow.

## Features

**C++/CUDA library (`libforge`)**

| Area | Contents |
|------|----------|
| `forge::nn` | Fused MLP (whole network in one kernel, forward + backward), FP32 linear, conv2d, INT8 conv2d / depthwise conv2d / transposed conv2d, fused softmax + cross-entropy (incl. fused GEMM + softmax + CE), embedding gather/scatter, gating, MSE, reductions, Philox RNG, trilinear sampling |
| `forge::optim` | Adam with on-device step scalars (LR schedule, bias correction) and on-device gradient-norm clipping, so a training step needs no host round-trip and is CUDA Graph capturable; warmup + cosine LR schedule |
| `forge::gemm` | FP32 tiled GEMM (all transpose combos, split-K); INT8 GEMM: dense Tensor Core (wmma), 2:4 structured-sparse Tensor Core (PTX `mma.sp`), and `dp4a` for GPUs without INT8 Tensor Cores |
| `forge::quant` / `forge::sparsity` | Fused FP16 to INT8 per-token quantization, fused INT32 to FP16 dequantization + bias, 2:4 pruning/compression and hardware metadata packing |
| Tensor / autograd | `forge::Tensor` with caching allocator, views/strides, elementwise, comparison, reduction, indexing and concat ops, and a dynamic-graph reverse-mode autograd |
| `forge::sim` | 3D thermal conduction and Stefan-problem (solidification) solvers with GPU training-data generation (not included by `forge.h`) |

The sparse INT8 GEMM is derived from [sparsemma](https://github.com/WizardsForgeGames/sparsemma).

**Python package (`forge-gpu`, PyTorch extension)**

- `forge.nn`: autograd-wrapped modules (`Linear`, `FusedMLP`, `Conv2d`, activations, losses) and a `Sequential` that auto-fuses `[Linear, activation]*N + Linear` stacks into `FusedMLP`
- `forge.optim.Adam`: PyTorch wrapper over the on-device Adam kernel
- `forge.quant`: post-training quantization (`quantize`) and quantization-aware training (`prepare_qat` / `finalize_qat`) targeting the INT8 kernels
- `forge.compile`: a `torch.compile` backend (`backend="forge"`) that pattern-matches graphs onto Forge kernels
- `forge.training`: `Trainer` (training loop replayed as a CUDA Graph) and `ConvergencePredictor` (see [docs/training.md](docs/training.md))

See [docs/architecture.md](docs/architecture.md) for the full layout and kernel inventory.

## Build

C++ library:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# With tests
cmake -B build -DFORGE_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

On Windows: `cmake -B build -G "Visual Studio 17 2022" -A x64`.

The default CUDA architectures are `80;86;89`; override with `-DCMAKE_CUDA_ARCHITECTURES=...`.

Python extension (requires PyTorch with CUDA):

```bash
cd python
pip install -e .
```

**Requirements:** CMake 3.18+, CUDA Toolkit, C++17 compiler (developed with MSVC on Windows).

## Hardware tested

Developed and tested on an NVIDIA RTX 4090 (sm_89). The default build also targets sm_80 and sm_86 (Ampere), which are untested. Pre-Ampere GPUs are not supported: the sparse GEMM needs `cp.async` and `mma.sp`.

## Usage

```cpp
#include <forge/forge.h>

// Adam step with on-device gradient clipping (CUDA Graph capturable)
void train_step(float* param, float* grad, float* m, float* v,
                float* norm_sq, float* clip_scale, float* scalars,
                int n, int step, cudaStream_t s) {
    cudaMemsetAsync(norm_sq, 0, sizeof(float), s);
    forge::optim::grad_norm_sq(grad, n, norm_sq, s);
    forge::optim::compute_clip_scale(clip_scale, norm_sq, 1.0f, 1.0f/32, s);
    forge::optim::update_step_scalars(scalars, step, 100, 10000, 0.01f, 0.9f, 0.999f, s);
    forge::optim::adam_step(param, grad, m, v, n, 0.001f, scalars,
                            0.9f, 0.999f, 1e-8f, 0.01f, clip_scale,
                            false, 0, 0, s);
}
```

## License

MIT. See [LICENSE](LICENSE).
