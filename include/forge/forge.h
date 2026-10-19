// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Master include -- pulls in the full Forge API.
#pragma once

// Core
#include "error.h"
#include "types.h"
#include "dtype.h"
#include "buffer.h"
#include "stream.h"
#include "allocator.h"
#include "storage.h"
#include "tensor.h"
#include "tensor_ops.h"
#include "comparison_ops.h"
#include "reduction_ops.h"
#include "indexing_ops.h"
#include "concat_ops.h"
#include "autograd.h"
#include "graph.h"

// Primitives (composable device functions)
#include "primitives/reduce.cuh"
#include "primitives/distance.cuh"
#include "primitives/activation.cuh"
#include "primitives/atomic.cuh"
#include "primitives/geometry.cuh"
#include "primitives/sparse.cuh"

// Optimizer
#include "optim/adam.cuh"
#include "optim/grad_clip.cuh"
#include "optim/lr_schedule.h"

// Neural network ops
#include "nn/activations.cuh"
#include "nn/softmax_ce.cuh"
#include "nn/embedding.cuh"
#include "nn/bias.cuh"
#include "nn/gating.cuh"
#include "nn/trilinear.cuh"
#include "nn/mse.cuh"
#include "nn/linear.cuh"
#include "nn/conv2d.cuh"
#include "nn/reduce.cuh"
#include "nn/rng.cuh"
#include "nn/fused_mlp.cuh"

// Quantization
#include "quant/quantize.cuh"
#include "quant/dequantize.cuh"

// GEMM
#include "gemm/int8_dense.cuh"
#include "gemm/int8_sparse.cuh"

// Sparsity
#include "sparsity/prune_compress.cuh"
#include "sparsity/pack_metadata.cuh"
