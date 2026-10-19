// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Indexing and selection operations on Tensors.
//
// index_select: select slices along a dimension by Int64 index.
// gather/scatter: PyTorch-style element-wise indexing.
// masked_fill_/masked_select: Bool-mask-based operations.
#pragma once

#include "forge/tensor.h"

namespace forge {

/// Select slices along dim using an Int64 index tensor.
/// Output shape: input shape with sizes[dim] replaced by index.numel().
Tensor index_select(const Tensor& input, int dim, const Tensor& index);

/// Gather values along dim using an Int64 index tensor.
/// Output shape == index shape. index must have same ndim as input.
Tensor gather(const Tensor& input, int dim, const Tensor& index);

/// Scatter src values into a copy of input at positions given by index along dim.
/// Returns a new tensor (non-mutating).
Tensor scatter(const Tensor& input, int dim, const Tensor& index, const Tensor& src);

/// Scatter src values into self with atomicAdd accumulation (in-place).
void scatter_add_(Tensor& self, int dim, const Tensor& index, const Tensor& src);

/// Fill elements of self where mask is true (in-place). Mask broadcasts.
void masked_fill_(Tensor& self, const Tensor& mask, float value);

/// Select elements where mask is true, returning a 1D tensor.
Tensor masked_select(const Tensor& input, const Tensor& mask);

} // namespace forge
