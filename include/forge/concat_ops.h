// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Concatenation, stacking, and splitting operations on Tensors.
//
// cat/stack: join multiple tensors into one.
// split/chunk: split one tensor into multiple (zero-copy views).
// hstack/vstack: convenience wrappers.
#pragma once

#include "forge/tensor.h"
#include <vector>

namespace forge {

/// Concatenate tensors along an existing dimension.
/// All tensors must have the same shape except along dim.
Tensor cat(const std::vector<Tensor>& tensors, int dim = 0);

/// Stack tensors along a new dimension.
/// All tensors must have identical shape. Inserts a new dim at position dim.
Tensor stack(const std::vector<Tensor>& tensors, int dim = 0);

/// Horizontal stack: cat along dim=1 (or dim=0 for 1D tensors).
Tensor hstack(const std::vector<Tensor>& tensors);

/// Vertical stack: cat along dim=0.
Tensor vstack(const std::vector<Tensor>& tensors);

/// Split a tensor into chunks of split_size along dim.
/// Last chunk may be smaller.
std::vector<Tensor> split(const Tensor& tensor, int64_t split_size, int dim = 0);

/// Split a tensor into n_chunks roughly equal pieces along dim.
std::vector<Tensor> chunk(const Tensor& tensor, int n_chunks, int dim = 0);

} // namespace forge
