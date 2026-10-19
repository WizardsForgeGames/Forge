// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Reduction operations on Tensors.
//
// Full-tensor reductions collapse all elements to a scalar Tensor {1}.
// Dimension-wise reductions reduce along a specific axis.
// max/min return {values, indices} pairs.
#pragma once

#include "forge/tensor.h"
#include <utility>

namespace forge {

// ------------------------------------------------------------------
// Full-tensor reductions (all elements -> scalar Tensor of shape {1})
// ------------------------------------------------------------------
Tensor sum(const Tensor& input);
Tensor mean(const Tensor& input);
Tensor prod(const Tensor& input);
Tensor amax(const Tensor& input);
Tensor amin(const Tensor& input);
Tensor norm(const Tensor& input, float p = 2.0f);

// ------------------------------------------------------------------
// Dimension-wise reductions
// ------------------------------------------------------------------
Tensor sum(const Tensor& input, int dim, bool keepdim = false);
Tensor mean(const Tensor& input, int dim, bool keepdim = false);
Tensor prod(const Tensor& input, int dim, bool keepdim = false);
Tensor amax(const Tensor& input, int dim, bool keepdim = false);
Tensor amin(const Tensor& input, int dim, bool keepdim = false);

// max/min with indices: returns {values, indices}
std::pair<Tensor, Tensor> max(const Tensor& input, int dim, bool keepdim = false);
std::pair<Tensor, Tensor> min(const Tensor& input, int dim, bool keepdim = false);

Tensor argmax(const Tensor& input, int dim);
Tensor argmin(const Tensor& input, int dim);

// ------------------------------------------------------------------
// Statistical reductions
// ------------------------------------------------------------------
Tensor var(const Tensor& input, int dim, bool keepdim = false, int correction = 1);
Tensor std_dev(const Tensor& input, int dim, bool keepdim = false, int correction = 1);
Tensor logsumexp(const Tensor& input, int dim, bool keepdim = false);

} // namespace forge
