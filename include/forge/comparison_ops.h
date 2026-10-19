// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Comparison, logical, and conditional operations on Tensors.
//
// Comparison ops (eq, ne, lt, le, gt, ge) return Bool tensors.
// Logical ops (logical_and/or/xor/not) operate on Bool tensors.
// where(condition, x, y) selects elements from x or y based on a Bool mask.
#pragma once

#include "forge/tensor.h"

namespace forge {

// ------------------------------------------------------------------
// Element-wise comparison (float inputs -> Bool output)
// ------------------------------------------------------------------
Tensor eq(const Tensor& a, const Tensor& b);
Tensor ne(const Tensor& a, const Tensor& b);
Tensor lt(const Tensor& a, const Tensor& b);
Tensor le(const Tensor& a, const Tensor& b);
Tensor gt(const Tensor& a, const Tensor& b);
Tensor ge(const Tensor& a, const Tensor& b);

// Tensor-scalar comparison
Tensor eq(const Tensor& a, float scalar);
Tensor ne(const Tensor& a, float scalar);
Tensor lt(const Tensor& a, float scalar);
Tensor le(const Tensor& a, float scalar);
Tensor gt(const Tensor& a, float scalar);
Tensor ge(const Tensor& a, float scalar);

// ------------------------------------------------------------------
// Logical operations (Bool inputs -> Bool output)
// ------------------------------------------------------------------
Tensor logical_and(const Tensor& a, const Tensor& b);
Tensor logical_or(const Tensor& a, const Tensor& b);
Tensor logical_xor(const Tensor& a, const Tensor& b);
Tensor logical_not(const Tensor& a);

// ------------------------------------------------------------------
// Special value checks (float input -> Bool output)
// ------------------------------------------------------------------
Tensor isnan(const Tensor& a);
Tensor isinf(const Tensor& a);
Tensor isfinite(const Tensor& a);

// ------------------------------------------------------------------
// Conditional selection
// ------------------------------------------------------------------

/// Select elements from x or y based on a Bool condition mask.
/// Supports broadcasting across condition, x, and y.
Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y);

/// Approximate equality check (returns true if all elements are close).
bool allclose(const Tensor& a, const Tensor& b,
              float rtol = 1e-5f, float atol = 1e-8f);

// ------------------------------------------------------------------
// Whole-tensor reductions on Bool tensors
// ------------------------------------------------------------------

/// Returns true if any element is true (non-zero).
bool any(const Tensor& a);

/// Returns true if all elements are true (non-zero).
bool all(const Tensor& a);

} // namespace forge
