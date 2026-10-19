// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Elementwise math operations on Tensors.
//
// All operations return new Tensors (out-of-place). Tensor methods with
// trailing underscore (add_, mul_, etc.) are in-place variants that mutate
// the receiver and bump its version counter.
//
// Broadcasting follows NumPy rules: shapes are right-aligned, size-1
// dimensions expand via stride-0.
#pragma once

#include "forge/tensor.h"

namespace forge {

// ------------------------------------------------------------------
// Binary ops: tensor-tensor (with broadcasting)
// ------------------------------------------------------------------
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

// ------------------------------------------------------------------
// Binary ops: tensor-scalar
// ------------------------------------------------------------------
Tensor add(const Tensor& a, float scalar);
Tensor sub(const Tensor& a, float scalar);
Tensor mul(const Tensor& a, float scalar);
Tensor div(const Tensor& a, float scalar);

// ------------------------------------------------------------------
// Unary ops
// ------------------------------------------------------------------
Tensor neg(const Tensor& a);
Tensor abs(const Tensor& a);
Tensor sqrt(const Tensor& a);
Tensor rsqrt(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor square(const Tensor& a);
Tensor reciprocal(const Tensor& a);
Tensor sign(const Tensor& a);
Tensor sin(const Tensor& a);
Tensor cos(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor sigmoid(const Tensor& a);

// ------------------------------------------------------------------
// Utility ops
// ------------------------------------------------------------------
Tensor clamp(const Tensor& a, float min_val, float max_val);
Tensor pow(const Tensor& a, float exponent);

// ------------------------------------------------------------------
// Operator overloads
// ------------------------------------------------------------------

// tensor-tensor
inline Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
inline Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
inline Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }
inline Tensor operator/(const Tensor& a, const Tensor& b) { return div(a, b); }

// tensor-scalar
inline Tensor operator+(const Tensor& a, float s) { return add(a, s); }
inline Tensor operator-(const Tensor& a, float s) { return sub(a, s); }
inline Tensor operator*(const Tensor& a, float s) { return mul(a, s); }
inline Tensor operator/(const Tensor& a, float s) { return div(a, s); }

// scalar-tensor
inline Tensor operator+(float s, const Tensor& a) { return add(a, s); }
inline Tensor operator*(float s, const Tensor& a) { return mul(a, s); }
inline Tensor operator-(float s, const Tensor& a) { return add(neg(a), s); }
inline Tensor operator/(float s, const Tensor& a) { return mul(reciprocal(a), s); }

// unary minus
inline Tensor operator-(const Tensor& a) { return neg(a); }

// ------------------------------------------------------------------
// Broadcasting utility (exposed for use by other ops, e.g. comparison)
// ------------------------------------------------------------------
namespace detail {

/// Compute output shape from two input shapes using NumPy broadcasting rules.
/// Returns the output ndim. Aborts if shapes are not broadcastable.
int broadcast_shapes(const int64_t* a_sizes, int a_ndim,
                     const int64_t* b_sizes, int b_ndim,
                     int64_t* out_sizes);

} // namespace detail

} // namespace forge
