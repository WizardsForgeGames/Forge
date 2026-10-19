// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Elementwise math operations: kernels, functor structs, dispatch functions.
//
// Architecture:
//   - Functor structs define the math (AddOp, ExpOp, etc.)
//   - Template kernels apply any functor over tensor elements
//   - Dispatch helpers choose contiguous fast-path or strided general-path
//   - Public free functions instantiate the templates with specific functors
#include "forge/tensor_ops.h"
#include "forge/autograd.h"
#include "forge/error.h"
#include "internal/cuda_utils.cuh"

#include <cstdio>
#include <cstring>
#include <algorithm>

// Forward declarations for autograd recording functions (defined in autograd_ops.cu)
namespace forge {
void record_add(Tensor& out, const Tensor& a, const Tensor& b);
void record_sub(Tensor& out, const Tensor& a, const Tensor& b);
void record_mul(Tensor& out, const Tensor& a, const Tensor& b);
void record_div(Tensor& out, const Tensor& a, const Tensor& b);
void record_add_scalar(Tensor& out, const Tensor& a);
void record_sub_scalar(Tensor& out, const Tensor& a);
void record_mul_scalar(Tensor& out, const Tensor& a, float s);
void record_div_scalar(Tensor& out, const Tensor& a, float s);
void record_neg(Tensor& out, const Tensor& a);
void record_exp(Tensor& out, const Tensor& a);
void record_log(Tensor& out, const Tensor& a);
void record_sqrt(Tensor& out, const Tensor& a);
void record_rsqrt(Tensor& out, const Tensor& a);
void record_square(Tensor& out, const Tensor& a);
void record_reciprocal(Tensor& out, const Tensor& a);
void record_abs(Tensor& out, const Tensor& a);
void record_sin(Tensor& out, const Tensor& a);
void record_cos(Tensor& out, const Tensor& a);
void record_tanh(Tensor& out, const Tensor& a);
void record_sigmoid(Tensor& out, const Tensor& a);
void record_clamp(Tensor& out, const Tensor& a, float lo, float hi);
void record_pow(Tensor& out, const Tensor& a, float exponent);
} // namespace forge

namespace forge {

// =====================================================================
// Broadcasting utilities
// =====================================================================

namespace detail {

int broadcast_shapes(const int64_t* a_sizes, int a_ndim,
                     const int64_t* b_sizes, int b_ndim,
                     int64_t* out_sizes) {
    int out_ndim = std::max(a_ndim, b_ndim);
    int a_off = out_ndim - a_ndim;
    int b_off = out_ndim - b_ndim;

    for (int i = 0; i < out_ndim; ++i) {
        int64_t da = (i >= a_off) ? a_sizes[i - a_off] : 1;
        int64_t db = (i >= b_off) ? b_sizes[i - b_off] : 1;

        if (da == db) {
            out_sizes[i] = da;
        } else if (da == 1) {
            out_sizes[i] = db;
        } else if (db == 1) {
            out_sizes[i] = da;
        } else {
            std::fprintf(stderr,
                "Forge: broadcast error: dim %d sizes %lld vs %lld\n",
                i, (long long)da, (long long)db);
            std::abort();
        }
    }
    return out_ndim;
}

} // namespace detail

// =====================================================================
// Internal helpers
// =====================================================================

namespace {

/// Right-align input strides to output ndim, stride=0 for broadcast dims.
inline void expand_strides(const int64_t* in_sizes, const int64_t* in_strides,
                           int in_ndim, const int64_t* out_sizes, int out_ndim,
                           int64_t* expanded) {
    int off = out_ndim - in_ndim;
    for (int i = 0; i < off; ++i)
        expanded[i] = 0;
    for (int i = 0; i < in_ndim; ++i)
        expanded[off + i] = (in_sizes[i] == out_sizes[off + i]) ? in_strides[i] : 0;
}

inline cudaStream_t resolve_stream(cudaStream_t stream) {
    return stream ? stream : getCurrentStream();
}

} // anonymous namespace

// =====================================================================
// Functor structs
// =====================================================================

// ----- Binary -----
struct AddOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
};
struct SubOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return a - b; }
};
struct MulOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return a * b; }
};
struct DivOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return a / b; }
};

// ----- Unary -----
struct NegOp {
    __device__ __forceinline__ float operator()(float x) const { return -x; }
};
struct AbsOp {
    __device__ __forceinline__ float operator()(float x) const { return fabsf(x); }
};
struct SqrtOp {
    __device__ __forceinline__ float operator()(float x) const { return sqrtf(x); }
};
struct RsqrtOp {
    __device__ __forceinline__ float operator()(float x) const { return rsqrtf(x); }
};
struct ExpOp {
    __device__ __forceinline__ float operator()(float x) const { return expf(x); }
};
struct LogOp {
    __device__ __forceinline__ float operator()(float x) const { return logf(x); }
};
struct SquareOp {
    __device__ __forceinline__ float operator()(float x) const { return x * x; }
};
struct ReciprocalOp {
    __device__ __forceinline__ float operator()(float x) const { return 1.0f / x; }
};
struct SignOp {
    __device__ __forceinline__ float operator()(float x) const {
        return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
    }
};
struct SinOp {
    __device__ __forceinline__ float operator()(float x) const { return sinf(x); }
};
struct CosOp {
    __device__ __forceinline__ float operator()(float x) const { return cosf(x); }
};
struct TanhOp {
    __device__ __forceinline__ float operator()(float x) const { return tanhf(x); }
};
struct SigmoidOp {
    __device__ __forceinline__ float operator()(float x) const {
        return 1.0f / (1.0f + expf(-x));
    }
};

// ----- Parameterized -----
struct ClampOp {
    float lo, hi;
    __device__ __forceinline__ float operator()(float x) const {
        return fminf(fmaxf(x, lo), hi);
    }
};
struct PowOp {
    float exponent;
    __device__ __forceinline__ float operator()(float x) const {
        return powf(x, exponent);
    }
};

// ----- Scalar wrappers (wraps a BinaryOp into a UnaryOp with captured scalar) -----
// Must be at namespace scope for NVCC kernel template compatibility.
template <typename BinOp>
struct ScalarWrap {
    BinOp base_op;
    float s;
    __device__ __forceinline__ float operator()(float x) const {
        return base_op(x, s);
    }
};

// =====================================================================
// Kernel templates
// =====================================================================

// ----- Contiguous fast-path -----

template <typename Op>
__global__ void unary_contiguous_kernel(
    const float* __restrict__ in,
    float* __restrict__       out,
    int64_t                   numel,
    Op                        op)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) out[idx] = op(in[idx]);
}

template <typename Op>
__global__ void binary_contiguous_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__       out,
    int64_t                   numel,
    Op                        op)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) out[idx] = op(a[idx], b[idx]);
}

template <typename Op>
__global__ void binary_scalar_kernel(
    const float* __restrict__ a,
    float                     scalar,
    float* __restrict__       out,
    int64_t                   numel,
    Op                        op)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) out[idx] = op(a[idx], scalar);
}

// ----- Strided general-path -----

template <typename Op>
__global__ void unary_strided_kernel(
    const float* __restrict__ in,
    float* __restrict__       out,
    int                       ndim,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ in_strides,
    const int64_t* __restrict__ out_strides,
    int64_t                   numel,
    Op                        op)
{
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= numel) return;

    int64_t remaining = linear;
    int64_t in_off = 0, out_off = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        int64_t coord = remaining % sizes[d];
        remaining /= sizes[d];
        in_off  += coord * in_strides[d];
        out_off += coord * out_strides[d];
    }
    out[out_off] = op(in[in_off]);
}

template <typename Op>
__global__ void binary_strided_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__       out,
    int                       ndim,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ a_strides,
    const int64_t* __restrict__ b_strides,
    const int64_t* __restrict__ out_strides,
    int64_t                   numel,
    Op                        op)
{
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= numel) return;

    int64_t remaining = linear;
    int64_t a_off = 0, b_off = 0, out_off = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        int64_t coord = remaining % sizes[d];
        remaining /= sizes[d];
        a_off   += coord * a_strides[d];
        b_off   += coord * b_strides[d];
        out_off += coord * out_strides[d];
    }
    out[out_off] = op(a[a_off], b[b_off]);
}

// In-place strided binary: writes result back to a[] at a_off
template <typename Op>
__global__ void binary_strided_inplace_kernel(
    float* __restrict__       a,
    const float* __restrict__ b,
    int                       ndim,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ a_strides,
    const int64_t* __restrict__ b_strides,
    int64_t                   numel,
    Op                        op)
{
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= numel) return;

    int64_t remaining = linear;
    int64_t a_off = 0, b_off = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        int64_t coord = remaining % sizes[d];
        remaining /= sizes[d];
        a_off += coord * a_strides[d];
        b_off += coord * b_strides[d];
    }
    a[a_off] = op(a[a_off], b[b_off]);
}

// In-place strided unary
template <typename Op>
__global__ void unary_strided_inplace_kernel(
    float* __restrict__       data,
    int                       ndim,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ strides,
    int64_t                   numel,
    Op                        op)
{
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= numel) return;

    int64_t remaining = linear;
    int64_t off = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        int64_t coord = remaining % sizes[d];
        remaining /= sizes[d];
        off += coord * strides[d];
    }
    data[off] = op(data[off]);
}

// =====================================================================
// Dispatch helpers (template, internal)
// =====================================================================

template <typename Op>
static Tensor unary_op_impl(const Tensor& input, Op op) {
    cudaStream_t stream = resolve_stream(nullptr);
    int64_t numel = input.numel();
    if (numel == 0) return Tensor::empty(input.sizes(), input.ndim(), input.dtype(), stream);

    Tensor out = Tensor::empty(input.sizes(), input.ndim(), input.dtype(), stream);

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(numel), block);

    if (input.is_contiguous()) {
        unary_contiguous_kernel<<<grid, block, 0, stream>>>(
            input.data<float>(), out.data<float>(), numel, op);
    } else {
        int nd = input.ndim();
        int64_t* d_meta = nullptr;
        size_t meta_bytes = 3 * nd * sizeof(int64_t);
        FORGE_CHECK(cudaMallocAsync(&d_meta, meta_bytes, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta,          input.sizes(),   nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + nd,     input.strides(), nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + 2 * nd, out.strides(),   nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        unary_strided_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(input.data_ptr()),
            out.data<float>(), nd,
            d_meta, d_meta + nd, d_meta + 2 * nd, numel, op);

        FORGE_CHECK(cudaFreeAsync(d_meta, stream));
    }
    return out;
}

template <typename Op>
static Tensor binary_op_impl(const Tensor& a, const Tensor& b, Op op) {
    cudaStream_t stream = resolve_stream(nullptr);

    // Compute broadcast output shape
    int64_t out_sizes[kMaxDim];
    int out_ndim = detail::broadcast_shapes(
        a.sizes(), a.ndim(), b.sizes(), b.ndim(), out_sizes);

    int64_t out_numel = compute_numel(out_sizes, out_ndim);
    if (out_numel == 0)
        return Tensor::empty(out_sizes, out_ndim, a.dtype(), stream);

    Tensor out = Tensor::empty(out_sizes, out_ndim, a.dtype(), stream);

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(out_numel), block);

    // Check if both are contiguous AND same shape (no broadcast needed)
    bool same_shape = (a.ndim() == b.ndim());
    if (same_shape) {
        for (int i = 0; i < a.ndim(); ++i) {
            if (a.size(i) != b.size(i)) { same_shape = false; break; }
        }
    }

    if (same_shape && a.is_contiguous() && b.is_contiguous()) {
        binary_contiguous_kernel<<<grid, block, 0, stream>>>(
            a.data<float>(), b.data<float>(), out.data<float>(),
            out_numel, op);
    } else {
        // General strided path with broadcasting
        int64_t a_strides_exp[kMaxDim], b_strides_exp[kMaxDim];
        expand_strides(a.sizes(), a.strides(), a.ndim(),
                       out_sizes, out_ndim, a_strides_exp);
        expand_strides(b.sizes(), b.strides(), b.ndim(),
                       out_sizes, out_ndim, b_strides_exp);

        // Upload: sizes + a_strides + b_strides + out_strides
        int64_t* d_meta = nullptr;
        size_t meta_bytes = 4 * out_ndim * sizeof(int64_t);
        FORGE_CHECK(cudaMallocAsync(&d_meta, meta_bytes, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta,                out_sizes,       out_ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + out_ndim,     a_strides_exp,   out_ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + 2 * out_ndim, b_strides_exp,   out_ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + 3 * out_ndim, out.strides(),   out_ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        binary_strided_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(a.data_ptr()),
            static_cast<const float*>(b.data_ptr()),
            out.data<float>(), out_ndim,
            d_meta, d_meta + out_ndim, d_meta + 2 * out_ndim,
            d_meta + 3 * out_ndim, out_numel, op);

        FORGE_CHECK(cudaFreeAsync(d_meta, stream));
    }
    return out;
}

template <typename Op>
static Tensor binary_scalar_op_impl(const Tensor& a, float scalar, Op op) {
    cudaStream_t stream = resolve_stream(nullptr);
    int64_t numel = a.numel();
    if (numel == 0) return Tensor::empty(a.sizes(), a.ndim(), a.dtype(), stream);

    Tensor out = Tensor::empty(a.sizes(), a.ndim(), a.dtype(), stream);

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(numel), block);

    if (a.is_contiguous()) {
        binary_scalar_kernel<<<grid, block, 0, stream>>>(
            a.data<float>(), scalar, out.data<float>(), numel, op);
    } else {
        int nd = a.ndim();
        int64_t* d_meta = nullptr;
        size_t meta_bytes = 3 * nd * sizeof(int64_t);
        FORGE_CHECK(cudaMallocAsync(&d_meta, meta_bytes, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta,          a.sizes(),   nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + nd,     a.strides(), nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + 2 * nd, out.strides(), nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        unary_strided_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(a.data_ptr()),
            out.data<float>(), nd,
            d_meta, d_meta + nd, d_meta + 2 * nd,
            numel, ScalarWrap<Op>{op, scalar});

        FORGE_CHECK(cudaFreeAsync(d_meta, stream));
    }
    return out;
}

// ----- In-place dispatch -----

template <typename Op>
static void unary_inplace_impl(Tensor& self, Op op) {
    cudaStream_t stream = resolve_stream(nullptr);
    int64_t numel = self.numel();
    if (numel == 0) return;

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(numel), block);

    if (self.is_contiguous()) {
        float* ptr = self.data<float>();
        unary_contiguous_kernel<<<grid, block, 0, stream>>>(ptr, ptr, numel, op);
    } else {
        int nd = self.ndim();
        int64_t* d_meta = nullptr;
        size_t meta_bytes = 2 * nd * sizeof(int64_t);
        FORGE_CHECK(cudaMallocAsync(&d_meta, meta_bytes, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta,      self.sizes(),   nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + nd, self.strides(), nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        unary_strided_inplace_kernel<<<grid, block, 0, stream>>>(
            static_cast<float*>(self.data_ptr()), nd,
            d_meta, d_meta + nd, numel, op);

        FORGE_CHECK(cudaFreeAsync(d_meta, stream));
    }
    self.bump_version();
}

template <typename Op>
static void binary_inplace_impl(Tensor& self, const Tensor& other, Op op) {
    cudaStream_t stream = resolve_stream(nullptr);
    int64_t numel = self.numel();
    if (numel == 0) return;

    // Verify broadcast of other doesn't require expanding self
    int64_t out_sizes[kMaxDim];
    int out_ndim = detail::broadcast_shapes(
        self.sizes(), self.ndim(), other.sizes(), other.ndim(), out_sizes);

    // Self shape must match the broadcast output shape
    bool shape_ok = (out_ndim == self.ndim());
    if (shape_ok) {
        for (int i = 0; i < out_ndim; ++i) {
            if (out_sizes[i] != self.size(i)) { shape_ok = false; break; }
        }
    }
    if (!shape_ok) {
        std::fprintf(stderr,
            "Forge: in-place op requires output shape to match self shape\n");
        std::abort();
    }

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(numel), block);

    // Check if same shape and both contiguous
    bool same_shape = (self.ndim() == other.ndim());
    if (same_shape) {
        for (int i = 0; i < self.ndim(); ++i) {
            if (self.size(i) != other.size(i)) { same_shape = false; break; }
        }
    }

    if (same_shape && self.is_contiguous() && other.is_contiguous()) {
        float* self_ptr = self.data<float>();
        binary_contiguous_kernel<<<grid, block, 0, stream>>>(
            self_ptr, other.data<float>(), self_ptr, numel, op);
    } else {
        // Strided in-place with broadcasting
        int nd = self.ndim();
        int64_t b_strides_exp[kMaxDim];
        expand_strides(other.sizes(), other.strides(), other.ndim(),
                       self.sizes(), nd, b_strides_exp);

        int64_t* d_meta = nullptr;
        size_t meta_bytes = 3 * nd * sizeof(int64_t);
        FORGE_CHECK(cudaMallocAsync(&d_meta, meta_bytes, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta,          self.sizes(),   nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + nd,     self.strides(), nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + 2 * nd, b_strides_exp, nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        binary_strided_inplace_kernel<<<grid, block, 0, stream>>>(
            static_cast<float*>(self.data_ptr()),
            static_cast<const float*>(other.data_ptr()),
            nd, d_meta, d_meta + nd, d_meta + 2 * nd, numel, op);

        FORGE_CHECK(cudaFreeAsync(d_meta, stream));
    }
    self.bump_version();
}

template <typename Op>
static void binary_scalar_inplace_impl(Tensor& self, float scalar, Op op) {
    cudaStream_t stream = resolve_stream(nullptr);
    int64_t numel = self.numel();
    if (numel == 0) return;

    constexpr int block = internal::kDefaultBlockSize;
    int grid = internal::divup(static_cast<int>(numel), block);

    if (self.is_contiguous()) {
        float* ptr = self.data<float>();
        binary_scalar_kernel<<<grid, block, 0, stream>>>(ptr, scalar, ptr, numel, op);
    } else {
        int nd = self.ndim();
        int64_t* d_meta = nullptr;
        size_t meta_bytes = 2 * nd * sizeof(int64_t);
        FORGE_CHECK(cudaMallocAsync(&d_meta, meta_bytes, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta,      self.sizes(),   nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        FORGE_CHECK(cudaMemcpyAsync(d_meta + nd, self.strides(), nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        unary_strided_inplace_kernel<<<grid, block, 0, stream>>>(
            static_cast<float*>(self.data_ptr()), nd,
            d_meta, d_meta + nd, numel, ScalarWrap<Op>{op, scalar});

        FORGE_CHECK(cudaFreeAsync(d_meta, stream));
    }
    self.bump_version();
}

// =====================================================================
// Public free functions
// =====================================================================

// ----- Binary tensor-tensor -----
Tensor add(const Tensor& a, const Tensor& b) {
    Tensor out = binary_op_impl(a, b, AddOp{});
    if (should_record(a, b)) record_add(out, a, b);
    return out;
}
Tensor sub(const Tensor& a, const Tensor& b) {
    Tensor out = binary_op_impl(a, b, SubOp{});
    if (should_record(a, b)) record_sub(out, a, b);
    return out;
}
Tensor mul(const Tensor& a, const Tensor& b) {
    Tensor out = binary_op_impl(a, b, MulOp{});
    if (should_record(a, b)) record_mul(out, a, b);
    return out;
}
Tensor div(const Tensor& a, const Tensor& b) {
    Tensor out = binary_op_impl(a, b, DivOp{});
    if (should_record(a, b)) record_div(out, a, b);
    return out;
}

// ----- Binary tensor-scalar -----
Tensor add(const Tensor& a, float s) {
    Tensor out = binary_scalar_op_impl(a, s, AddOp{});
    if (should_record(a)) record_add_scalar(out, a);
    return out;
}
Tensor sub(const Tensor& a, float s) {
    Tensor out = binary_scalar_op_impl(a, s, SubOp{});
    if (should_record(a)) record_sub_scalar(out, a);
    return out;
}
Tensor mul(const Tensor& a, float s) {
    Tensor out = binary_scalar_op_impl(a, s, MulOp{});
    if (should_record(a)) record_mul_scalar(out, a, s);
    return out;
}
Tensor div(const Tensor& a, float s) {
    Tensor out = binary_scalar_op_impl(a, s, DivOp{});
    if (should_record(a)) record_div_scalar(out, a, s);
    return out;
}

// ----- Unary -----
Tensor neg(const Tensor& a) {
    Tensor out = unary_op_impl(a, NegOp{});
    if (should_record(a)) record_neg(out, a);
    return out;
}
Tensor abs(const Tensor& a) {
    Tensor out = unary_op_impl(a, AbsOp{});
    if (should_record(a)) record_abs(out, a);
    return out;
}
Tensor sqrt(const Tensor& a) {
    Tensor out = unary_op_impl(a, SqrtOp{});
    if (should_record(a)) record_sqrt(out, a);
    return out;
}
Tensor rsqrt(const Tensor& a) {
    Tensor out = unary_op_impl(a, RsqrtOp{});
    if (should_record(a)) record_rsqrt(out, a);
    return out;
}
Tensor exp(const Tensor& a) {
    Tensor out = unary_op_impl(a, ExpOp{});
    if (should_record(a)) record_exp(out, a);
    return out;
}
Tensor log(const Tensor& a) {
    Tensor out = unary_op_impl(a, LogOp{});
    if (should_record(a)) record_log(out, a);
    return out;
}
Tensor square(const Tensor& a) {
    Tensor out = unary_op_impl(a, SquareOp{});
    if (should_record(a)) record_square(out, a);
    return out;
}
Tensor reciprocal(const Tensor& a) {
    Tensor out = unary_op_impl(a, ReciprocalOp{});
    if (should_record(a)) record_reciprocal(out, a);
    return out;
}
Tensor sign(const Tensor& a) {
    // sign is not differentiable — no graph recording
    return unary_op_impl(a, SignOp{});
}
Tensor sin(const Tensor& a) {
    Tensor out = unary_op_impl(a, SinOp{});
    if (should_record(a)) record_sin(out, a);
    return out;
}
Tensor cos(const Tensor& a) {
    Tensor out = unary_op_impl(a, CosOp{});
    if (should_record(a)) record_cos(out, a);
    return out;
}
Tensor tanh(const Tensor& a) {
    Tensor out = unary_op_impl(a, TanhOp{});
    if (should_record(a)) record_tanh(out, a);
    return out;
}
Tensor sigmoid(const Tensor& a) {
    Tensor out = unary_op_impl(a, SigmoidOp{});
    if (should_record(a)) record_sigmoid(out, a);
    return out;
}

// ----- Utility -----
Tensor clamp(const Tensor& a, float min_val, float max_val) {
    Tensor out = unary_op_impl(a, ClampOp{min_val, max_val});
    if (should_record(a)) record_clamp(out, a, min_val, max_val);
    return out;
}
Tensor pow(const Tensor& a, float exponent) {
    Tensor out = unary_op_impl(a, PowOp{exponent});
    if (should_record(a)) record_pow(out, a, exponent);
    return out;
}

// =====================================================================
// In-place Tensor methods
// =====================================================================

Tensor& Tensor::add_(const Tensor& other) { binary_inplace_impl(*this, other, AddOp{}); return *this; }
Tensor& Tensor::add_(float s)             { binary_scalar_inplace_impl(*this, s, AddOp{}); return *this; }
Tensor& Tensor::sub_(const Tensor& other) { binary_inplace_impl(*this, other, SubOp{}); return *this; }
Tensor& Tensor::sub_(float s)             { binary_scalar_inplace_impl(*this, s, SubOp{}); return *this; }
Tensor& Tensor::mul_(const Tensor& other) { binary_inplace_impl(*this, other, MulOp{}); return *this; }
Tensor& Tensor::mul_(float s)             { binary_scalar_inplace_impl(*this, s, MulOp{}); return *this; }
Tensor& Tensor::div_(const Tensor& other) { binary_inplace_impl(*this, other, DivOp{}); return *this; }
Tensor& Tensor::div_(float s)             { binary_scalar_inplace_impl(*this, s, DivOp{}); return *this; }
Tensor& Tensor::neg_()                    { unary_inplace_impl(*this, NegOp{}); return *this; }

} // namespace forge
