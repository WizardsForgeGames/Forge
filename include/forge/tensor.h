// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Tensor: user-facing handle wrapping TensorImpl*.
//
// A Tensor is a lightweight 8-byte handle (single pointer) that manages
// the refcount of an underlying TensorImpl. Copy increments refcount;
// move transfers ownership with zero atomic operations.
//
// Key API:
//   .data<T>()     — explicit device pointer extraction (no implicit conversion)
//   .numel()       — total element count
//   .size(dim)     — dimension size
//   .stride(dim)   — dimension stride
//   .view(shape)   — zero-copy reshape (contiguous only)
//   .transpose()   — zero-copy dimension swap
//   .contiguous()  — returns self if already contiguous, copies otherwise
//
// Factory methods:
//   Tensor::empty(shape, dtype, stream)
//   Tensor::zeros(shape, dtype, stream)
//   Tensor::ones(shape, dtype, stream)
//   Tensor::from_ptr(ptr, shape, stream)     — non-owning wrap
//   Tensor::from_blob(ptr, shape, dtype)     — non-owning wrap (untyped)
#pragma once

#include <cstdint>
#include <initializer_list>
#include <vector>

#include "forge/dtype.h"
#include "forge/stream.h"
#include "forge/tensor_impl.h"

namespace forge {

class Tensor {
public:
    // ------------------------------------------------------------------
    // Constructors / assignment
    // ------------------------------------------------------------------

    Tensor() : impl_(nullptr) {}

    /// Move constructor — zero atomic operations (H4).
    Tensor(Tensor&& o) noexcept : impl_(o.impl_) { o.impl_ = nullptr; }

    /// Move assignment — zero atomic operations.
    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            reset();
            impl_ = o.impl_;
            o.impl_ = nullptr;
        }
        return *this;
    }

    /// Copy constructor — atomic increment.
    Tensor(const Tensor& o) : impl_(o.impl_) {
        if (impl_) impl_->retain();
    }

    /// Copy assignment — atomic increment + decrement.
    Tensor& operator=(const Tensor& o) {
        if (this != &o) {
            reset();
            impl_ = o.impl_;
            if (impl_) impl_->retain();
        }
        return *this;
    }

    ~Tensor() { reset(); }

    // ------------------------------------------------------------------
    // Factory methods
    // ------------------------------------------------------------------

    /// Allocate an uninitialized tensor with the given shape.
    static Tensor empty(std::initializer_list<int64_t> shape,
                        DType dtype = DType::Float32,
                        cudaStream_t stream = nullptr);

    static Tensor empty(const int64_t* shape, int ndim,
                        DType dtype = DType::Float32,
                        cudaStream_t stream = nullptr);

    /// Allocate a zero-filled tensor.
    static Tensor zeros(std::initializer_list<int64_t> shape,
                        DType dtype = DType::Float32,
                        cudaStream_t stream = nullptr);

    static Tensor zeros(const int64_t* shape, int ndim,
                        DType dtype = DType::Float32,
                        cudaStream_t stream = nullptr);

    /// Allocate a tensor filled with ones.
    static Tensor ones(std::initializer_list<int64_t> shape,
                       DType dtype = DType::Float32,
                       cudaStream_t stream = nullptr);

    static Tensor ones(const int64_t* shape, int ndim,
                       DType dtype = DType::Float32,
                       cudaStream_t stream = nullptr);

    /// Non-owning wrap of an existing typed device pointer.
    template <typename T>
    static Tensor from_ptr(T* device_ptr,
                           std::initializer_list<int64_t> shape);

    /// Non-owning wrap of an existing untyped device pointer.
    static Tensor from_blob(void* device_ptr,
                            std::initializer_list<int64_t> shape,
                            DType dtype = DType::Float32);

    static Tensor from_blob(void* device_ptr,
                            const int64_t* shape, int ndim,
                            DType dtype = DType::Float32);

    /// Fill all elements with a constant value (in-place).
    Tensor& fill_(float value);

    /// Zero all elements (in-place).
    Tensor& zero_();

    /// Copy data from src into this tensor (in-place).
    Tensor& copy_(const Tensor& src);

    /// Deep copy: new storage, same dtype and data.
    Tensor clone() const;

    /// Detach from autograd graph (returns tensor sharing storage).
    Tensor detach() const;

    /// Convert to a different dtype, returning a new tensor.
    /// If target == current dtype, returns a clone.
    Tensor to(DType target_dtype) const;

    /// Convenience type-cast methods.
    Tensor to_float()   const { return to(DType::Float32); }
    Tensor to_half()    const { return to(DType::Float16); }
    Tensor to_bfloat16() const { return to(DType::BFloat16); }
    Tensor to_int()     const { return to(DType::Int32); }
    Tensor to_long()    const { return to(DType::Int64); }
    Tensor to_bool()    const { return to(DType::Bool); }

    /// Match another tensor's dtype.
    Tensor type_as(const Tensor& other) const { return to(other.dtype()); }

    // ------------------------------------------------------------------
    // Data access — explicit, NO implicit conversion (C1)
    // ------------------------------------------------------------------

    /// Returns a typed device pointer. Tensor must be contiguous.
    template <typename T>
    T* data() const {
        return impl_->data_ptr<T>();
    }

    /// Returns an untyped device pointer (for generic code paths).
    void* data_ptr() const { return impl_ ? impl_->data_ptr_ : nullptr; }

    // ------------------------------------------------------------------
    // Metadata accessors
    // ------------------------------------------------------------------

    int64_t numel()        const { return impl_ ? impl_->numel_ : 0; }
    int     ndim()         const { return impl_ ? impl_->ndim_ : 0; }
    DType   dtype()        const { return impl_ ? impl_->dtype_ : DType::Float32; }
    bool    is_contiguous() const { return impl_ && impl_->is_contiguous(); }
    bool    defined()      const { return impl_ != nullptr; }

    int64_t size(int dim) const {
        if (dim < 0) dim += impl_->ndim_;
        return impl_->sizes_[dim];
    }

    int64_t stride(int dim) const {
        if (dim < 0) dim += impl_->ndim_;
        return impl_->strides_[dim];
    }

    const int64_t* sizes()   const { return impl_ ? impl_->sizes_ : nullptr; }
    const int64_t* strides() const { return impl_ ? impl_->strides_ : nullptr; }

    /// Total size in bytes of the underlying storage.
    size_t nbytes() const {
        return impl_ ? impl_->numel_ * dtype_size(impl_->dtype_) : 0;
    }

    // ------------------------------------------------------------------
    // Autograd
    // ------------------------------------------------------------------

    bool requires_grad() const {
        return impl_ && impl_->requires_grad();
    }

    Tensor& set_requires_grad_(bool val) {
        impl_->set_requires_grad(val);
        return *this;
    }

    /// Get the accumulated gradient (empty Tensor if none).
    Tensor grad() const;

    /// Run backward pass from this (scalar) tensor.
    void backward(const Tensor& grad_output = Tensor()) const;

    /// Keep gradient for non-leaf tensors (by default only leaf grads are retained).
    Tensor& retain_grad_();

    /// True if this tensor is a leaf (created by user, not by an op).
    bool is_leaf() const {
        if (!impl_) return true;
        return impl_->is_leaf();
    }

    /// True if this tensor has a grad_fn (was created by a differentiable op).
    bool has_grad_fn() const;

    // ------------------------------------------------------------------
    // Storage / version
    // ------------------------------------------------------------------

    Storage& storage() { return impl_->storage_; }
    const Storage& storage() const { return impl_->storage_; }

    uint32_t version() const {
        return impl_ ? impl_->storage_->version() : 0;
    }

    /// Bump the version counter (marks data as mutated).
    void bump_version() {
        if (impl_) impl_->storage_->bump_version();
    }

    int use_count() const {
        return impl_ ? impl_->refcount_.load(std::memory_order_relaxed) : 0;
    }

    // ------------------------------------------------------------------
    // View operations (zero-copy, metadata-only)
    // ------------------------------------------------------------------

    /// Reshape to a new shape (must be contiguous, same numel).
    Tensor view(std::initializer_list<int64_t> shape) const;
    Tensor view(const int64_t* shape, int ndim) const;

    /// Reshape (view if contiguous, copy otherwise).
    Tensor reshape(std::initializer_list<int64_t> shape) const;
    Tensor reshape(const int64_t* shape, int ndim) const;

    /// Swap two dimensions.
    Tensor transpose(int dim0, int dim1) const;

    /// Slice along a dimension: [start, start+length).
    Tensor slice(int dim, int64_t start, int64_t length) const;

    /// Insert a size-1 dimension at the given position.
    Tensor unsqueeze(int dim) const;

    /// Remove a size-1 dimension at the given position.
    Tensor squeeze(int dim) const;

    /// Expand size-1 dimensions to match a target shape (stride-0 broadcast).
    Tensor expand(std::initializer_list<int64_t> shape) const;
    Tensor expand(const int64_t* shape, int ndim) const;

    /// Returns self if contiguous, otherwise allocates + copies.
    Tensor contiguous() const;

    // ------------------------------------------------------------------
    // In-place arithmetic (defined in tensor_ops.cu)
    // ------------------------------------------------------------------

    Tensor& add_(const Tensor& other);
    Tensor& add_(float scalar);
    Tensor& sub_(const Tensor& other);
    Tensor& sub_(float scalar);
    Tensor& mul_(const Tensor& other);
    Tensor& mul_(float scalar);
    Tensor& div_(const Tensor& other);
    Tensor& div_(float scalar);
    Tensor& neg_();

    // ------------------------------------------------------------------
    // Internal
    // ------------------------------------------------------------------

    TensorImpl* unsafeGetTensorImpl() const { return impl_; }

private:
    /// Construct from a raw TensorImpl (takes ownership, does NOT retain).
    explicit Tensor(TensorImpl* impl) : impl_(impl) {}

    void reset() {
        if (impl_ && impl_->deref()) {
            impl_->release(); // return to pool
        }
        impl_ = nullptr;
    }

    TensorImpl* impl_;

    // Make a view TensorImpl sharing the same storage.
    TensorImpl* make_view_impl(int64_t storage_offset,
                               uint8_t ndim,
                               const int64_t* sizes,
                               const int64_t* strides) const;
};

} // namespace forge
