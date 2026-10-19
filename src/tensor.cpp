// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include "forge/tensor.h"
#include "forge/autograd.h"
#include "forge/allocator.h"
#include "forge/error.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace forge {

// ---------------------------------------------------------------------------
// AutogradMeta destructor (breaks circular include)
// ---------------------------------------------------------------------------

AutogradMeta::~AutogradMeta() {
    delete grad;
    grad = nullptr;
}

// ---------------------------------------------------------------------------
// Tensor kernel declarations (defined in tensor_kernels.cu)
// ---------------------------------------------------------------------------

namespace detail {
    void fill_kernel_dispatch(void* data, float value, int64_t numel,
                              DType dtype, cudaStream_t stream);
    void strided_copy_dispatch(const void* src, void* dst,
                               int ndim,
                               const int64_t* sizes,
                               const int64_t* src_strides,
                               const int64_t* dst_strides,
                               DType dtype, cudaStream_t stream);
    void cast_dispatch(const void* src, void* dst, int64_t numel,
                       DType src_dtype, DType dst_dtype, cudaStream_t stream);
} // namespace detail

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

/// Resolve default stream: use provided stream or fall back to getCurrentStream.
inline cudaStream_t resolve_stream(cudaStream_t stream) {
    return stream ? stream : getCurrentStream();
}

/// Infer dtype from C++ type.
template <typename T> DType infer_dtype();
template <> DType infer_dtype<float>()    { return DType::Float32; }
template <> DType infer_dtype<int32_t>()  { return DType::Int32; }
template <> DType infer_dtype<int64_t>()  { return DType::Int64; }
template <> DType infer_dtype<int8_t>()   { return DType::Int8; }
template <> DType infer_dtype<bool>()     { return DType::Bool; }

} // anonymous namespace

// ---------------------------------------------------------------------------
// Factory methods
// ---------------------------------------------------------------------------

Tensor Tensor::empty(std::initializer_list<int64_t> shape,
                     DType dtype, cudaStream_t stream) {
    return empty(shape.begin(), static_cast<int>(shape.size()), dtype, stream);
}

Tensor Tensor::empty(const int64_t* shape, int ndim,
                     DType dtype, cudaStream_t stream) {
    stream = resolve_stream(stream);
    int64_t numel = compute_numel(shape, ndim);
    size_t nbytes = numel * dtype_size(dtype);

    Storage storage(storage_create(nbytes, stream));
    TensorImpl* impl = TensorImpl::create(
        std::move(storage), /*storage_offset=*/0, dtype,
        static_cast<uint8_t>(ndim), shape, /*strides=*/nullptr);

    return Tensor(impl);
}

Tensor Tensor::zeros(std::initializer_list<int64_t> shape,
                     DType dtype, cudaStream_t stream) {
    return zeros(shape.begin(), static_cast<int>(shape.size()), dtype, stream);
}

Tensor Tensor::zeros(const int64_t* shape, int ndim,
                     DType dtype, cudaStream_t stream) {
    stream = resolve_stream(stream);
    Tensor t = empty(shape, ndim, dtype, stream);
    FORGE_CHECK(cudaMemsetAsync(t.data_ptr(), 0, t.nbytes(), stream));
    return t;
}

Tensor Tensor::ones(std::initializer_list<int64_t> shape,
                    DType dtype, cudaStream_t stream) {
    return ones(shape.begin(), static_cast<int>(shape.size()), dtype, stream);
}

Tensor Tensor::ones(const int64_t* shape, int ndim,
                    DType dtype, cudaStream_t stream) {
    stream = resolve_stream(stream);
    Tensor t = empty(shape, ndim, dtype, stream);
    detail::fill_kernel_dispatch(t.data_ptr(), 1.0f, t.numel(), dtype, stream);
    return t;
}

template <typename T>
Tensor Tensor::from_ptr(T* device_ptr, std::initializer_list<int64_t> shape) {
    DType dtype = infer_dtype<T>();
    int ndim = static_cast<int>(shape.size());
    const int64_t* sizes = shape.begin();
    int64_t numel = compute_numel(sizes, ndim);
    size_t nbytes = numel * dtype_size(dtype);

    int device = 0;
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, device_ptr) == cudaSuccess) {
        device = attr.device;
    } else {
        cudaGetLastError(); // clear error
        cudaGetDevice(&device);
    }

    Storage storage(storage_wrap(device_ptr, nbytes, device));
    TensorImpl* impl = TensorImpl::create(
        std::move(storage), 0, dtype,
        static_cast<uint8_t>(ndim), sizes, nullptr);

    return Tensor(impl);
}

// Explicit instantiations
template Tensor Tensor::from_ptr<float>(float*, std::initializer_list<int64_t>);
template Tensor Tensor::from_ptr<int32_t>(int32_t*, std::initializer_list<int64_t>);
template Tensor Tensor::from_ptr<int64_t>(int64_t*, std::initializer_list<int64_t>);
template Tensor Tensor::from_ptr<int8_t>(int8_t*, std::initializer_list<int64_t>);

Tensor Tensor::from_blob(void* device_ptr,
                         std::initializer_list<int64_t> shape, DType dtype) {
    return from_blob(device_ptr, shape.begin(),
                     static_cast<int>(shape.size()), dtype);
}

Tensor Tensor::from_blob(void* device_ptr,
                         const int64_t* shape, int ndim, DType dtype) {
    int64_t numel = compute_numel(shape, ndim);
    size_t nbytes = numel * dtype_size(dtype);

    int device = 0;
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, device_ptr) == cudaSuccess) {
        device = attr.device;
    } else {
        cudaGetLastError();
        cudaGetDevice(&device);
    }

    Storage storage(storage_wrap(device_ptr, nbytes, device));
    TensorImpl* impl = TensorImpl::create(
        std::move(storage), 0, dtype,
        static_cast<uint8_t>(ndim), shape, nullptr);

    return Tensor(impl);
}

Tensor& Tensor::fill_(float value) {
    cudaStream_t stream = getCurrentStream();
    detail::fill_kernel_dispatch(data_ptr(), value, numel(), dtype(), stream);
    bump_version();
    return *this;
}

Tensor& Tensor::zero_() {
    cudaStream_t stream = getCurrentStream();
    FORGE_CHECK(cudaMemsetAsync(data_ptr(), 0, nbytes(), stream));
    bump_version();
    return *this;
}

Tensor& Tensor::copy_(const Tensor& src) {
    cudaStream_t stream = getCurrentStream();
    if (is_contiguous() && src.is_contiguous() && numel() == src.numel()) {
        FORGE_CHECK(cudaMemcpyAsync(data_ptr(), src.data_ptr(),
                                    nbytes(), cudaMemcpyDeviceToDevice, stream));
    } else {
        detail::strided_copy_dispatch(
            src.data_ptr(), data_ptr(), ndim(),
            sizes(), src.strides(), strides(),
            dtype(), stream);
    }
    bump_version();
    return *this;
}

Tensor Tensor::clone() const {
    if (!impl_) return Tensor();
    cudaStream_t stream = getCurrentStream();
    Tensor x = contiguous();
    Tensor out = Tensor::empty(x.sizes(), x.ndim(), x.dtype(), stream);
    FORGE_CHECK(cudaMemcpyAsync(out.data_ptr(), x.data_ptr(),
                                x.nbytes(), cudaMemcpyDeviceToDevice, stream));
    return out;
}

Tensor Tensor::detach() const {
    if (!impl_) return Tensor();
    // Create a NEW TensorImpl sharing the same storage but with no autograd
    // connection. This avoids mutating the original tensor's metadata.
    Storage storage_copy = impl_->storage_;
    TensorImpl* new_impl = TensorImpl::create(
        std::move(storage_copy), impl_->storage_offset_, impl_->dtype_,
        impl_->ndim_, impl_->sizes_, impl_->strides_);
    // TensorImpl::create defaults: kIsLeaf set, no autograd_meta_, requires_grad=false
    return Tensor(new_impl);
}

Tensor Tensor::grad() const {
    if (!impl_ || !impl_->autograd_meta_ || !impl_->autograd_meta_->grad)
        return Tensor();
    return *(impl_->autograd_meta_->grad);
}

void Tensor::backward(const Tensor& grad_output) const {
    forge::backward(*this, grad_output);
}

Tensor& Tensor::retain_grad_() {
    if (impl_) {
        if (!impl_->autograd_meta_)
            impl_->autograd_meta_ = new AutogradMeta();
        impl_->autograd_meta_->retain_grad = true;
    }
    return *this;
}

bool Tensor::has_grad_fn() const {
    return impl_ && impl_->autograd_meta_ && impl_->autograd_meta_->grad_fn;
}

Tensor Tensor::to(DType target_dtype) const {
    if (!impl_) return Tensor();
    if (dtype() == target_dtype) return clone();

    cudaStream_t stream = getCurrentStream();
    Tensor x = contiguous();

    // Check if cast_dispatch handles this pair directly
    auto has_direct = [](DType s, DType d) -> bool {
        if (s == DType::Float32 || d == DType::Float32) return true;
        if (s == DType::Int32 && d == DType::Int64) return true;
        if (s == DType::Int64 && d == DType::Int32) return true;
        if (d == DType::Bool) return true;
        return false;
    };

    if (has_direct(x.dtype(), target_dtype)) {
        Tensor out = Tensor::empty(x.sizes(), x.ndim(), target_dtype, stream);
        detail::cast_dispatch(x.data_ptr(), out.data_ptr(), x.numel(),
                              x.dtype(), target_dtype, stream);
        return out;
    }

    // Two-step for non-Float32 → non-Float32: src → Float32 → target
    Tensor intermediate = x.to(DType::Float32);
    return intermediate.to(target_dtype);
}

// ---------------------------------------------------------------------------
// View operations
// ---------------------------------------------------------------------------

TensorImpl* Tensor::make_view_impl(int64_t storage_offset,
                                   uint8_t ndim,
                                   const int64_t* sizes,
                                   const int64_t* strides) const {
    Storage storage_copy = impl_->storage_; // share storage (refcount++)
    TensorImpl* view = TensorImpl::create(
        std::move(storage_copy), storage_offset, impl_->dtype_,
        ndim, sizes, strides);
    // Mark as view (not leaf)
    view->flags_ = (view->flags_ & ~kIsLeaf) | kIsView;
    return view;
}

Tensor Tensor::view(std::initializer_list<int64_t> shape) const {
    return view(shape.begin(), static_cast<int>(shape.size()));
}

Tensor Tensor::view(const int64_t* shape, int ndim) const {
    if (!is_contiguous()) {
        std::fprintf(stderr, "Forge: view() requires a contiguous tensor\n");
        std::abort();
    }
    int64_t new_numel = compute_numel(shape, ndim);
    if (new_numel != numel()) {
        std::fprintf(stderr, "Forge: view() shape numel %lld != tensor numel %lld\n",
                     (long long)new_numel, (long long)numel());
        std::abort();
    }

    int64_t new_strides[kMaxDim];
    compute_contiguous_strides(shape, ndim, new_strides);

    return Tensor(make_view_impl(impl_->storage_offset_,
                                 static_cast<uint8_t>(ndim),
                                 shape, new_strides));
}

Tensor Tensor::reshape(std::initializer_list<int64_t> shape) const {
    return reshape(shape.begin(), static_cast<int>(shape.size()));
}

Tensor Tensor::reshape(const int64_t* shape, int ndim) const {
    if (is_contiguous()) {
        return view(shape, ndim);
    }
    // Non-contiguous: copy to contiguous first, then view
    return contiguous().view(shape, ndim);
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    int nd = impl_->ndim_;
    if (dim0 < 0) dim0 += nd;
    if (dim1 < 0) dim1 += nd;

    int64_t new_sizes[kMaxDim];
    int64_t new_strides[kMaxDim];
    std::memcpy(new_sizes, impl_->sizes_, nd * sizeof(int64_t));
    std::memcpy(new_strides, impl_->strides_, nd * sizeof(int64_t));

    std::swap(new_sizes[dim0], new_sizes[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);

    return Tensor(make_view_impl(impl_->storage_offset_,
                                 static_cast<uint8_t>(nd),
                                 new_sizes, new_strides));
}

Tensor Tensor::slice(int dim, int64_t start, int64_t length) const {
    int nd = impl_->ndim_;
    if (dim < 0) dim += nd;

    int64_t new_sizes[kMaxDim];
    std::memcpy(new_sizes, impl_->sizes_, nd * sizeof(int64_t));
    new_sizes[dim] = length;

    int64_t new_offset = impl_->storage_offset_ + start * impl_->strides_[dim];

    return Tensor(make_view_impl(new_offset,
                                 static_cast<uint8_t>(nd),
                                 new_sizes, impl_->strides_));
}

Tensor Tensor::unsqueeze(int dim) const {
    int nd = impl_->ndim_;
    if (dim < 0) dim += nd + 1;
    int new_nd = nd + 1;

    if (new_nd > kMaxDim) {
        std::fprintf(stderr, "Forge: unsqueeze would exceed kMaxDim=%d\n", kMaxDim);
        std::abort();
    }

    int64_t new_sizes[kMaxDim];
    int64_t new_strides[kMaxDim];

    for (int i = 0; i < dim; ++i) {
        new_sizes[i] = impl_->sizes_[i];
        new_strides[i] = impl_->strides_[i];
    }
    new_sizes[dim] = 1;
    new_strides[dim] = (dim < nd) ? impl_->strides_[dim] * impl_->sizes_[dim] :
                       (nd > 0 ? impl_->strides_[nd - 1] : 1);
    for (int i = dim + 1; i < new_nd; ++i) {
        new_sizes[i] = impl_->sizes_[i - 1];
        new_strides[i] = impl_->strides_[i - 1];
    }

    return Tensor(make_view_impl(impl_->storage_offset_,
                                 static_cast<uint8_t>(new_nd),
                                 new_sizes, new_strides));
}

Tensor Tensor::squeeze(int dim) const {
    int nd = impl_->ndim_;
    if (dim < 0) dim += nd;

    if (impl_->sizes_[dim] != 1) {
        // Cannot squeeze a non-1 dimension, return a copy of self
        return *this;
    }

    int new_nd = nd - 1;
    if (new_nd < 0) new_nd = 0;

    int64_t new_sizes[kMaxDim];
    int64_t new_strides[kMaxDim];
    int j = 0;
    for (int i = 0; i < nd; ++i) {
        if (i == dim) continue;
        new_sizes[j] = impl_->sizes_[i];
        new_strides[j] = impl_->strides_[i];
        j++;
    }

    return Tensor(make_view_impl(impl_->storage_offset_,
                                 static_cast<uint8_t>(new_nd),
                                 new_sizes, new_strides));
}

Tensor Tensor::expand(std::initializer_list<int64_t> shape) const {
    return expand(shape.begin(), static_cast<int>(shape.size()));
}

Tensor Tensor::expand(const int64_t* shape, int ndim) const {
    int nd = impl_->ndim_;

    // Expand can increase dimensionality (prepend size-1 dims)
    if (ndim < nd) {
        std::fprintf(stderr, "Forge: expand() target ndim < source ndim\n");
        std::abort();
    }

    int64_t new_sizes[kMaxDim];
    int64_t new_strides[kMaxDim];

    // Right-align: extra leading dims get stride 0
    int offset = ndim - nd;
    for (int i = 0; i < offset; ++i) {
        new_sizes[i] = shape[i];
        new_strides[i] = 0; // broadcast
    }

    for (int i = 0; i < nd; ++i) {
        int64_t src_size = impl_->sizes_[i];
        int64_t tgt_size = shape[offset + i];

        if (src_size == tgt_size) {
            new_sizes[offset + i] = src_size;
            new_strides[offset + i] = impl_->strides_[i];
        } else if (src_size == 1) {
            new_sizes[offset + i] = tgt_size;
            new_strides[offset + i] = 0; // broadcast via stride-0
        } else {
            std::fprintf(stderr,
                "Forge: expand() cannot expand dim %d from %lld to %lld\n",
                i, (long long)src_size, (long long)tgt_size);
            std::abort();
        }
    }

    return Tensor(make_view_impl(impl_->storage_offset_,
                                 static_cast<uint8_t>(ndim),
                                 new_sizes, new_strides));
}

// ---------------------------------------------------------------------------
// contiguous()
// ---------------------------------------------------------------------------

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this; // zero-cost check

    cudaStream_t stream = getCurrentStream();
    int nd = impl_->ndim_;
    Tensor dst = Tensor::empty(impl_->sizes_, nd, impl_->dtype_, stream);

    detail::strided_copy_dispatch(
        impl_->data_ptr_,
        dst.data_ptr(),
        nd,
        impl_->sizes_,
        impl_->strides_,
        dst.impl_->strides_,
        impl_->dtype_,
        stream);

    return dst;
}

} // namespace forge
