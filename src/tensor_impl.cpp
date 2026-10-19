// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include "forge/tensor_impl.h"
#include "forge/error.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace forge {

// ---------------------------------------------------------------------------
// Thread-local object pool (H3)
// ---------------------------------------------------------------------------

namespace {

constexpr size_t kPoolHighWater = 64;
thread_local std::vector<TensorImpl*> tls_pool;

} // anonymous namespace

TensorImpl* TensorImpl::create(Storage storage,
                               int64_t storage_offset,
                               DType dtype,
                               uint8_t ndim,
                               const int64_t* sizes,
                               const int64_t* strides) {
    TensorImpl* p;
    if (!tls_pool.empty()) {
        p = tls_pool.back();
        tls_pool.pop_back();
    } else {
        // Construct the object (storage_ must be a valid null Storage before init()
        // move-assigns into it, or the assignment releases a garbage pointer).
        p = new (::operator new(sizeof(TensorImpl))) TensorImpl;
    }
    p->init(std::move(storage), storage_offset, dtype, ndim, sizes, strides);
    return p;
}

void TensorImpl::release() {
    // Clean up without calling ~TensorImpl(), which would destroy storage_
    // and leave a dangling impl_ pointer if the object is recycled via pool.
    // Instead, release storage via assignment (leaves storage_ in valid null state)
    // and manually delete autograd_meta_.
    delete autograd_meta_;
    autograd_meta_ = nullptr;
    storage_ = Storage(); // release refcount, then set to null

    if (tls_pool.size() < kPoolHighWater) {
        tls_pool.push_back(this);
    } else {
        ::operator delete(this);
    }
}

// ---------------------------------------------------------------------------
// TensorImpl construction / destruction
// ---------------------------------------------------------------------------

void TensorImpl::init(Storage storage,
                      int64_t storage_offset,
                      DType dtype,
                      uint8_t ndim,
                      const int64_t* sizes,
                      const int64_t* strides) {
    if (ndim > kMaxDim) {
        std::fprintf(stderr, "Forge: ndim=%d exceeds kMaxDim=%d\n", ndim, kMaxDim);
        std::abort();
    }

    storage_ = std::move(storage);
    storage_offset_ = storage_offset;
    dtype_ = dtype;
    ndim_ = ndim;
    autograd_meta_ = nullptr;
    requires_grad_.store(false, std::memory_order_relaxed);
    refcount_.store(1, std::memory_order_relaxed);

    // Copy sizes
    std::memcpy(sizes_, sizes, ndim * sizeof(int64_t));
    // Zero remaining
    if (ndim < kMaxDim)
        std::memset(sizes_ + ndim, 0, (kMaxDim - ndim) * sizeof(int64_t));

    // Copy or compute strides
    if (strides) {
        std::memcpy(strides_, strides, ndim * sizeof(int64_t));
    } else {
        compute_contiguous_strides(sizes_, ndim, strides_);
    }
    if (ndim < kMaxDim)
        std::memset(strides_ + ndim, 0, (kMaxDim - ndim) * sizeof(int64_t));

    // Compute derived fields
    numel_ = compute_numel(sizes_, ndim);
    refresh_contiguity();
    refresh_data_ptr();

    // New tensors are leaves (not produced by autograd)
    flags_ |= kIsLeaf;
}

TensorImpl::~TensorImpl() {
    delete autograd_meta_;
    autograd_meta_ = nullptr;
}

void TensorImpl::refresh_contiguity() {
    if (is_contiguous_layout(sizes_, strides_, ndim_)) {
        flags_ |= kContiguous;
    } else {
        flags_ &= ~kContiguous;
    }
}

void TensorImpl::refresh_data_ptr() {
    if (storage_) {
        data_ptr_ = static_cast<char*>(storage_->data_) +
                    storage_offset_ * dtype_size(dtype_);
    } else {
        data_ptr_ = nullptr;
    }
}

} // namespace forge
