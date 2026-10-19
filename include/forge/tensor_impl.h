// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// TensorImpl: core tensor metadata (192 bytes, 3 cache lines).
//
// Layout:
//   Cache line 1 (64B) — HOT: data_ptr, numel, storage, offset, autograd, dtype, flags
//   Cache lines 2-3 (128B) — COLD: sizes[8], strides[8]
//
// Object pool: thread-local free list eliminates heap churn across 250K+
// training iterations. Use TensorImpl::create() / release() instead of
// new / delete.
//
// KNOWN FLAW (M3): TensorImpl fields are struct-public. Nothing prevents
// external code from mutating sizes_/strides_ directly, bypassing contiguity
// flag updates. Currently safe because all mutation goes through Tensor's
// view methods. If third-party access is ever permitted, these must become
// private with accessor methods enforcing contiguity flag consistency.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <initializer_list>

#include "forge/dtype.h"
#include "forge/storage.h"
#include "forge/autograd_fwd.h"

namespace forge {

/// Maximum number of dimensions. Hard cap — no heap fallback.
constexpr int kMaxDim = 8;

/// Flag bits packed into TensorImpl::flags_ (IMMUTABLE after construction).
enum TensorFlags : uint8_t {
    kContiguous = 1 << 0,
    kIsLeaf     = 1 << 1,
    kIsView     = 1 << 2,
};

class TensorImpl {
public:
    // === Cache line 1 (64 bytes) — HOT PATH ===
    void*                    data_ptr_;        //  8B  cached device pointer (H1)
    int64_t                  numel_;           //  8B
    Storage                  storage_;         //  8B  intrusive ptr to StorageImpl
    int64_t                  storage_offset_;  //  8B  element offset into storage
    AutogradMeta*            autograd_meta_;   //  8B  nullptr until needed
    DType                    dtype_;           //  1B
    uint8_t                  ndim_;            //  1B
    uint8_t                  flags_;           //  1B  is_contiguous | is_leaf | is_view
    std::atomic<bool>        requires_grad_;   //  1B  separate atomic (C8)
    std::atomic<int>         refcount_;        //  4B  intrusive refcount
    // 16B padding to 64B boundary

    // === Cache lines 2-3 (128 bytes) — COLD PATH ===
    int64_t                  sizes_[kMaxDim];   // 64B
    int64_t                  strides_[kMaxDim]; // 64B

    // === Total: 192 bytes (3 cache lines) ===

    // ------------------------------------------------------------------
    // Object pool (H3)
    // ------------------------------------------------------------------

    /// Create a new TensorImpl from the pool or heap.
    static TensorImpl* create(Storage storage,
                              int64_t storage_offset,
                              DType dtype,
                              uint8_t ndim,
                              const int64_t* sizes,
                              const int64_t* strides);

    /// Return this TensorImpl to the pool (or free to heap).
    void release();

    /// Increment reference count.
    void retain() { refcount_.fetch_add(1, std::memory_order_relaxed); }

    /// Decrement reference count. Returns true if this was the last reference.
    bool deref() { return refcount_.fetch_sub(1, std::memory_order_acq_rel) == 1; }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    template <typename T>
    T* data_ptr() const {
        return static_cast<T*>(data_ptr_);
    }

    bool is_contiguous() const { return flags_ & kContiguous; }
    bool is_leaf()       const { return flags_ & kIsLeaf; }
    bool is_view()       const { return flags_ & kIsView; }

    bool requires_grad() const {
        return requires_grad_.load(std::memory_order_relaxed);
    }

    void set_requires_grad(bool val) {
        requires_grad_.store(val, std::memory_order_relaxed);
        if (val && !autograd_meta_) {
            autograd_meta_ = new AutogradMeta();
        }
    }

    /// Recompute the contiguity flag from current sizes/strides.
    void refresh_contiguity();

    /// Compute and cache data_ptr_ from storage base + offset.
    void refresh_data_ptr();

private:
    TensorImpl() = default;
    ~TensorImpl();

    void init(Storage storage,
              int64_t storage_offset,
              DType dtype,
              uint8_t ndim,
              const int64_t* sizes,
              const int64_t* strides);

    friend class Tensor;
};

/// Compute the number of elements from a shape array.
inline int64_t compute_numel(const int64_t* sizes, int ndim) {
    int64_t n = 1;
    for (int i = 0; i < ndim; ++i) n *= sizes[i];
    return n;
}

/// Compute row-major (C-contiguous) strides from a shape array.
inline void compute_contiguous_strides(const int64_t* sizes, int ndim,
                                       int64_t* out_strides) {
    if (ndim == 0) return;
    out_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * sizes[i + 1];
    }
}

/// Check whether a set of sizes/strides describes a contiguous layout.
inline bool is_contiguous_layout(const int64_t* sizes, const int64_t* strides,
                                 int ndim) {
    if (ndim == 0) return true;
    int64_t expected = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (sizes[i] != 1 && strides[i] != expected) return false;
        expected *= sizes[i];
    }
    return true;
}

} // namespace forge
