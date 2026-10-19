// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Intrusive reference-counted device memory block.
//
// StorageImpl holds the raw device pointer, allocation size, and metadata.
// Storage is a lightweight handle (single pointer) that manages the refcount.
//
// Two modes:
//   - Owning: allocated via CachingAllocator, freed when refcount hits zero.
//   - Non-owning: wraps an external pointer (e.g., PyTorch interop). Never freed.
//
// Views (Tensors with different shapes/strides) share the same Storage.
// The version counter tracks data mutations and is shared between all views
// by construction (it lives in StorageImpl).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace forge {

/// Core storage implementation. Reference-counted, not directly used by callers.
struct StorageImpl {
    void*                data_;         //  8B  device pointer
    size_t               nbytes_;       //  8B  allocation size in bytes
    cudaStream_t         alloc_stream_; //  8B  stream used for allocation
    std::atomic<int>     refcount_;     //  4B  intrusive refcount
    std::atomic<uint32_t> version_;     //  4B  mutation counter (shared by views)
    int8_t               device_;       //  1B  CUDA device index
    bool                 owns_;         //  1B  true = freed via CachingAllocator
    // 6B padding (48 bytes total)

    /// Increment the reference count.
    void retain() { refcount_.fetch_add(1, std::memory_order_relaxed); }

    /// Decrement the reference count. Returns true if this was the last reference.
    bool release() {
        // Release semantics: all preceding writes visible to the thread that
        // observes the refcount drop to zero.
        return refcount_.fetch_sub(1, std::memory_order_acq_rel) == 1;
    }

    /// Bump the version counter. Called on any in-place data mutation.
    void bump_version() { version_.fetch_add(1, std::memory_order_relaxed); }

    /// Read the current version.
    uint32_t version() const { return version_.load(std::memory_order_relaxed); }
};

/// Create an owning StorageImpl backed by the CachingAllocator.
StorageImpl* storage_create(size_t nbytes, cudaStream_t stream);

/// Create a non-owning StorageImpl wrapping an external device pointer.
StorageImpl* storage_wrap(void* data, size_t nbytes, int device);

/// Destroy a StorageImpl, returning memory to the allocator if owning.
void storage_destroy(StorageImpl* s);


/// Lightweight handle to StorageImpl. Manages refcount on copy/destroy.
/// 8 bytes — just a pointer.
class Storage {
public:
    Storage() : impl_(nullptr) {}

    /// Takes ownership of a raw StorageImpl pointer (does NOT increment refcount).
    explicit Storage(StorageImpl* impl) : impl_(impl) {}

    Storage(const Storage& o) : impl_(o.impl_) {
        if (impl_) impl_->retain();
    }

    Storage(Storage&& o) noexcept : impl_(o.impl_) {
        o.impl_ = nullptr;
    }

    Storage& operator=(const Storage& o) {
        if (this != &o) {
            if (impl_ && impl_->release()) storage_destroy(impl_);
            impl_ = o.impl_;
            if (impl_) impl_->retain();
        }
        return *this;
    }

    Storage& operator=(Storage&& o) noexcept {
        if (this != &o) {
            if (impl_ && impl_->release()) storage_destroy(impl_);
            impl_ = o.impl_;
            o.impl_ = nullptr;
        }
        return *this;
    }

    ~Storage() {
        if (impl_ && impl_->release()) storage_destroy(impl_);
    }

    StorageImpl* get() const { return impl_; }
    StorageImpl* operator->() const { return impl_; }
    explicit operator bool() const { return impl_ != nullptr; }

    /// Number of active references to this storage.
    int use_count() const {
        return impl_ ? impl_->refcount_.load(std::memory_order_relaxed) : 0;
    }

private:
    StorageImpl* impl_;
};

} // namespace forge
