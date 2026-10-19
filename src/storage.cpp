// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include "forge/storage.h"
#include "forge/allocator.h"
#include "forge/error.h"

namespace forge {

StorageImpl* storage_create(size_t nbytes, cudaStream_t stream) {
    auto* s = new StorageImpl{};
    s->nbytes_ = nbytes;
    s->alloc_stream_ = stream;
    s->refcount_.store(1, std::memory_order_relaxed);
    s->version_.store(0, std::memory_order_relaxed);
    s->owns_ = true;

    // Record current device
    int device = 0;
    cudaGetDevice(&device);
    s->device_ = static_cast<int8_t>(device);

    if (nbytes > 0) {
        s->data_ = CachingAllocator::instance(device).allocate(nbytes, stream);
    } else {
        s->data_ = nullptr;
    }

    return s;
}

StorageImpl* storage_wrap(void* data, size_t nbytes, int device) {
    auto* s = new StorageImpl{};
    s->data_ = data;
    s->nbytes_ = nbytes;
    s->alloc_stream_ = nullptr;
    s->refcount_.store(1, std::memory_order_relaxed);
    s->version_.store(0, std::memory_order_relaxed);
    s->device_ = static_cast<int8_t>(device);
    s->owns_ = false;
    return s;
}

void storage_destroy(StorageImpl* s) {
    if (!s) return;
    if (s->owns_ && s->data_) {
        CachingAllocator::instance(s->device_)
            .deallocate(s->data_, s->nbytes_, s->alloc_stream_);
    }
    delete s;
}

} // namespace forge
