// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// DeviceBuffer<T>: minimal GPU buffer abstraction.
// NOT a tensor library. Exists to carry a device pointer + element count
// together for safe passing and optional RAII lifetime.
//
// All Forge kernel APIs accept raw pointers -- DeviceBuffer is a convenience
// wrapper that implicitly converts to T*.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "error.h"

namespace forge {

template <typename T>
struct DeviceBuffer {
    T*      ptr   = nullptr;
    int64_t count = 0;
    bool    owned = false;

    DeviceBuffer() = default;

    /// Non-owning wrap of an existing device pointer.
    static DeviceBuffer wrap(T* device_ptr, int64_t num_elements) {
        DeviceBuffer b;
        b.ptr   = device_ptr;
        b.count = num_elements;
        b.owned = false;
        return b;
    }

    /// Owning allocation. Freed on destruction or explicit free().
    static DeviceBuffer alloc(int64_t num_elements) {
        DeviceBuffer b;
        b.count = num_elements;
        b.owned = true;
        if (num_elements > 0) {
            FORGE_CHECK(cudaMalloc(&b.ptr, num_elements * sizeof(T)));
            FORGE_CHECK(cudaMemset(b.ptr, 0, num_elements * sizeof(T)));
        }
        return b;
    }

    void free() {
        if (owned && ptr) { cudaFree(ptr); ptr = nullptr; }
        count = 0;
    }

    ~DeviceBuffer() { free(); }

    // Move-only
    DeviceBuffer(DeviceBuffer&& o) noexcept
        : ptr(o.ptr), count(o.count), owned(o.owned) {
        o.ptr = nullptr; o.count = 0; o.owned = false;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            free();
            ptr = o.ptr; count = o.count; owned = o.owned;
            o.ptr = nullptr; o.count = 0; o.owned = false;
        }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    int64_t bytes() const { return count * sizeof(T); }

    operator T*() { return ptr; }
    operator const T*() const { return ptr; }
};

} // namespace forge
