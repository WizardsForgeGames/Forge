// Forge GPU Compute Library -- Internal
// Copyright (c) 2026 Darren Sander. MIT License.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace internal {

inline int divup(int a, int b) { return (a + b - 1) / b; }

// Default block size for simple elementwise kernels.
constexpr int kDefaultBlockSize = 256;

/// Returns the multiprocessor count for the given device.
/// Cached after first call (thread-safe via static local).
inline int get_sm_count(int device = 0) {
    static int cached = 0;
    if (cached == 0)
        cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, device);
    return cached;
}

}} // namespace forge::internal
