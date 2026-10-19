// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Thread-local CUDA stream management.
//
// Every thread gets a non-default stream by default, avoiding the silent
// synchronization bug where cudaStreamDefault (the null stream) serializes
// with every other stream on the device.
//
// Usage:
//   cudaStream_t s = forge::getCurrentStream();   // thread-local
//   {
//       forge::StreamGuard guard(my_stream);       // sets + restores
//       // ... all Forge ops use my_stream ...
//   }
//   // back to previous stream
#pragma once

#include <cuda_runtime.h>

namespace forge {

/// Returns the current CUDA stream for this thread.
/// On first call per thread, lazily creates a non-default stream.
cudaStream_t getCurrentStream();

/// Sets the current CUDA stream for this thread.
void setCurrentStream(cudaStream_t stream);

/// RAII guard that sets the current stream on construction and restores
/// the previous stream on destruction.
struct StreamGuard {
    explicit StreamGuard(cudaStream_t stream);
    ~StreamGuard();

    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;

private:
    cudaStream_t prev_;
};

} // namespace forge
