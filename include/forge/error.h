// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace forge {

#define FORGE_CHECK(call) do {                                               \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess) {                                               \
        std::fprintf(stderr, "Forge CUDA error at %s:%d: %s\n",             \
                     __FILE__, __LINE__, cudaGetErrorString(err_));          \
        std::abort();                                                        \
    }                                                                        \
} while (0)

} // namespace forge
