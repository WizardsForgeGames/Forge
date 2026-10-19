// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Float atomic min/max via CAS loop.
// CUDA only provides atomicMin/atomicMax for integers.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace primitives {

/// Atomic max for float. Updates *addr = max(*addr, val).
__device__ __forceinline__ void atomic_max_float(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

/// Atomic min for float. Updates *addr = min(*addr, val).
__device__ __forceinline__ void atomic_min_float(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

}} // namespace forge::primitives
