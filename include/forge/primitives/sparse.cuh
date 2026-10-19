// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Minimal CSR/CSC accessor helpers for device code.
// Readability wrappers -- inline to bare pointer arithmetic.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace primitives {

/// Starting index of nonzeros for the given CSR row.
__device__ __forceinline__ int csr_row_begin(const int* row_ptr, int row) {
    return row_ptr[row];
}

/// One-past-end index of nonzeros for the given CSR row.
__device__ __forceinline__ int csr_row_end(const int* row_ptr, int row) {
    return row_ptr[row + 1];
}

/// Number of nonzeros in the given CSR row.
__device__ __forceinline__ int csr_row_nnz(const int* row_ptr, int row) {
    return row_ptr[row + 1] - row_ptr[row];
}

}} // namespace forge::primitives
