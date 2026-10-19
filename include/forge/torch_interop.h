// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Zero-copy interop between Forge::Tensor and torch::Tensor.
// Compiled only when FORGE_HAS_TORCH is defined (i.e., when building
// the Python extension with PyTorch present).
//
// Both directions are zero-copy — they share the underlying device pointer.
// The non-owning side must not outlive the owning side.
#pragma once

#include "forge/tensor.h"

#ifdef FORGE_HAS_TORCH

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>

namespace forge {

/// Convert a torch::Tensor to a Forge Tensor (zero-copy, non-owning).
/// The returned Forge Tensor wraps the same device pointer and will NOT
/// free it. The original PyTorch tensor must stay alive.
inline Tensor from_torch(const torch::Tensor& t) {
    // Ensure contiguous and on CUDA
    auto ct = t.contiguous();

    // Map PyTorch dtype to Forge dtype
    DType dtype;
    switch (ct.scalar_type()) {
        case torch::kFloat32:  dtype = DType::Float32;  break;
        case torch::kFloat16:  dtype = DType::Float16;  break;
        case torch::kBFloat16: dtype = DType::BFloat16; break;
        case torch::kInt8:     dtype = DType::Int8;     break;
        case torch::kInt32:    dtype = DType::Int32;    break;
        case torch::kInt64:    dtype = DType::Int64;    break;
        case torch::kBool:     dtype = DType::Bool;     break;
        default:
            std::fprintf(stderr, "Forge: unsupported PyTorch dtype for interop\n");
            std::abort();
    }

    int ndim = ct.dim();
    int64_t sizes[kMaxDim];
    for (int i = 0; i < ndim; ++i) sizes[i] = ct.size(i);

    return Tensor::from_blob(ct.data_ptr(), sizes, ndim, dtype);
}

/// Convert a Forge Tensor to a torch::Tensor (zero-copy, non-owning).
/// The returned PyTorch tensor wraps the same device pointer and will NOT
/// free it. The original Forge tensor must stay alive.
inline torch::Tensor to_torch(const Tensor& t) {
    // Map Forge dtype to PyTorch dtype
    auto options = torch::TensorOptions().device(torch::kCUDA);
    switch (t.dtype()) {
        case DType::Float32:  options = options.dtype(torch::kFloat32);  break;
        case DType::Float16:  options = options.dtype(torch::kFloat16);  break;
        case DType::BFloat16: options = options.dtype(torch::kBFloat16); break;
        case DType::Int8:     options = options.dtype(torch::kInt8);     break;
        case DType::Int32:    options = options.dtype(torch::kInt32);    break;
        case DType::Int64:    options = options.dtype(torch::kInt64);    break;
        case DType::Bool:     options = options.dtype(torch::kBool);     break;
    }

    std::vector<int64_t> sizes(t.sizes(), t.sizes() + t.ndim());
    return torch::from_blob(t.data_ptr(), sizes, options);
}

} // namespace forge

#endif // FORGE_HAS_TORCH
