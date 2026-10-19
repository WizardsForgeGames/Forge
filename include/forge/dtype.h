// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Data-type enum and size query.
// Simple enum — NOT type-erased like PyTorch's TypeMeta.
#pragma once

#include <cstddef>
#include <cstdint>

namespace forge {

enum class DType : uint8_t {
    Float32,
    Float16,
    BFloat16,
    Int8,
    Int32,
    Int64,
    Bool
};

/// Returns the size in bytes of a single element of the given dtype.
constexpr size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32:  return 4;
        case DType::Float16:  return 2;
        case DType::BFloat16: return 2;
        case DType::Int8:     return 1;
        case DType::Int32:    return 4;
        case DType::Int64:    return 8;
        case DType::Bool:     return 1;
    }
    return 0; // unreachable
}

/// Returns a human-readable name for the given dtype.
constexpr const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::Float32:  return "float32";
        case DType::Float16:  return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int8:     return "int8";
        case DType::Int32:    return "int32";
        case DType::Int64:    return "int64";
        case DType::Bool:     return "bool";
    }
    return "unknown";
}

} // namespace forge
