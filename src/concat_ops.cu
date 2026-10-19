// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Concatenation, stacking, and splitting operations: kernels + dispatch.
#include "forge/concat_ops.h"
#include "forge/error.h"
#include "internal/cuda_utils.cuh"

#include <cstdio>
#include <cstring>
#include <algorithm>

namespace forge {

// =====================================================================
// Internal helpers
// =====================================================================

namespace {

inline cudaStream_t resolve_stream(cudaStream_t stream) {
    return stream ? stream : getCurrentStream();
}

inline int normalize_dim(int dim, int ndim) {
    if (dim < 0) dim += ndim;
    return dim;
}

} // anonymous namespace

// =====================================================================
// cat kernel: copy one input tensor's data into the output at offset
// =====================================================================

// Each invocation copies one source tensor into the correct slice of output.
// Decompose: linear → (outer, k, inner)
//   dst[outer][dim_offset + k][inner] = src[outer][k][inner]
__global__ void cat_copy_kernel(
    const float* __restrict__ src,
    float* __restrict__       dst,
    int64_t                   outer_size,
    int64_t                   src_dim_size,
    int64_t                   inner_size,
    int64_t                   dst_dim_size,
    int64_t                   dim_offset)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    int64_t inner = idx % inner_size;
    int64_t k = (idx / inner_size) % src_dim_size;
    int64_t outer = idx / (src_dim_size * inner_size);

    int64_t dst_off = outer * dst_dim_size * inner_size
                    + (dim_offset + k) * inner_size + inner;
    dst[dst_off] = src[idx];
}

// =====================================================================
// cat
// =====================================================================

Tensor cat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) {
        std::fprintf(stderr, "Forge: cat() requires at least one tensor\n");
        std::abort();
    }

    cudaStream_t stream = resolve_stream(nullptr);

    // Make all inputs contiguous
    std::vector<Tensor> contig;
    contig.reserve(tensors.size());
    for (auto& t : tensors) contig.push_back(t.contiguous());

    int nd = contig[0].ndim();
    dim = normalize_dim(dim, nd);

    // Validate shapes and compute output dim size
    int64_t cat_dim_total = 0;
    for (size_t i = 0; i < contig.size(); i++) {
        if (contig[i].ndim() != nd) {
            std::fprintf(stderr, "Forge: cat() all tensors must have same ndim\n");
            std::abort();
        }
        for (int d = 0; d < nd; d++) {
            if (d != dim && contig[i].size(d) != contig[0].size(d)) {
                std::fprintf(stderr,
                    "Forge: cat() shape mismatch at dim %d: %lld vs %lld\n",
                    d, (long long)contig[i].size(d), (long long)contig[0].size(d));
                std::abort();
            }
        }
        cat_dim_total += contig[i].size(dim);
    }

    // Compute output shape
    int64_t out_sizes[kMaxDim];
    for (int d = 0; d < nd; d++)
        out_sizes[d] = (d == dim) ? cat_dim_total : contig[0].size(d);

    int64_t outer_size = 1;
    for (int d = 0; d < dim; d++) outer_size *= out_sizes[d];
    int64_t inner_size = 1;
    for (int d = dim + 1; d < nd; d++) inner_size *= out_sizes[d];

    Tensor out = Tensor::empty(out_sizes, nd, DType::Float32, stream);
    if (out.numel() == 0) return out;

    // Fast path: dim=0 contiguous, use cudaMemcpyAsync
    if (dim == 0) {
        int64_t byte_offset = 0;
        for (auto& t : contig) {
            size_t nbytes = t.numel() * sizeof(float);
            if (nbytes > 0) {
                FORGE_CHECK(cudaMemcpyAsync(
                    static_cast<char*>(out.data_ptr()) + byte_offset,
                    t.data_ptr(), nbytes, cudaMemcpyDeviceToDevice, stream));
            }
            byte_offset += nbytes;
        }
        return out;
    }

    // General path: launch one kernel per input
    constexpr int BLOCK = 256;
    int64_t dim_offset = 0;
    for (auto& t : contig) {
        int64_t src_numel = t.numel();
        if (src_numel > 0) {
            int grid = internal::divup(static_cast<int>(src_numel), BLOCK);
            cat_copy_kernel<<<grid, BLOCK, 0, stream>>>(
                t.data<float>(), out.data<float>(),
                outer_size, t.size(dim), inner_size,
                cat_dim_total, dim_offset);
        }
        dim_offset += t.size(dim);
    }

    return out;
}

// =====================================================================
// stack: unsqueeze each tensor, then cat
// =====================================================================

Tensor stack(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) {
        std::fprintf(stderr, "Forge: stack() requires at least one tensor\n");
        std::abort();
    }

    int nd = tensors[0].ndim();
    if (dim < 0) dim += nd + 1;

    // Validate all tensors have the same shape
    for (size_t i = 1; i < tensors.size(); i++) {
        if (tensors[i].ndim() != nd) {
            std::fprintf(stderr, "Forge: stack() all tensors must have same ndim\n");
            std::abort();
        }
        for (int d = 0; d < nd; d++) {
            if (tensors[i].size(d) != tensors[0].size(d)) {
                std::fprintf(stderr,
                    "Forge: stack() shape mismatch at dim %d\n", d);
                std::abort();
            }
        }
    }

    std::vector<Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());
    for (auto& t : tensors)
        unsqueezed.push_back(t.unsqueeze(dim));

    return cat(unsqueezed, dim);
}

// =====================================================================
// hstack / vstack
// =====================================================================

Tensor hstack(const std::vector<Tensor>& tensors) {
    if (tensors.empty()) {
        std::fprintf(stderr, "Forge: hstack() requires at least one tensor\n");
        std::abort();
    }
    // 1D tensors: cat along dim=0
    if (tensors[0].ndim() == 1) return cat(tensors, 0);
    // Otherwise: cat along dim=1
    return cat(tensors, 1);
}

Tensor vstack(const std::vector<Tensor>& tensors) {
    return cat(tensors, 0);
}

// =====================================================================
// split / chunk (zero-copy views via slice)
// =====================================================================

std::vector<Tensor> split(const Tensor& tensor, int64_t split_size, int dim) {
    int nd = tensor.ndim();
    dim = normalize_dim(dim, nd);
    int64_t dim_size = tensor.size(dim);

    std::vector<Tensor> result;
    int64_t offset = 0;
    while (offset < dim_size) {
        int64_t length = std::min(split_size, dim_size - offset);
        result.push_back(tensor.slice(dim, offset, length));
        offset += length;
    }
    return result;
}

std::vector<Tensor> chunk(const Tensor& tensor, int n_chunks, int dim) {
    int nd = tensor.ndim();
    dim = normalize_dim(dim, nd);
    int64_t dim_size = tensor.size(dim);

    int64_t chunk_size = (dim_size + n_chunks - 1) / n_chunks;
    return split(tensor, chunk_size, dim);
}

} // namespace forge
