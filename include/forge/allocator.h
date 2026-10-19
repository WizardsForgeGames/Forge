// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Caching CUDA memory allocator.
//
// Amortizes cudaMalloc overhead (~100-500us) to ~1us by maintaining a pool
// of previously allocated blocks. Stream-ordered: freed blocks record a CUDA
// event and are only reused once that event completes.
//
// Graph-aware: blocks allocated during CUDA Graph capture are pinned and
// cannot be evicted by release_cached(), ensuring replay uses the same
// device addresses.
//
// KNOWN FLAW (M4): Single global mutex for all allocations. Safe and correct
// for single-stream workloads. Will serialize under multi-stream patterns.
// The interface already takes cudaStream_t, so the fix is purely internal:
// replace this mutex with per-stream free lists. No call-site changes needed.
//
// KNOWN FLAW (M1): No pinned host memory allocator. Host<->device transfers
// require caller to manage pinned memory manually. Tensor::to_cpu() (when
// implemented) will use pageable memory (2-3x slower, no compute overlap).
//
// KNOWN FLAW (M2): No sub-block splitting. All returned pointers are
// cudaMalloc-aligned (256 bytes). If splitting is added later, it MUST
// preserve 256-byte alignment for float4 loads and Tensor Core wmma.
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace forge {

/// Allocator statistics.
struct AllocatorStats {
    size_t bytes_allocated;     // currently held by user code
    size_t bytes_cached;        // in free pool, reusable
    size_t bytes_peak;          // high-water mark of bytes_allocated
    size_t num_allocs;          // total allocate() calls
    size_t num_frees;           // total deallocate() calls
    size_t num_cache_hits;      // allocations served from cache
    size_t num_cudamallocs;     // actual cudaMalloc calls made
    size_t num_cudafrees;       // actual cudaFree calls made
};

class CachingAllocator {
public:
    /// Returns the singleton allocator for the given CUDA device.
    static CachingAllocator& instance(int device = 0);

    /// Allocate device memory. Returns a pointer to at least `nbytes` bytes.
    /// The returned block may be larger due to size-class rounding.
    /// The allocation is associated with `stream` for ordering purposes.
    void* allocate(size_t nbytes, cudaStream_t stream);

    /// Return memory to the cache. The deallocation is ordered with respect
    /// to `stream` — the block will not be reused until all preceding work
    /// on this stream completes.
    void deallocate(void* ptr, size_t nbytes, cudaStream_t stream);

    /// Free all cached (unused) blocks back to CUDA, except graph-pinned ones.
    /// Called automatically as part of OOM recovery.
    void release_cached();

    /// Unpin all blocks associated with a CUDA Graph capture ID, allowing
    /// them to be evicted by future release_cached() calls.
    void unpin_graph(uint32_t graph_id);

    /// Return current allocator statistics.
    AllocatorStats stats() const;

    ~CachingAllocator();

    CachingAllocator(const CachingAllocator&) = delete;
    CachingAllocator& operator=(const CachingAllocator&) = delete;

private:
    CachingAllocator(int device);

    struct Impl;
    Impl* impl_;
};

/// Round `nbytes` up to the appropriate size class.
/// Tiered rounding to balance fragmentation vs. waste:
///   <= 1 KB:    256 B granularity  (max 25% waste, tiny allocs)
///   1 KB-1 MB:  512 B granularity  (max <0.05% waste)
///   1 MB-32 MB: 2 MB granularity   (max <6.25% waste)
///   >32 MB:     exact size          (0% waste)
size_t round_size(size_t nbytes);

} // namespace forge
