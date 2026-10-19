// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include "forge/allocator.h"
#include "forge/error.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace forge {

// ---------------------------------------------------------------------------
// Size-class rounding
// ---------------------------------------------------------------------------

size_t round_size(size_t nbytes) {
    if (nbytes == 0) return 0;

    constexpr size_t k1KB  = 1024;
    constexpr size_t k1MB  = 1024 * 1024;
    constexpr size_t k32MB = 32 * 1024 * 1024;

    if (nbytes <= k1KB) {
        // 256-byte granularity
        return ((nbytes + 255) / 256) * 256;
    } else if (nbytes <= k1MB) {
        // 512-byte granularity
        return ((nbytes + 511) / 512) * 512;
    } else if (nbytes <= k32MB) {
        // 2 MB granularity
        constexpr size_t k2MB = 2 * 1024 * 1024;
        return ((nbytes + k2MB - 1) / k2MB) * k2MB;
    } else {
        // Exact size (already cudaMalloc-aligned)
        return nbytes;
    }
}

// ---------------------------------------------------------------------------
// Block metadata
// ---------------------------------------------------------------------------

struct Block {
    void*        ptr;
    size_t       size;           // allocated (post-rounding)
    size_t       requested_size; // pre-rounding (for stats)
    cudaStream_t stream;
    cudaEvent_t  free_event;
    bool         event_recorded;
    bool         graph_pinned;
    uint32_t     graph_id;
};

// ---------------------------------------------------------------------------
// CachingAllocator::Impl
// ---------------------------------------------------------------------------

struct CachingAllocator::Impl {
    int device_;
    mutable std::mutex mutex_;

    // Free blocks indexed by rounded size for O(log n) lookup.
    // Multiple blocks can share the same size, so we use multimap.
    std::multimap<size_t, Block> free_blocks_;

    // Active blocks indexed by pointer for O(1) deallocation.
    std::unordered_map<void*, Block> active_blocks_;

    AllocatorStats stats_{};

    explicit Impl(int device) : device_(device) {}

    ~Impl() {
        // Free all blocks (both active and cached).
        // Active blocks indicate a leak, but we free them to avoid
        // CUDA driver errors at process exit.
        for (auto& [ptr, block] : active_blocks_) {
            if (block.free_event) cudaEventDestroy(block.free_event);
            cudaFree(block.ptr);
        }
        for (auto& [size, block] : free_blocks_) {
            if (block.free_event) cudaEventDestroy(block.free_event);
            cudaFree(block.ptr);
        }
    }

    // Detect whether we are inside a CUDA Graph capture.
    bool is_capturing(cudaStream_t stream) const {
        cudaStreamCaptureStatus status;
        // cudaStreamIsCapturing returns error if stream is null or invalid;
        // treat any error as "not capturing".
        cudaError_t err = cudaStreamIsCapturing(stream, &status);
        if (err != cudaSuccess) {
            cudaGetLastError(); // clear error
            return false;
        }
        return status == cudaStreamCaptureStatusActive;
    }

    // Get total and free device memory.
    void query_device_memory(size_t& total, size_t& free) const {
        int prev_device;
        cudaGetDevice(&prev_device);
        if (prev_device != device_) cudaSetDevice(device_);
        cudaMemGetInfo(&free, &total);
        if (prev_device != device_) cudaSetDevice(prev_device);
    }

    void* allocate(size_t nbytes, cudaStream_t stream) {
        if (nbytes == 0) return nullptr;

        size_t rounded = round_size(nbytes);
        std::lock_guard<std::mutex> lock(mutex_);

        stats_.num_allocs++;

        // Step 1: Try cache — find a free block of matching size
        void* ptr = try_cache(rounded, stream);
        if (ptr) {
            record_active(ptr, rounded, nbytes, stream);
            return ptr;
        }

        // Proactive GC: if we're above 80% VRAM, free cached blocks first
        {
            size_t total, free_mem;
            query_device_memory(total, free_mem);
            if (free_mem < total / 5) { // less than 20% free
                release_cached_locked();
                // Try cache again after GC
                ptr = try_cache(rounded, stream);
                if (ptr) {
                    record_active(ptr, rounded, nbytes, stream);
                    return ptr;
                }
            }
        }

        // Step 2: Try cudaMalloc
        ptr = try_cudamalloc(rounded);
        if (ptr) {
            record_active(ptr, rounded, nbytes, stream);
            return ptr;
        }

        // Step 3: OOM recovery — free cached blocks, retry
        release_cached_locked();
        ptr = try_cudamalloc(rounded);
        if (ptr) {
            record_active(ptr, rounded, nbytes, stream);
            return ptr;
        }

        // Step 4: Fatal OOM
        size_t total, free_mem;
        query_device_memory(total, free_mem);
        std::fprintf(stderr,
            "Forge OOM: requested %zu bytes (rounded %zu), "
            "allocated %zu, cached %zu, device free %zu / %zu total\n",
            nbytes, rounded,
            stats_.bytes_allocated, stats_.bytes_cached,
            free_mem, total);
        std::abort();
        return nullptr; // unreachable
    }

    void deallocate(void* ptr, size_t /*nbytes*/, cudaStream_t stream) {
        if (!ptr) return;

        std::lock_guard<std::mutex> lock(mutex_);
        stats_.num_frees++;

        auto it = active_blocks_.find(ptr);
        if (it == active_blocks_.end()) {
            std::fprintf(stderr, "Forge allocator: deallocate of unknown pointer %p\n", ptr);
            return;
        }

        Block block = it->second;
        active_blocks_.erase(it);

        stats_.bytes_allocated -= block.size;

        // Record event so this block is only reused after pending work completes
        if (!block.free_event) {
            FORGE_CHECK(cudaEventCreateWithFlags(&block.free_event, cudaEventDisableTiming));
        }
        FORGE_CHECK(cudaEventRecord(block.free_event, stream));
        block.event_recorded = true;
        block.stream = stream;

        stats_.bytes_cached += block.size;
        free_blocks_.emplace(block.size, block);
    }

    void release_cached() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_cached_locked();
    }

    void unpin_graph(uint32_t graph_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Unpin active blocks
        for (auto& [ptr, block] : active_blocks_) {
            if (block.graph_pinned && block.graph_id == graph_id) {
                block.graph_pinned = false;
                block.graph_id = 0;
            }
        }
        // Unpin free blocks
        for (auto& [size, block] : free_blocks_) {
            if (block.graph_pinned && block.graph_id == graph_id) {
                block.graph_pinned = false;
                block.graph_id = 0;
            }
        }
    }

    AllocatorStats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

private:
    // Try to find a matching free block. Returns nullptr on miss.
    void* try_cache(size_t rounded, cudaStream_t stream) {
        auto it = free_blocks_.lower_bound(rounded);
        // Look for an exact-size match (or the smallest block >= rounded)
        while (it != free_blocks_.end() && it->first == rounded) {
            Block& block = it->second;
            // Check if the event has completed (block is safe to reuse)
            if (block.event_recorded) {
                cudaError_t err = cudaEventQuery(block.free_event);
                if (err == cudaErrorNotReady) {
                    ++it;
                    continue; // still in use on another stream
                }
                // cudaSuccess or other error — treat as ready
                if (err != cudaSuccess) cudaGetLastError(); // clear error
            }

            void* ptr = block.ptr;
            // Detect graph capture and pin the block
            if (is_capturing(stream)) {
                block.graph_pinned = true;
                // Use a simple incrementing ID; real graph ID would come
                // from the capture API but that requires additional tracking.
                static uint32_t next_graph_id = 1;
                block.graph_id = next_graph_id++;
            }

            stats_.bytes_cached -= block.size;
            stats_.num_cache_hits++;
            free_blocks_.erase(it);
            return ptr;
        }
        return nullptr;
    }

    // Try a cudaMalloc. Returns nullptr on failure (does not abort).
    void* try_cudamalloc(size_t nbytes) {
        int prev_device;
        cudaGetDevice(&prev_device);
        if (prev_device != device_) cudaSetDevice(device_);

        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, nbytes);

        if (prev_device != device_) cudaSetDevice(prev_device);

        if (err != cudaSuccess) {
            cudaGetLastError(); // clear error
            return nullptr;
        }
        stats_.num_cudamallocs++;
        return ptr;
    }

    // Record a block as active (in use by caller).
    void record_active(void* ptr, size_t rounded, size_t requested,
                       cudaStream_t stream) {
        Block block{};
        block.ptr = ptr;
        block.size = rounded;
        block.requested_size = requested;
        block.stream = stream;
        block.free_event = nullptr;
        block.event_recorded = false;
        block.graph_pinned = false;
        block.graph_id = 0;

        // Detect graph capture
        if (is_capturing(stream)) {
            block.graph_pinned = true;
            static uint32_t next_graph_id = 1;
            block.graph_id = next_graph_id++;
        }

        active_blocks_[ptr] = block;
        stats_.bytes_allocated += rounded;
        if (stats_.bytes_allocated > stats_.bytes_peak)
            stats_.bytes_peak = stats_.bytes_allocated;
    }

    // Free all cached blocks that are not graph-pinned.
    void release_cached_locked() {
        auto it = free_blocks_.begin();
        while (it != free_blocks_.end()) {
            Block& block = it->second;
            if (block.graph_pinned) {
                ++it;
                continue; // skip graph-pinned blocks
            }
            if (block.free_event) {
                // Wait for any pending work before freeing
                cudaEventSynchronize(block.free_event);
                cudaEventDestroy(block.free_event);
            }
            cudaFree(block.ptr);
            stats_.num_cudafrees++;
            stats_.bytes_cached -= block.size;
            it = free_blocks_.erase(it);
        }
    }
};

// ---------------------------------------------------------------------------
// CachingAllocator public interface
// ---------------------------------------------------------------------------

CachingAllocator& CachingAllocator::instance(int device) {
    // One allocator per device. Max 16 devices should be more than enough.
    static CachingAllocator* allocators[16] = {};
    static std::mutex init_mutex;

    if (device < 0 || device >= 16) {
        std::fprintf(stderr, "Forge allocator: invalid device %d\n", device);
        std::abort();
    }

    if (!allocators[device]) {
        std::lock_guard<std::mutex> lock(init_mutex);
        if (!allocators[device]) {
            allocators[device] = new CachingAllocator(device);
        }
    }
    return *allocators[device];
}

CachingAllocator::CachingAllocator(int device)
    : impl_(new Impl(device)) {}

CachingAllocator::~CachingAllocator() {
    delete impl_;
}

void* CachingAllocator::allocate(size_t nbytes, cudaStream_t stream) {
    return impl_->allocate(nbytes, stream);
}

void CachingAllocator::deallocate(void* ptr, size_t nbytes, cudaStream_t stream) {
    impl_->deallocate(ptr, nbytes, stream);
}

void CachingAllocator::release_cached() {
    impl_->release_cached();
}

void CachingAllocator::unpin_graph(uint32_t graph_id) {
    impl_->unpin_graph(graph_id);
}

AllocatorStats CachingAllocator::stats() const {
    return impl_->get_stats();
}

} // namespace forge
