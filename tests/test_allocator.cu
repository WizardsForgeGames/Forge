// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/allocator.h>
#include "test_helpers.cuh"

using namespace forge;

int main() {
    std::printf("=== Test: CachingAllocator ===\n");

    auto& alloc = CachingAllocator::instance(0);
    cudaStream_t stream = nullptr;

    // ------ round_size correctness ------
    {
        // Zero
        FORGE_ASSERT(round_size(0) == 0, "round_size(0) should be 0");

        // <= 1KB: 256-byte granularity
        FORGE_ASSERT(round_size(1) == 256, "round_size(1) should be 256");
        FORGE_ASSERT(round_size(256) == 256, "round_size(256) should be 256");
        FORGE_ASSERT(round_size(257) == 512, "round_size(257) should be 512");
        FORGE_ASSERT(round_size(512) == 512, "round_size(512) should be 512");
        FORGE_ASSERT(round_size(1024) == 1024, "round_size(1024) should be 1024");

        // 1KB-1MB: 512-byte granularity
        FORGE_ASSERT(round_size(1025) == 1536, "round_size(1025) should be 1536");
        FORGE_ASSERT(round_size(2048) == 2048, "round_size(2048) should be 2048");
        FORGE_ASSERT(round_size(2049) == 2560, "round_size(2049) should be 2560");
        size_t k1MB = 1024 * 1024;
        FORGE_ASSERT(round_size(k1MB) == k1MB, "round_size(1MB) should be 1MB");

        // 1MB-32MB: 2MB granularity
        size_t k2MB = 2 * 1024 * 1024;
        FORGE_ASSERT(round_size(k1MB + 1) == k2MB, "round_size(1MB+1) should be 2MB");
        size_t k32MB = 32 * 1024 * 1024;
        FORGE_ASSERT(round_size(k32MB) == k32MB, "round_size(32MB) should be 32MB");

        // >32MB: exact size
        FORGE_ASSERT(round_size(k32MB + 1) == k32MB + 1,
                     "round_size(>32MB) should be exact");
        FORGE_ASSERT(round_size(100 * 1024 * 1024) == 100 * 1024 * 1024,
                     "round_size(100MB) should be exact");

        std::printf("  PASS [round_size]\n");
    }

    // ------ Alloc/free roundtrip ------
    {
        size_t nbytes = 4096;
        void* ptr = alloc.allocate(nbytes, stream);
        FORGE_ASSERT(ptr != nullptr, "allocate should return non-null");

        // Write and readback to verify memory is usable
        float val = 42.0f;
        cudaMemcpy(ptr, &val, sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        float readback = 0.0f;
        cudaMemcpy(&readback, ptr, sizeof(float), cudaMemcpyDeviceToHost);
        FORGE_ASSERT(readback == 42.0f, "alloc roundtrip: readback should be 42.0");

        alloc.deallocate(ptr, nbytes, stream);
        cudaDeviceSynchronize();
        std::printf("  PASS [alloc_free_roundtrip]\n");
    }

    // ------ Cache hit ------
    {
        // Start fresh by releasing any cached blocks
        alloc.release_cached();
        cudaDeviceSynchronize();

        AllocatorStats before = alloc.stats();

        size_t nbytes = 8192;
        void* ptr1 = alloc.allocate(nbytes, stream);
        alloc.deallocate(ptr1, nbytes, stream);
        cudaDeviceSynchronize();

        AllocatorStats mid = alloc.stats();
        size_t hits_before = mid.num_cache_hits;

        void* ptr2 = alloc.allocate(nbytes, stream);
        AllocatorStats after = alloc.stats();

        FORGE_ASSERT(ptr2 == ptr1, "cache hit: re-alloc should return same pointer");
        FORGE_ASSERT(after.num_cache_hits == hits_before + 1,
                     "cache hit: num_cache_hits should increment");

        alloc.deallocate(ptr2, nbytes, stream);
        cudaDeviceSynchronize();
        std::printf("  PASS [cache_hit]\n");
    }

    // ------ Stats accuracy ------
    {
        alloc.release_cached();
        cudaDeviceSynchronize();

        AllocatorStats base = alloc.stats();

        size_t nbytes = 16384;
        size_t rounded = round_size(nbytes);
        void* ptr = alloc.allocate(nbytes, stream);

        AllocatorStats after_alloc = alloc.stats();
        FORGE_ASSERT(after_alloc.bytes_allocated == base.bytes_allocated + rounded,
                     "stats: bytes_allocated should increase by rounded size");
        FORGE_ASSERT(after_alloc.num_allocs == base.num_allocs + 1,
                     "stats: num_allocs should increment");

        alloc.deallocate(ptr, nbytes, stream);
        cudaDeviceSynchronize();

        AllocatorStats after_free = alloc.stats();
        FORGE_ASSERT(after_free.bytes_allocated == base.bytes_allocated,
                     "stats: bytes_allocated should return to base after free");
        FORGE_ASSERT(after_free.bytes_cached == base.bytes_cached + rounded,
                     "stats: bytes_cached should increase after free");
        FORGE_ASSERT(after_free.num_frees == base.num_frees + 1,
                     "stats: num_frees should increment");

        // Clean up
        alloc.release_cached();
        cudaDeviceSynchronize();
        std::printf("  PASS [stats_accuracy]\n");
    }

    // ------ Stress test: 1000 alloc/free cycles ------
    {
        alloc.release_cached();
        cudaDeviceSynchronize();
        AllocatorStats before = alloc.stats();

        size_t nbytes = 1024;
        for (int i = 0; i < 1000; i++) {
            void* ptr = alloc.allocate(nbytes, stream);
            FORGE_ASSERT(ptr != nullptr, "stress: alloc should succeed");
            alloc.deallocate(ptr, nbytes, stream);
        }
        cudaDeviceSynchronize();

        AllocatorStats after = alloc.stats();
        // After all alloc/free pairs, bytes_allocated should be same as before
        FORGE_ASSERT(after.bytes_allocated == before.bytes_allocated,
                     "stress: no net allocation increase after 1000 cycles");
        FORGE_ASSERT(after.num_allocs == before.num_allocs + 1000,
                     "stress: num_allocs should increase by 1000");
        FORGE_ASSERT(after.num_frees == before.num_frees + 1000,
                     "stress: num_frees should increase by 1000");

        alloc.release_cached();
        cudaDeviceSynchronize();
        std::printf("  PASS [stress_1000_cycles]\n");
    }

    // ------ release_cached ------
    {
        size_t nbytes = 32768;
        void* ptr = alloc.allocate(nbytes, stream);
        alloc.deallocate(ptr, nbytes, stream);
        cudaDeviceSynchronize();

        AllocatorStats before_release = alloc.stats();
        FORGE_ASSERT(before_release.bytes_cached > 0,
                     "release_cached: should have cached bytes before release");

        alloc.release_cached();
        cudaDeviceSynchronize();

        AllocatorStats after_release = alloc.stats();
        FORGE_ASSERT(after_release.bytes_cached == 0,
                     "release_cached: bytes_cached should be 0 after release");

        std::printf("  PASS [release_cached]\n");
    }

    // ------ Different sizes: cache hits for each ------
    {
        alloc.release_cached();
        cudaDeviceSynchronize();

        size_t sizes[] = {256, 1024, 4096, 16384};
        void* ptrs[4];

        // Allocate all
        for (int i = 0; i < 4; i++) {
            ptrs[i] = alloc.allocate(sizes[i], stream);
            FORGE_ASSERT(ptrs[i] != nullptr, "diff_sizes: alloc should succeed");
        }

        // Free all
        for (int i = 0; i < 4; i++) {
            alloc.deallocate(ptrs[i], sizes[i], stream);
        }
        cudaDeviceSynchronize();

        AllocatorStats before_realloc = alloc.stats();

        // Re-allocate: should all be cache hits returning same pointers
        void* ptrs2[4];
        for (int i = 0; i < 4; i++) {
            ptrs2[i] = alloc.allocate(sizes[i], stream);
        }

        AllocatorStats after_realloc = alloc.stats();
        FORGE_ASSERT(after_realloc.num_cache_hits == before_realloc.num_cache_hits + 4,
                     "diff_sizes: should get 4 cache hits");

        for (int i = 0; i < 4; i++) {
            FORGE_ASSERT(ptrs2[i] == ptrs[i],
                         "diff_sizes: re-alloc should return same pointer");
            alloc.deallocate(ptrs2[i], sizes[i], stream);
        }
        cudaDeviceSynchronize();
        alloc.release_cached();

        std::printf("  PASS [different_sizes]\n");
    }

    std::printf("ALL PASSED\n");
    return 0;
}
