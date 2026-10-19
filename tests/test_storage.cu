// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/storage.h>
#include <forge/allocator.h>
#include "test_helpers.cuh"

using namespace forge;

int main() {
    std::printf("=== Test: Storage ===\n");

    // ------ Create + data valid ------
    {
        cudaStream_t stream = nullptr;
        size_t nbytes = 4096;
        StorageImpl* s = storage_create(nbytes, stream);
        FORGE_ASSERT(s != nullptr, "storage_create should return non-null");
        FORGE_ASSERT(s->data_ != nullptr, "storage data should be non-null");
        FORGE_ASSERT(s->nbytes_ == nbytes, "storage nbytes should match");
        FORGE_ASSERT(s->owns_ == true, "storage should be owning");
        FORGE_ASSERT(s->refcount_.load() == 1, "initial refcount should be 1");
        FORGE_ASSERT(s->version() == 0, "initial version should be 0");

        // Wrap in Storage handle to manage lifetime
        Storage h(s);
        std::printf("  PASS [create_data_valid]\n");
    }

    // ------ Refcount lifecycle ------
    {
        Storage s1(storage_create(1024, nullptr));
        FORGE_ASSERT(s1.use_count() == 1, "refcount: initial use_count should be 1");

        {
            Storage s2 = s1; // copy
            FORGE_ASSERT(s1.use_count() == 2, "refcount: after copy, use_count should be 2");
            FORGE_ASSERT(s2.use_count() == 2, "refcount: copy should share count");
            FORGE_ASSERT(s1.get() == s2.get(), "refcount: copies share same impl");
        }
        // s2 destroyed
        FORGE_ASSERT(s1.use_count() == 1, "refcount: after copy destroyed, use_count should be 1");

        std::printf("  PASS [refcount_lifecycle]\n");
    }

    // ------ Non-owning wrap ------
    {
        // Allocate some device memory manually
        float* d_ptr = nullptr;
        cudaMalloc(&d_ptr, 256);

        StorageImpl* s = storage_wrap(d_ptr, 256, 0);
        FORGE_ASSERT(s->data_ == d_ptr, "wrap: data should match input pointer");
        FORGE_ASSERT(s->owns_ == false, "wrap: should be non-owning");
        FORGE_ASSERT(s->nbytes_ == 256, "wrap: nbytes should match");

        {
            Storage h(s);
            // Storage destructor should NOT free d_ptr since owns_=false
        }

        // Verify pointer is still valid by writing to it
        float val = 99.0f;
        cudaMemcpy(d_ptr, &val, sizeof(float), cudaMemcpyHostToDevice);
        float readback = 0.0f;
        cudaMemcpy(&readback, d_ptr, sizeof(float), cudaMemcpyDeviceToHost);
        FORGE_ASSERT(readback == 99.0f,
                     "wrap: pointer should still be valid after Storage destroyed");

        cudaFree(d_ptr);
        std::printf("  PASS [non_owning_wrap]\n");
    }

    // ------ Version counter ------
    {
        Storage s1(storage_create(512, nullptr));
        FORGE_ASSERT(s1->version() == 0, "version: initial should be 0");

        s1->bump_version();
        FORGE_ASSERT(s1->version() == 1, "version: after bump should be 1");

        s1->bump_version();
        s1->bump_version();
        FORGE_ASSERT(s1->version() == 3, "version: after 3 bumps should be 3");

        // Shared copies see same version
        Storage s2 = s1;
        FORGE_ASSERT(s2->version() == 3, "version: copy should see same version");
        s2->bump_version();
        FORGE_ASSERT(s1->version() == 4, "version: bump via copy visible to original");

        std::printf("  PASS [version_counter]\n");
    }

    // ------ Device field ------
    {
        int current_device = 0;
        cudaGetDevice(&current_device);

        Storage s(storage_create(256, nullptr));
        FORGE_ASSERT(s->device_ == static_cast<int8_t>(current_device),
                     "device: should match current CUDA device");

        std::printf("  PASS [device_field]\n");
    }

    // ------ Owns flag ------
    {
        Storage owning(storage_create(256, nullptr));
        FORGE_ASSERT(owning->owns_ == true, "owns: created storage should own");

        float* d_ptr = nullptr;
        cudaMalloc(&d_ptr, 64);
        Storage wrapped(storage_wrap(d_ptr, 64, 0));
        FORGE_ASSERT(wrapped->owns_ == false, "owns: wrapped storage should not own");

        // Clean up wrapped pointer
        wrapped = Storage(); // release handle (does NOT free d_ptr)
        cudaFree(d_ptr);

        std::printf("  PASS [owns_flag]\n");
    }

    std::printf("ALL PASSED\n");
    return 0;
}
