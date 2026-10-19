// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Integration test: existing Forge kernels through new Tensor .data<float>() interface.
#include <forge/tensor.h>
#include <forge/allocator.h>
#include <forge/nn/mse.cuh>
#include "test_helpers.cuh"
#include <cmath>

using namespace forge;

int main() {
    std::printf("=== Test: Integration (Tensor + existing kernels) ===\n");

    // ------ MSE via Tensor ------
    {
        const int N = 1000;
        std::vector<float> h_pred(N), h_target(N);
        test::fill_random(h_pred, -2.0f, 2.0f, 42);
        test::fill_random(h_target, -2.0f, 2.0f, 43);

        // Use Tensor factories
        Tensor t_pred = Tensor::empty({N});
        Tensor t_target = Tensor::empty({N});
        Tensor t_loss = Tensor::zeros({1});

        // Upload data via raw pointers
        cudaMemcpy(t_pred.data<float>(), h_pred.data(),
                   N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(t_target.data<float>(), h_target.data(),
                   N * sizeof(float), cudaMemcpyHostToDevice);

        // Call existing kernel via .data<float>()
        nn::mse_forward(t_pred.data<float>(), t_target.data<float>(),
                        t_loss.data<float>(), N);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(t_loss.data<float>(), 1);

        // CPU reference
        double ref = 0.0;
        for (int i = 0; i < N; i++) {
            double d = (double)h_pred[i] - (double)h_target[i];
            ref += d * d;
        }
        ref /= N;

        float diff = std::abs(gpu[0] - (float)ref);
        std::printf("  mse_via_tensor: gpu=%.8f ref=%.8f diff=%.2e\n",
                    gpu[0], (float)ref, diff);
        FORGE_ASSERT(diff < 1e-3f, "mse_via_tensor: mismatch");
        std::printf("  PASS [mse_via_tensor]\n");
    }

    // ------ Fill + readback via Tensor ------
    {
        Tensor t = Tensor::ones({256});
        cudaDeviceSynchronize();

        auto h = test::to_host(t.data<float>(), 256);
        bool all_one = true;
        for (int i = 0; i < 256; i++) {
            if (h[i] != 1.0f) { all_one = false; break; }
        }
        FORGE_ASSERT(all_one, "fill_readback: all elements should be 1.0");
        std::printf("  PASS [fill_readback]\n");
    }

    // ------ Allocator reuse through Tensor lifecycle ------
    {
        auto& alloc = CachingAllocator::instance(0);
        alloc.release_cached();
        cudaDeviceSynchronize();
        AllocatorStats before = alloc.stats();

        void* first_ptr;
        {
            Tensor t = Tensor::empty({1024});
            first_ptr = t.data_ptr();
            // t destroyed here -> memory returned to allocator cache
        }
        cudaDeviceSynchronize();

        AllocatorStats after_destroy = alloc.stats();
        FORGE_ASSERT(after_destroy.bytes_cached > before.bytes_cached,
                     "alloc_reuse: destroying tensor should cache memory");

        // New tensor of same size should reuse cached block
        AllocatorStats before_realloc = alloc.stats();
        {
            Tensor t2 = Tensor::empty({1024});
            AllocatorStats after_realloc = alloc.stats();
            FORGE_ASSERT(after_realloc.num_cache_hits > before_realloc.num_cache_hits,
                         "alloc_reuse: re-alloc should be cache hit");
            FORGE_ASSERT(t2.data_ptr() == first_ptr,
                         "alloc_reuse: should get same pointer from cache");
        }

        alloc.release_cached();
        cudaDeviceSynchronize();
        std::printf("  PASS [allocator_reuse]\n");
    }

    std::printf("ALL PASSED\n");
    return 0;
}
