// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/rng.cuh>
#include "test_helpers.cuh"
#include <cmath>
#include <numeric>

using namespace forge;

int main() {
    std::printf("=== Test: RNG ===\n");

    // ------ fill_normal: check mean and std ------
    {
        const int N = 100000;
        float* d_data = test::to_device_zeroed<float>(N);

        nn::fill_normal(d_data, N, 0.0f, 1.0f, /*seed=*/42, /*offset=*/0);
        cudaDeviceSynchronize();

        auto h = test::to_host(d_data, N);

        // Compute empirical mean and std
        double mean = 0.0;
        for (int i = 0; i < N; i++) mean += h[i];
        mean /= N;

        double var = 0.0;
        for (int i = 0; i < N; i++) var += (h[i] - mean) * (h[i] - mean);
        var /= (N - 1);
        double std_dev = std::sqrt(var);

        std::printf("  fill_normal: mean=%.4f (expect ~0.0) std=%.4f (expect ~1.0)\n",
                    mean, std_dev);
        FORGE_ASSERT(std::abs(mean) < 0.02, "fill_normal mean too far from 0");
        FORGE_ASSERT(std::abs(std_dev - 1.0) < 0.02, "fill_normal std too far from 1");
        std::printf("  PASS [fill_normal]\n");

        test::free_device(d_data);
    }

    // ------ fill_normal: non-zero mean, custom std ------
    {
        const int N = 50000;
        float* d_data = test::to_device_zeroed<float>(N);

        nn::fill_normal(d_data, N, 5.0f, 0.5f, /*seed=*/99, /*offset=*/0);
        cudaDeviceSynchronize();

        auto h = test::to_host(d_data, N);

        double mean = 0.0;
        for (int i = 0; i < N; i++) mean += h[i];
        mean /= N;

        double var = 0.0;
        for (int i = 0; i < N; i++) var += (h[i] - mean) * (h[i] - mean);
        var /= (N - 1);
        double std_dev = std::sqrt(var);

        std::printf("  fill_normal custom: mean=%.4f (expect ~5.0) std=%.4f (expect ~0.5)\n",
                    mean, std_dev);
        FORGE_ASSERT(std::abs(mean - 5.0) < 0.02, "fill_normal mean too far from 5");
        FORGE_ASSERT(std::abs(std_dev - 0.5) < 0.02, "fill_normal std too far from 0.5");
        std::printf("  PASS [fill_normal_custom]\n");

        test::free_device(d_data);
    }

    // ------ fill_uniform: check range ------
    {
        const int N = 50000;
        float* d_data = test::to_device_zeroed<float>(N);

        nn::fill_uniform(d_data, N, 2.0f, 5.0f, /*seed=*/77, /*offset=*/0);
        cudaDeviceSynchronize();

        auto h = test::to_host(d_data, N);

        float lo = h[0], hi = h[0];
        double mean = 0.0;
        for (int i = 0; i < N; i++) {
            if (h[i] < lo) lo = h[i];
            if (h[i] > hi) hi = h[i];
            mean += h[i];
        }
        mean /= N;

        std::printf("  fill_uniform: min=%.4f max=%.4f mean=%.4f (expect ~3.5)\n",
                    lo, hi, mean);
        FORGE_ASSERT(lo >= 2.0f, "fill_uniform below lower bound");
        FORGE_ASSERT(hi <= 5.0f, "fill_uniform above upper bound");
        FORGE_ASSERT(std::abs(mean - 3.5) < 0.05, "fill_uniform mean too far from 3.5");
        std::printf("  PASS [fill_uniform]\n");

        test::free_device(d_data);
    }

    // ------ Determinism: same seed+offset = same output ------
    {
        const int N = 1024;
        float* d_a = test::to_device_zeroed<float>(N);
        float* d_b = test::to_device_zeroed<float>(N);

        nn::fill_normal(d_a, N, 0.0f, 1.0f, 42, 0);
        nn::fill_normal(d_b, N, 0.0f, 1.0f, 42, 0);
        cudaDeviceSynchronize();

        auto ha = test::to_host(d_a, N);
        auto hb = test::to_host(d_b, N);

        FORGE_ASSERT(test::check_arrays(ha, hb, 0.0f, 0.0f, "determinism"),
                     "RNG not deterministic with same seed+offset");

        test::free_device(d_a);
        test::free_device(d_b);
    }

    // ------ Different offset = different output ------
    {
        const int N = 1024;
        float* d_a = test::to_device_zeroed<float>(N);
        float* d_b = test::to_device_zeroed<float>(N);

        nn::fill_normal(d_a, N, 0.0f, 1.0f, 42, 0);
        nn::fill_normal(d_b, N, 0.0f, 1.0f, 42, N); // different offset
        cudaDeviceSynchronize();

        auto ha = test::to_host(d_a, N);
        auto hb = test::to_host(d_b, N);

        // They should differ
        bool all_same = true;
        for (int i = 0; i < N && all_same; i++)
            if (ha[i] != hb[i]) all_same = false;

        FORGE_ASSERT(!all_same, "Different offsets produced same output");
        std::printf("  PASS [different_offset]\n");

        test::free_device(d_a);
        test::free_device(d_b);
    }

    std::printf("ALL PASSED\n");
    return 0;
}
