// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/reduce.cuh>
#include "test_helpers.cuh"
#include <cmath>
#include <numeric>

using namespace forge;

int main() {
    std::printf("=== Test: Reduce ===\n");

    // ------ reduce_sum ------
    {
        const int N = 10007; // non-power-of-2 to test edge cases
        std::vector<float> h_x(N);
        test::fill_random(h_x, -1.0f, 1.0f, 42);

        float* d_x = test::to_device(h_x);
        float* d_out = test::to_device_zeroed<float>(1);

        nn::reduce_sum(d_x, d_out, N);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(d_out, 1);

        // CPU reference (Kahan summation for accuracy)
        double ref_sum = 0.0;
        double c = 0.0;
        for (int i = 0; i < N; i++) {
            double y = (double)h_x[i] - c;
            double t = ref_sum + y;
            c = (t - ref_sum) - y;
            ref_sum = t;
        }

        float diff = std::abs(gpu[0] - (float)ref_sum);
        std::printf("  reduce_sum: gpu=%.6f ref=%.6f diff=%.2e\n",
                    gpu[0], (float)ref_sum, diff);
        FORGE_ASSERT(diff < 0.05f, "reduce_sum mismatch");
        std::printf("  PASS [reduce_sum]\n");

        test::free_device(d_x);
        test::free_device(d_out);
    }

    // ------ reduce_sum (small, single-pass) ------
    {
        const int N = 128;
        std::vector<float> h_x(N);
        test::fill_random(h_x, -1.0f, 1.0f, 99);

        float* d_x = test::to_device(h_x);
        float* d_out = test::to_device_zeroed<float>(1);

        nn::reduce_sum(d_x, d_out, N);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(d_out, 1);
        float ref = 0.0f;
        for (int i = 0; i < N; i++) ref += h_x[i];

        float diff = std::abs(gpu[0] - ref);
        FORGE_ASSERT(diff < 1e-3f, "reduce_sum small mismatch");
        std::printf("  PASS [reduce_sum_small]\n");

        test::free_device(d_x);
        test::free_device(d_out);
    }

    // ------ reduce_mean_dim0 ------
    {
        const int D0 = 20, D1 = 100;
        std::vector<float> h_x(D0 * D1);
        test::fill_random(h_x, -1.0f, 1.0f, 44);

        float* d_x = test::to_device(h_x);
        float* d_out = test::to_device_zeroed<float>(D1);

        nn::reduce_mean_dim0(d_x, d_out, D0, D1);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(d_out, D1);

        // CPU reference
        std::vector<float> ref(D1, 0.0f);
        for (int i = 0; i < D0; i++)
            for (int j = 0; j < D1; j++)
                ref[j] += h_x[i * D1 + j];
        for (int j = 0; j < D1; j++)
            ref[j] /= (float)D0;

        FORGE_ASSERT(test::check_arrays(gpu, ref, 1e-5f, 1e-4f, "reduce_mean_dim0"),
                     "reduce_mean_dim0 mismatch");

        test::free_device(d_x);
        test::free_device(d_out);
    }

    // ------ reduce_sum_dim0 ------
    {
        const int D0 = 50, D1 = 64;
        std::vector<float> h_x(D0 * D1);
        test::fill_random(h_x, -2.0f, 2.0f, 45);

        float* d_x = test::to_device(h_x);
        float* d_out = test::to_device_zeroed<float>(D1);

        nn::reduce_sum_dim0(d_x, d_out, D0, D1);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(d_out, D1);

        std::vector<float> ref(D1, 0.0f);
        for (int i = 0; i < D0; i++)
            for (int j = 0; j < D1; j++)
                ref[j] += h_x[i * D1 + j];

        FORGE_ASSERT(test::check_arrays(gpu, ref, 1e-4f, 1e-3f, "reduce_sum_dim0"),
                     "reduce_sum_dim0 mismatch");

        test::free_device(d_x);
        test::free_device(d_out);
    }

    // ------ reduce_sumsq ------
    {
        const int N = 10007;
        std::vector<float> h_x(N);
        test::fill_random(h_x, -2.0f, 2.0f, 77);

        float* d_x = test::to_device(h_x);
        float* d_out = test::to_device_zeroed<float>(1);

        nn::reduce_sumsq(d_x, d_out, N);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(d_out, 1);

        double ref = 0.0;
        for (int i = 0; i < N; i++)
            ref += (double)h_x[i] * (double)h_x[i];

        float diff = std::abs(gpu[0] - (float)ref);
        float rel = diff / (float)ref;
        std::printf("  reduce_sumsq: gpu=%.6f ref=%.6f rel=%.2e\n",
                    gpu[0], (float)ref, rel);
        FORGE_ASSERT(rel < 1e-4f, "reduce_sumsq mismatch");
        std::printf("  PASS [reduce_sumsq]\n");

        test::free_device(d_x);
        test::free_device(d_out);
    }

    // ------ reduce_absmax ------
    {
        const int N = 10007;
        std::vector<float> h_x(N);
        test::fill_random(h_x, -5.0f, 5.0f, 88);
        // Plant a known max
        h_x[4321] = -7.5f;

        float* d_x = test::to_device(h_x);
        float* d_out = test::to_device_zeroed<float>(1);

        nn::reduce_absmax(d_x, d_out, N);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(d_out, 1);

        float ref = 0.0f;
        for (int i = 0; i < N; i++)
            ref = std::max(ref, std::abs(h_x[i]));

        std::printf("  reduce_absmax: gpu=%.6f ref=%.6f\n", gpu[0], ref);
        FORGE_ASSERT(gpu[0] == ref, "reduce_absmax mismatch");
        std::printf("  PASS [reduce_absmax]\n");

        test::free_device(d_x);
        test::free_device(d_out);
    }

    // ------ reduce_sumsq (small, single-pass) ------
    {
        const int N = 64;
        std::vector<float> h_x(N);
        test::fill_random(h_x, -1.0f, 1.0f, 55);

        float* d_x = test::to_device(h_x);
        float* d_out = test::to_device_zeroed<float>(1);

        nn::reduce_sumsq(d_x, d_out, N);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(d_out, 1);
        float ref = 0.0f;
        for (int i = 0; i < N; i++) ref += h_x[i] * h_x[i];

        float diff = std::abs(gpu[0] - ref);
        FORGE_ASSERT(diff < 1e-3f, "reduce_sumsq small mismatch");
        std::printf("  PASS [reduce_sumsq_small]\n");

        test::free_device(d_x);
        test::free_device(d_out);
    }

    std::printf("ALL PASSED\n");
    return 0;
}
