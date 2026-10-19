// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/mse.cuh>
#include "test_helpers.cuh"
#include <cmath>

using namespace forge;

int main() {
    std::printf("=== Test: MSE Loss ===\n");

    // ------ MSE forward ------
    {
        const int N = 1000;
        std::vector<float> h_pred(N), h_target(N);
        test::fill_random(h_pred, -2.0f, 2.0f, 42);
        test::fill_random(h_target, -2.0f, 2.0f, 43);

        float* d_pred   = test::to_device(h_pred);
        float* d_target = test::to_device(h_target);
        float* d_loss   = test::to_device_zeroed<float>(1);

        nn::mse_forward(d_pred, d_target, d_loss, N);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(d_loss, 1);

        // CPU reference
        double ref = 0.0;
        for (int i = 0; i < N; i++) {
            double d = (double)h_pred[i] - (double)h_target[i];
            ref += d * d;
        }
        ref /= N;

        float diff = std::abs(gpu[0] - (float)ref);
        std::printf("  mse_forward: gpu=%.8f ref=%.8f diff=%.2e\n",
                    gpu[0], (float)ref, diff);
        FORGE_ASSERT(diff < 1e-3f, "mse_forward mismatch");
        std::printf("  PASS [mse_forward]\n");

        test::free_device(d_pred);
        test::free_device(d_target);
        test::free_device(d_loss);
    }

    // ------ MSE forward (large, two-pass path) ------
    {
        const int N = 100000;
        std::vector<float> h_pred(N), h_target(N);
        test::fill_random(h_pred, -1.0f, 1.0f, 55);
        test::fill_random(h_target, -1.0f, 1.0f, 56);

        float* d_pred   = test::to_device(h_pred);
        float* d_target = test::to_device(h_target);
        float* d_loss   = test::to_device_zeroed<float>(1);

        nn::mse_forward(d_pred, d_target, d_loss, N);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(d_loss, 1);

        double ref = 0.0;
        for (int i = 0; i < N; i++) {
            double d = (double)h_pred[i] - (double)h_target[i];
            ref += d * d;
        }
        ref /= N;

        float diff = std::abs(gpu[0] - (float)ref);
        std::printf("  mse_forward_large: gpu=%.8f ref=%.8f diff=%.2e\n",
                    gpu[0], (float)ref, diff);
        FORGE_ASSERT(diff < 1e-2f, "mse_forward large mismatch");
        std::printf("  PASS [mse_forward_large]\n");

        test::free_device(d_pred);
        test::free_device(d_target);
        test::free_device(d_loss);
    }

    // ------ MSE backward ------
    {
        const int N = 2048;
        std::vector<float> h_pred(N), h_target(N);
        test::fill_random(h_pred, -2.0f, 2.0f, 42);
        test::fill_random(h_target, -2.0f, 2.0f, 43);

        float* d_pred   = test::to_device(h_pred);
        float* d_target = test::to_device(h_target);
        float* d_dpred  = test::to_device_zeroed<float>(N);

        nn::mse_backward(d_pred, d_target, d_dpred, N, 1.0f);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(d_dpred, N);

        // CPU reference: d_pred[i] = (2/n) * (pred[i] - target[i])
        std::vector<float> ref(N);
        float coeff = 2.0f / (float)N;
        for (int i = 0; i < N; i++)
            ref[i] = coeff * (h_pred[i] - h_target[i]);

        FORGE_ASSERT(test::check_arrays(gpu, ref, 1e-5f, 1e-4f, "mse_backward"),
                     "mse_backward mismatch");

        test::free_device(d_pred);
        test::free_device(d_target);
        test::free_device(d_dpred);
    }

    // ------ MSE backward with scale ------
    {
        const int N = 512;
        std::vector<float> h_pred(N), h_target(N);
        test::fill_random(h_pred, -1.0f, 1.0f, 77);
        test::fill_random(h_target, -1.0f, 1.0f, 78);

        float* d_pred   = test::to_device(h_pred);
        float* d_target = test::to_device(h_target);
        float* d_dpred  = test::to_device_zeroed<float>(N);

        float scale = 0.5f;
        nn::mse_backward(d_pred, d_target, d_dpred, N, scale);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(d_dpred, N);

        std::vector<float> ref(N);
        float coeff = scale * 2.0f / (float)N;
        for (int i = 0; i < N; i++)
            ref[i] = coeff * (h_pred[i] - h_target[i]);

        FORGE_ASSERT(test::check_arrays(gpu, ref, 1e-5f, 1e-4f, "mse_backward_scaled"),
                     "mse_backward scaled mismatch");

        test::free_device(d_pred);
        test::free_device(d_target);
        test::free_device(d_dpred);
    }

    std::printf("ALL PASSED\n");
    return 0;
}
