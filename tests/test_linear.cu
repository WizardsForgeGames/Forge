// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/linear.cuh>
#include "test_helpers.cuh"
#include <cmath>

using namespace forge;

// CPU reference: Y[B,N] = X[B,K] @ W[N,K]^T + bias[N]
void linear_fwd_cpu(
    const float* X, const float* W, const float* bias, float* Y,
    int B, int N, int K)
{
    for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
            float sum = (bias) ? bias[n] : 0.0f;
            for (int k = 0; k < K; k++)
                sum += X[b * K + k] * W[n * K + k];
            Y[b * N + n] = sum;
        }
    }
}

// CPU reference: dX[B,K] = dY[B,N] @ W[N,K]
void linear_bwd_data_cpu(
    const float* dY, const float* W, float* dX,
    int B, int N, int K)
{
    for (int b = 0; b < B; b++)
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            for (int n = 0; n < N; n++)
                sum += dY[b * N + n] * W[n * K + k];
            dX[b * K + k] = sum;
        }
}

// CPU reference: dW[N,K] = dY[B,N]^T @ X[B,K]
void linear_bwd_weight_cpu(
    const float* dY, const float* X, float* dW,
    int B, int N, int K, float beta)
{
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            for (int b = 0; b < B; b++)
                sum += dY[b * N + n] * X[b * K + k];
            dW[n * K + k] = beta * dW[n * K + k] + sum;
        }
}

int main() {
    std::printf("=== Test: Linear ===\n");

    const int B = 32, N = 64, K = 16;

    std::vector<float> h_X(B * K), h_W(N * K), h_bias(N), h_dY(B * N);
    test::fill_random(h_X, -1.0f, 1.0f, 42);
    test::fill_random(h_W, -0.5f, 0.5f, 43);
    test::fill_random(h_bias, -0.1f, 0.1f, 44);
    test::fill_random(h_dY, -1.0f, 1.0f, 45);

    float* d_X    = test::to_device(h_X);
    float* d_W    = test::to_device(h_W);
    float* d_bias = test::to_device(h_bias);
    float* d_Y    = test::to_device_zeroed<float>(B * N);
    float* d_dY   = test::to_device(h_dY);
    float* d_dX   = test::to_device_zeroed<float>(B * K);
    float* d_dW   = test::to_device_zeroed<float>(N * K);

    // ------ Forward with bias ------
    {
        nn::linear_forward(d_X, d_W, d_bias, d_Y, B, N, K);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(d_Y, B * N);
        std::vector<float> ref(B * N);
        linear_fwd_cpu(h_X.data(), h_W.data(), h_bias.data(), ref.data(), B, N, K);
        FORGE_ASSERT(test::check_arrays(gpu, ref, 1e-3f, 1e-2f, "linear_fwd_bias"),
                     "linear_forward with bias mismatch");
    }

    // ------ Forward without bias ------
    {
        nn::linear_forward(d_X, d_W, nullptr, d_Y, B, N, K);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(d_Y, B * N);
        std::vector<float> ref(B * N);
        linear_fwd_cpu(h_X.data(), h_W.data(), nullptr, ref.data(), B, N, K);
        FORGE_ASSERT(test::check_arrays(gpu, ref, 1e-3f, 1e-2f, "linear_fwd_nobias"),
                     "linear_forward no bias mismatch");
    }

    // ------ Backward data ------
    {
        nn::linear_backward_data(d_dY, d_W, d_dX, B, N, K);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(d_dX, B * K);
        std::vector<float> ref(B * K);
        linear_bwd_data_cpu(h_dY.data(), h_W.data(), ref.data(), B, N, K);
        FORGE_ASSERT(test::check_arrays(gpu, ref, 1e-3f, 1e-2f, "linear_bwd_data"),
                     "linear_backward_data mismatch");
    }

    // ------ Backward weight (overwrite) ------
    {
        nn::linear_backward_weight(d_dY, d_X, d_dW, B, N, K, 0.0f);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(d_dW, N * K);
        std::vector<float> ref(N * K, 0.0f);
        linear_bwd_weight_cpu(h_dY.data(), h_X.data(), ref.data(), B, N, K, 0.0f);
        FORGE_ASSERT(test::check_arrays(gpu, ref, 1e-2f, 1e-1f, "linear_bwd_weight"),
                     "linear_backward_weight mismatch");
    }

    // ------ Backward weight (accumulate) ------
    {
        // Pre-fill d_dW with known values
        std::vector<float> h_dW_init(N * K, 1.0f);
        float* d_dW2 = test::to_device(h_dW_init);

        nn::linear_backward_weight(d_dY, d_X, d_dW2, B, N, K, 1.0f);
        cudaDeviceSynchronize();

        auto gpu = test::to_host(d_dW2, N * K);
        std::vector<float> ref(N * K, 1.0f);
        linear_bwd_weight_cpu(h_dY.data(), h_X.data(), ref.data(), B, N, K, 1.0f);
        FORGE_ASSERT(test::check_arrays(gpu, ref, 1e-2f, 1e-1f, "linear_bwd_weight_acc"),
                     "linear_backward_weight accumulate mismatch");

        test::free_device(d_dW2);
    }

    test::free_device(d_X);
    test::free_device(d_W);
    test::free_device(d_bias);
    test::free_device(d_Y);
    test::free_device(d_dY);
    test::free_device(d_dX);
    test::free_device(d_dW);

    std::printf("ALL PASSED\n");
    return 0;
}
