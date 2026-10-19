// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/bias.cuh>
#include "test_helpers.cuh"

using namespace forge;

// CPU reference: sum d_upstream[i * B + b] over b for each i
void bias_grad_cpu(const float* upstream, float* bias, int N, int B) {
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int b = 0; b < B; b++)
            sum += upstream[i * B + b];
        bias[i] += sum;  // accumulate
    }
}

int main() {
    std::printf("=== Test: Bias Grad Accumulate ===\n");
    const int N = 512, B = 64;

    std::vector<float> h_upstream(N * B);
    test::fill_random(h_upstream, -1.0f, 1.0f, 42);

    // CPU reference
    std::vector<float> ref_bias(N, 0.0f);
    bias_grad_cpu(h_upstream.data(), ref_bias.data(), N, B);

    // GPU
    float* d_upstream = test::to_device(h_upstream);
    float* d_bias = test::to_device_zeroed<float>(N);

    nn::bias_grad_accumulate(d_upstream, d_bias, N, B);
    cudaDeviceSynchronize();

    auto gpu_bias = test::to_host(d_bias, N);
    FORGE_ASSERT(test::check_arrays(gpu_bias, ref_bias, 1e-3f, 1e-3f, "bias_grad"),
                 "bias_grad_accumulate mismatch");

    // Test accumulation (call again, CPU ref doubles)
    nn::bias_grad_accumulate(d_upstream, d_bias, N, B);
    cudaDeviceSynchronize();

    std::vector<float> ref_bias2(N, 0.0f);
    bias_grad_cpu(h_upstream.data(), ref_bias2.data(), N, B);
    for (int i = 0; i < N; i++) ref_bias2[i] *= 2.0f;

    auto gpu_bias2 = test::to_host(d_bias, N);
    FORGE_ASSERT(test::check_arrays(gpu_bias2, ref_bias2, 1e-3f, 1e-3f, "bias_grad_accum"),
                 "bias_grad_accumulate accumulation mismatch");

    test::free_device(d_upstream);
    test::free_device(d_bias);
    std::printf("ALL PASSED\n");
    return 0;
}
