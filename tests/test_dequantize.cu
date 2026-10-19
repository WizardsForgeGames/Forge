// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/quant/dequantize.cuh>
#include "test_helpers.cuh"
#include <cmath>
#include <cuda_fp16.h>

using namespace forge;

int main() {
    std::printf("=== Test: Dequantize Bias ===\n");
    const int M = 16, N = 64;

    // Generate random INT32 accumulator values
    std::vector<int32_t> h_accum(M * N);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(-50000, 50000);
    for (auto& v : h_accum) v = dist(rng);

    // Generate random scales and bias
    std::vector<float> h_act_scale(M);
    test::fill_random(h_act_scale, 0.001f, 0.1f, 43);

    std::vector<float> h_w_scale_f(N);
    test::fill_random(h_w_scale_f, 0.001f, 0.1f, 44);
    std::vector<__half> h_w_scale(N);
    for (int i = 0; i < N; i++) h_w_scale[i] = __float2half(h_w_scale_f[i]);

    std::vector<float> h_bias_f(N);
    test::fill_random(h_bias_f, -0.5f, 0.5f, 45);
    std::vector<__half> h_bias(N);
    for (int i = 0; i < N; i++) h_bias[i] = __float2half(h_bias_f[i]);

    // CPU reference: output = accum * act_scale[m] * w_scale[n] + bias[n]
    std::vector<float> ref_output(M * N);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float val = (float)h_accum[m * N + n] * h_act_scale[m]
                        * __half2float(h_w_scale[n]);
            val += __half2float(h_bias[n]);
            ref_output[m * N + n] = val;
        }

    // GPU: with bias
    int32_t* d_accum;
    cudaMalloc(&d_accum, M * N * sizeof(int32_t));
    cudaMemcpy(d_accum, h_accum.data(), M * N * sizeof(int32_t), cudaMemcpyHostToDevice);
    float* d_act_scale = test::to_device(h_act_scale);
    __half* d_w_scale;
    cudaMalloc(&d_w_scale, N * sizeof(__half));
    cudaMemcpy(d_w_scale, h_w_scale.data(), N * sizeof(__half), cudaMemcpyHostToDevice);
    __half* d_bias;
    cudaMalloc(&d_bias, N * sizeof(__half));
    cudaMemcpy(d_bias, h_bias.data(), N * sizeof(__half), cudaMemcpyHostToDevice);
    __half* d_output;
    cudaMalloc(&d_output, M * N * sizeof(__half));

    quant::fused_dequantize_bias(d_accum, d_act_scale, d_w_scale, d_bias, d_output, M, N);
    cudaDeviceSynchronize();

    std::vector<__half> gpu_output_h(M * N);
    cudaMemcpy(gpu_output_h.data(), d_output, M * N * sizeof(__half), cudaMemcpyDeviceToHost);

    // Compare (FP16 precision)
    float max_diff = 0.0f;
    int fails = 0;
    for (int i = 0; i < M * N; i++) {
        float gpu_val = __half2float(gpu_output_h[i]);
        float diff = std::abs(gpu_val - ref_output[i]);
        float rtol = diff / (std::abs(ref_output[i]) + 1e-4f);
        if (diff > max_diff) max_diff = diff;
        if (rtol > 0.05f && diff > 0.1f) fails++;
    }
    std::printf("  with_bias: max_diff=%.4f fails=%d/%d\n", max_diff, fails, M * N);
    FORGE_ASSERT(fails == 0, "dequantize with bias mismatch");
    std::printf("  PASS [with bias]\n");

    // GPU: without bias (nullptr)
    quant::fused_dequantize_bias(d_accum, d_act_scale, d_w_scale, nullptr, d_output, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(gpu_output_h.data(), d_output, M * N * sizeof(__half), cudaMemcpyDeviceToHost);
    fails = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float expected = (float)h_accum[m * N + n] * h_act_scale[m]
                             * __half2float(h_w_scale[n]);
            float gpu_val = __half2float(gpu_output_h[m * N + n]);
            float diff = std::abs(gpu_val - expected);
            float rtol = diff / (std::abs(expected) + 1e-4f);
            if (rtol > 0.05f && diff > 0.1f) fails++;
        }
    FORGE_ASSERT(fails == 0, "dequantize without bias mismatch");
    std::printf("  PASS [no bias]\n");

    cudaFree(d_accum);
    test::free_device(d_act_scale);
    cudaFree(d_w_scale);
    cudaFree(d_bias);
    cudaFree(d_output);
    std::printf("ALL PASSED\n");
    return 0;
}
