// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/gemm/int8_dense.cuh>
#include "test_helpers.cuh"
#include <cuda_fp16.h>
#include <cmath>

using namespace forge;

int main() {
    std::printf("=== Test: INT8 Dense GEMM ===\n");
    // Small problem: M=32, K=128, N=64
    const int M = 32, K = 128, N = 64;

    // Generate random INT8 matrices
    std::vector<int8_t> h_A(M * K), h_B(N * K);
    test::fill_random_int8(h_A, 42);
    test::fill_random_int8(h_B, 43);

    // Random scales and bias
    std::vector<float> h_act_scale(M);
    test::fill_random(h_act_scale, 0.001f, 0.05f, 44);
    std::vector<float> h_w_scale_f(N);
    test::fill_random(h_w_scale_f, 0.001f, 0.05f, 45);
    std::vector<__half> h_w_scale(N);
    for (int i = 0; i < N; i++) h_w_scale[i] = __float2half(h_w_scale_f[i]);
    std::vector<float> h_bias_f(N);
    test::fill_random(h_bias_f, -0.5f, 0.5f, 46);
    std::vector<__half> h_bias(N);
    for (int i = 0; i < N; i++) h_bias[i] = __float2half(h_bias_f[i]);

    // CPU reference: C = A(M,K) * B(N,K)^T -> (M,N)
    // Then dequant: output[m,n] = C[m,n] * act_scale[m] * w_scale[n] + bias[n]
    std::vector<float> ref_output(M * N);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)h_A[m * K + k] * (int32_t)h_B[n * K + k];
            float val = (float)acc * h_act_scale[m] * __half2float(h_w_scale[n]);
            val += __half2float(h_bias[n]);
            ref_output[m * N + n] = val;
        }
    }

    // GPU
    int8_t* d_A = test::to_device(h_A);
    int8_t* d_B = test::to_device(h_B);
    float* d_act_scale = test::to_device(h_act_scale);
    __half* d_w_scale;
    cudaMalloc(&d_w_scale, N * sizeof(__half));
    cudaMemcpy(d_w_scale, h_w_scale.data(), N * sizeof(__half), cudaMemcpyHostToDevice);
    __half* d_bias;
    cudaMalloc(&d_bias, N * sizeof(__half));
    cudaMemcpy(d_bias, h_bias.data(), N * sizeof(__half), cudaMemcpyHostToDevice);
    __half* d_output;
    cudaMalloc(&d_output, M * N * sizeof(__half));

    gemm::int8_gemm_dequant(d_A, d_B, d_act_scale, d_w_scale, d_bias,
                            d_output, M, K, N);
    cudaDeviceSynchronize();

    // Read back
    std::vector<__half> gpu_output_h(M * N);
    cudaMemcpy(gpu_output_h.data(), d_output, M * N * sizeof(__half), cudaMemcpyDeviceToHost);

    // Compare
    float max_diff = 0.0f;
    float mean_rel_err = 0.0f;
    int fails = 0;
    for (int i = 0; i < M * N; i++) {
        float gpu_val = __half2float(gpu_output_h[i]);
        float diff = std::abs(gpu_val - ref_output[i]);
        float rel_err = diff / (std::abs(ref_output[i]) + 1e-4f);
        if (diff > max_diff) max_diff = diff;
        mean_rel_err += rel_err;
        // Generous tolerance for INT8 GEMM + FP16 dequant
        if (rel_err > 0.1f && diff > 0.5f) fails++;
    }
    mean_rel_err /= (M * N);

    std::printf("  max_diff=%.4f mean_rel_err=%.4f fails=%d/%d\n",
                max_diff, mean_rel_err, fails, M * N);
    FORGE_ASSERT(fails == 0, "INT8 dense GEMM output mismatch");
    FORGE_ASSERT(mean_rel_err < 0.03f, "INT8 dense GEMM mean relative error too high");
    std::printf("  PASS [int8_gemm_dequant]\n");

    test::free_device(d_A);
    test::free_device(d_B);
    test::free_device(d_act_scale);
    cudaFree(d_w_scale);
    cudaFree(d_bias);
    cudaFree(d_output);
    std::printf("ALL PASSED\n");
    return 0;
}
