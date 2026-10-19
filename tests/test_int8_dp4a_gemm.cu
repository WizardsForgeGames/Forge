// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Tests dp4a INT8 GEMM kernel and unified dispatch.
// Validates against CPU reference with fused dequant + bias.
#include <forge/gemm/int8_dp4a.cuh>
#include <forge/gemm/int8_gemm.cuh>
#include "test_helpers.cuh"
#include <cuda_fp16.h>
#include <cmath>

using namespace forge;

// CPU reference: C[m,n] = sum_k A[m,k]*B[n,k] * act_scale[m] * w_scale[n] + bias[n]
static void cpu_int8_gemm_dequant(
    const int8_t* A, const int8_t* B,
    const float* act_scale, const __half* w_scale, const __half* bias,
    float* output, int M, int K, int N)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[m * K + k] * (int32_t)B[n * K + k];
            float val = (float)acc * act_scale[m] * __half2float(w_scale[n]);
            if (bias) val += __half2float(bias[n]);
            output[m * N + n] = val;
        }
    }
}

static int test_dp4a_gemm(int M, int K, int N, bool use_bias, const char* label) {
    std::printf("  %s (M=%d K=%d N=%d bias=%d)...\n", label, M, K, N, use_bias);

    std::vector<int8_t> h_A(M * K), h_B(N * K);
    test::fill_random_int8(h_A, 42);
    test::fill_random_int8(h_B, 43);

    std::vector<float> h_act_scale(M);
    test::fill_random(h_act_scale, 0.001f, 0.05f, 44);
    std::vector<float> h_w_scale_f(N);
    test::fill_random(h_w_scale_f, 0.001f, 0.05f, 45);
    std::vector<__half> h_w_scale(N);
    for (int i = 0; i < N; i++) h_w_scale[i] = __float2half(h_w_scale_f[i]);

    std::vector<__half> h_bias(N);
    if (use_bias) {
        std::vector<float> h_bias_f(N);
        test::fill_random(h_bias_f, -0.5f, 0.5f, 46);
        for (int i = 0; i < N; i++) h_bias[i] = __float2half(h_bias_f[i]);
    }

    // CPU reference
    std::vector<float> ref_output(M * N);
    cpu_int8_gemm_dequant(h_A.data(), h_B.data(), h_act_scale.data(),
                          h_w_scale.data(), use_bias ? h_bias.data() : nullptr,
                          ref_output.data(), M, K, N);

    // GPU
    int8_t* d_A = test::to_device(h_A);
    int8_t* d_B = test::to_device(h_B);
    float* d_act_scale = test::to_device(h_act_scale);
    __half* d_w_scale;
    cudaMalloc(&d_w_scale, N * sizeof(__half));
    cudaMemcpy(d_w_scale, h_w_scale.data(), N * sizeof(__half), cudaMemcpyHostToDevice);
    __half* d_bias = nullptr;
    if (use_bias) {
        cudaMalloc(&d_bias, N * sizeof(__half));
        cudaMemcpy(d_bias, h_bias.data(), N * sizeof(__half), cudaMemcpyHostToDevice);
    }
    __half* d_output;
    cudaMalloc(&d_output, M * N * sizeof(__half));

    gemm::int8_dp4a_gemm_dequant(d_A, d_B, d_act_scale, d_w_scale, d_bias,
                                  d_output, M, K, N);
    cudaDeviceSynchronize();

    // Read back and compare
    std::vector<__half> gpu_output_h(M * N);
    cudaMemcpy(gpu_output_h.data(), d_output, M * N * sizeof(__half), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    float mean_rel_err = 0.0f;
    int fails = 0;
    for (int i = 0; i < M * N; i++) {
        float gpu_val = __half2float(gpu_output_h[i]);
        float diff = std::abs(gpu_val - ref_output[i]);
        float rel_err = diff / (std::abs(ref_output[i]) + 1e-4f);
        if (diff > max_diff) max_diff = diff;
        mean_rel_err += rel_err;
        if (rel_err > 0.1f && diff > 0.5f) fails++;
    }
    mean_rel_err /= (M * N);

    std::printf("    max_diff=%.4f mean_rel_err=%.4f fails=%d/%d\n",
                max_diff, mean_rel_err, fails, M * N);

    test::free_device(d_A);
    test::free_device(d_B);
    test::free_device(d_act_scale);
    cudaFree(d_w_scale);
    if (d_bias) cudaFree(d_bias);
    cudaFree(d_output);

    if (fails > 0) {
        std::printf("    FAIL\n");
        return 1;
    }
    if (mean_rel_err > 0.03f) {
        std::printf("    FAIL (mean_rel_err too high)\n");
        return 1;
    }
    std::printf("    PASS\n");
    return 0;
}

static int test_unified_dispatch(int M, int K, int N) {
    std::printf("  unified dispatch (M=%d K=%d N=%d)...\n", M, K, N);

    std::vector<int8_t> h_A(M * K), h_B(N * K);
    test::fill_random_int8(h_A, 100);
    test::fill_random_int8(h_B, 101);
    std::vector<float> h_act_scale(M);
    test::fill_random(h_act_scale, 0.001f, 0.05f, 102);
    std::vector<float> h_w_scale_f(N);
    test::fill_random(h_w_scale_f, 0.001f, 0.05f, 103);
    std::vector<__half> h_w_scale(N);
    for (int i = 0; i < N; i++) h_w_scale[i] = __float2half(h_w_scale_f[i]);

    std::vector<float> ref_output(M * N);
    cpu_int8_gemm_dequant(h_A.data(), h_B.data(), h_act_scale.data(),
                          h_w_scale.data(), nullptr, ref_output.data(), M, K, N);

    int8_t* d_A = test::to_device(h_A);
    int8_t* d_B = test::to_device(h_B);
    float* d_act_scale = test::to_device(h_act_scale);
    __half* d_w_scale;
    cudaMalloc(&d_w_scale, N * sizeof(__half));
    cudaMemcpy(d_w_scale, h_w_scale.data(), N * sizeof(__half), cudaMemcpyHostToDevice);
    __half* d_output;
    cudaMalloc(&d_output, M * N * sizeof(__half));

    int cc = gemm::get_compute_capability();
    std::printf("    CC=%d, using %s path\n", cc, cc >= 80 ? "wmma" : "dp4a");

    gemm::int8_gemm(d_A, d_B, d_act_scale, d_w_scale, nullptr,
                    d_output, M, K, N);
    cudaDeviceSynchronize();

    std::vector<__half> gpu_output_h(M * N);
    cudaMemcpy(gpu_output_h.data(), d_output, M * N * sizeof(__half), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    int fails = 0;
    for (int i = 0; i < M * N; i++) {
        float gpu_val = __half2float(gpu_output_h[i]);
        float diff = std::abs(gpu_val - ref_output[i]);
        if (diff > max_diff) max_diff = diff;
        float rel_err = diff / (std::abs(ref_output[i]) + 1e-4f);
        if (rel_err > 0.1f && diff > 0.5f) fails++;
    }

    std::printf("    max_diff=%.4f fails=%d/%d\n", max_diff, fails, M * N);

    test::free_device(d_A);
    test::free_device(d_B);
    test::free_device(d_act_scale);
    cudaFree(d_w_scale);
    cudaFree(d_output);

    if (fails > 0) { std::printf("    FAIL\n"); return 1; }
    std::printf("    PASS\n");
    return 0;
}

int main() {
    std::printf("=== Test: INT8 dp4a GEMM ===\n");
    int failures = 0;

    // Basic: K aligned to 64 (matches wmma constraint)
    failures += test_dp4a_gemm(32, 128, 64, true, "basic_aligned");

    // Non-64-aligned K: K=28 (like FSR4 first conv: 7*2*2)
    failures += test_dp4a_gemm(16, 28, 16, true, "k_not_64_aligned");

    // K not multiple of 4 (edge case)
    failures += test_dp4a_gemm(8, 7, 8, true, "k_odd");

    // Large: exercise wide tile path (N > 2048)
    failures += test_dp4a_gemm(64, 64, 2112, false, "wide_tile");

    // Tall-skinny: like depth features
    failures += test_dp4a_gemm(256, 64, 16, true, "tall_skinny");

    // No bias
    failures += test_dp4a_gemm(32, 64, 32, false, "no_bias");

    // Unified dispatch
    failures += test_unified_dispatch(32, 128, 64);

    if (failures > 0) {
        std::printf("\nFAILED (%d test(s) failed)\n", failures);
        return 1;
    }
    std::printf("\nALL PASSED\n");
    return 0;
}
