// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/quant/quantize.cuh>
#include "test_helpers.cuh"
#include <cmath>
#include <algorithm>
#include <cuda_fp16.h>

using namespace forge;

int main() {
    std::printf("=== Test: Quantize Activation ===\n");
    const int M = 32, K = 256;

    // Generate random FP16 input on host (as float, convert on device)
    std::vector<float> h_input_f32(M * K);
    test::fill_random(h_input_f32, -3.0f, 3.0f, 42);

    // Convert to half on host
    std::vector<__half> h_input(M * K);
    for (int i = 0; i < M * K; i++)
        h_input[i] = __float2half(h_input_f32[i]);

    // CPU reference quantization
    std::vector<int8_t> ref_output(M * K);
    std::vector<float> ref_scales(M);
    for (int m = 0; m < M; m++) {
        float row_max = 0.0f;
        for (int k = 0; k < K; k++)
            row_max = std::max(row_max, std::abs(h_input_f32[m * K + k]));
        float scale = std::max(row_max, 1e-8f) / 127.0f;
        ref_scales[m] = scale;
        float inv_scale = 1.0f / scale;
        for (int k = 0; k < K; k++) {
            int q = (int)std::round(h_input_f32[m * K + k] * inv_scale);
            q = std::max(-128, std::min(127, q));
            ref_output[m * K + k] = (int8_t)q;
        }
    }

    // GPU
    __half* d_input;
    cudaMalloc(&d_input, M * K * sizeof(__half));
    cudaMemcpy(d_input, h_input.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice);
    int8_t* d_output = test::to_device_zeroed<int8_t>(M * K);
    float* d_scales = test::to_device_zeroed<float>(M);

    quant::fused_quantize_activation(d_input, d_output, d_scales, M, K);
    cudaDeviceSynchronize();

    auto gpu_output = test::to_host(d_output, (size_t)M * K);
    auto gpu_scales = test::to_host(d_scales, M);

    // Check scales
    bool scales_ok = true;
    for (int m = 0; m < M; m++) {
        float diff = std::abs(gpu_scales[m] - ref_scales[m]);
        float rtol = diff / (std::abs(ref_scales[m]) + 1e-8f);
        if (rtol > 0.01f) {
            std::fprintf(stderr, "FAIL: scale[%d] gpu=%.6f ref=%.6f\n",
                         m, gpu_scales[m], ref_scales[m]);
            scales_ok = false;
        }
    }
    FORGE_ASSERT(scales_ok, "quantize scales mismatch");
    std::printf("  PASS [scales]\n");

    // Check quantized values (allow +/- 1 for rounding)
    int mismatches = 0;
    int max_diff = 0;
    for (int i = 0; i < M * K; i++) {
        int diff = std::abs((int)gpu_output[i] - (int)ref_output[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1) mismatches++;
    }
    std::printf("  quant: max_diff=%d mismatches(>1)=%d/%d\n",
                max_diff, mismatches, M * K);
    FORGE_ASSERT(mismatches == 0, "quantize output mismatch (>1 away from ref)");
    std::printf("  PASS [quantize values]\n");

    cudaFree(d_input);
    test::free_device(d_output);
    test::free_device(d_scales);
    std::printf("ALL PASSED\n");
    return 0;
}
