// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// fused_quantize_activation against a CPU reference (+/- 1 for rounding) and against the
// two-pass kernel (bit-identical), across the dispatch paths: single-pass vector kernel with
// 1..4 vectors per thread, evict-first loads for inputs larger than L2, and the two-pass
// fallback (K % 8 != 0, K > 8192, misaligned input).
#include <forge/quant/quantize.cuh>
#include "test_helpers.cuh"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cuda_fp16.h>

using namespace forge;

static int run_case(int M, int K, int offset_elems, uint64_t seed) {
    std::printf("  M=%d K=%d offset=%d\n", M, K, offset_elems);
    const size_t n = (size_t)M * K;

    std::vector<float> h_input_f32(n);
    test::fill_random(h_input_f32, -3.0f, 3.0f, seed);
    std::vector<__half> h_input(n + offset_elems);
    for (size_t i = 0; i < n; i++)
        h_input[offset_elems + i] = __float2half(h_input_f32[i]);

    // CPU reference quantization on the fp16-rounded input
    std::vector<int8_t> ref_output(n);
    std::vector<float> ref_scales(M);
    for (int m = 0; m < M; m++) {
        float row_max = 0.0f;
        for (int k = 0; k < K; k++)
            row_max = std::max(row_max, std::abs(__half2float(h_input[offset_elems + (size_t)m * K + k])));
        float scale = std::max(row_max, 1e-8f) / 127.0f;
        ref_scales[m] = scale;
        float inv_scale = 1.0f / scale;
        for (int k = 0; k < K; k++) {
            int q = (int)std::round(__half2float(h_input[offset_elems + (size_t)m * K + k]) * inv_scale);
            q = std::max(-128, std::min(127, q));
            ref_output[(size_t)m * K + k] = (int8_t)q;
        }
    }

    __half* d_buf = test::to_device(h_input);
    const __half* d_input = d_buf + offset_elems;
    int8_t* d_output = test::to_device_zeroed<int8_t>(n);
    float*  d_scales = test::to_device_zeroed<float>(M);
    int8_t* d_output2 = test::to_device_zeroed<int8_t>(n);
    float*  d_scales2 = test::to_device_zeroed<float>(M);

    quant::fused_quantize_activation(d_input, d_output, d_scales, M, K);
    FORGE_CHECK(cudaGetLastError());
    // Two-pass kernel directly (the launcher's fallback path)
    int threads = std::min(256, ((K + 31) / 32) * 32);
    quant::fused_quantize_activation_kernel<<<M, threads>>>(d_input, d_output2, d_scales2, K);
    FORGE_CHECK(cudaGetLastError());
    FORGE_CHECK(cudaDeviceSynchronize());

    auto gpu_output  = test::to_host(d_output, n);
    auto gpu_scales  = test::to_host(d_scales, M);
    auto gpu_output2 = test::to_host(d_output2, n);
    auto gpu_scales2 = test::to_host(d_scales2, M);

    bool scales_ok = true;
    for (int m = 0; m < M; m++) {
        float diff = std::abs(gpu_scales[m] - ref_scales[m]);
        float rtol = diff / (std::abs(ref_scales[m]) + 1e-8f);
        if (rtol > 0.01f) {
            if (scales_ok)
                std::fprintf(stderr, "FAIL: scale[%d] gpu=%.6f ref=%.6f\n",
                             m, gpu_scales[m], ref_scales[m]);
            scales_ok = false;
        }
    }

    int mismatches = 0, max_diff = 0;
    for (size_t i = 0; i < n; i++) {
        int diff = std::abs((int)gpu_output[i] - (int)ref_output[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1) mismatches++;
    }
    const bool identical =
        std::memcmp(gpu_output.data(), gpu_output2.data(), n) == 0 &&
        std::memcmp(gpu_scales.data(), gpu_scales2.data(), M * sizeof(float)) == 0;
    std::printf("    quant: max_diff=%d mismatches(>1)=%d/%zu, identical to two-pass: %s\n",
                max_diff, mismatches, n, identical ? "yes" : "NO");

    test::free_device(d_buf);
    test::free_device(d_output);
    test::free_device(d_scales);
    test::free_device(d_output2);
    test::free_device(d_scales2);

    FORGE_ASSERT(scales_ok, "quantize scales mismatch");
    FORGE_ASSERT(mismatches == 0, "quantize output mismatch (>1 away from ref)");
    FORGE_ASSERT(identical, "single-pass and two-pass kernels differ");
    return 0;
}

int main() {
    std::printf("=== Test: Quantize Activation ===\n");
    struct Case { int M, K, offset; };
    const Case cases[] = {
        {  32,  256, 0},   // original shape: vector path, 1 vector per thread
        {   7,    8, 0},   // one vector per row, 32 threads (mostly idle)
        {  64, 2056, 0},   // 2 vectors per thread (ragged)
        {  16, 6144, 0},   // 3 vectors per thread
        {  16, 8192, 0},   // 4 vectors per thread (largest vector K)
        {   5, 8200, 0},   // K > 8192: two-pass fallback
        {   9,  100, 0},   // K % 8 != 0: two-pass fallback
        {   9,  256, 1},   // misaligned input: two-pass fallback
        {8192, 5120, 0},   // 84 MB input > L2: evict-first loads
    };
    uint64_t seed = 42;
    for (const Case& c : cases) {
        if (run_case(c.M, c.K, c.offset, seed++) != 0) return 1;
    }
    std::printf("  PASS [quantize values]\n");
    std::printf("ALL PASSED\n");
    return 0;
}
