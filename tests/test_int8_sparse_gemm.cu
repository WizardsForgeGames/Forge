// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Tests the full sparse INT8 GEMM pipeline:
//   prune_compress_24 -> pack_metadata -> sparse_int8_gemm
// Compares against a CPU reference that performs the same operations.
#include <forge/gemm/int8_sparse.cuh>
#include <forge/sparsity/prune_compress.cuh>
#include <forge/sparsity/pack_metadata.cuh>
#include "test_helpers.cuh"
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

using namespace forge;

// CPU sparse GEMM reference:
// Given compressed weight (N, K/2), metadata (N*K/4), and dense activation (M, K),
// compute output(M, N) = sum_k W_unpruned(n,k) * x(m,k) scaled and biased.
// We reconstruct the dot product using the compressed format.
void sparse_gemm_cpu(const int8_t* w_comp, const uint8_t* w_meta_raw,
                     const int8_t* x, const float* act_scale,
                     const float* w_scale, const float* bias,
                     float* output, int M, int K, int N) {
    int groups_per_row = K / 4;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int g = 0; g < groups_per_row; g++) {
                uint8_t nib = w_meta_raw[n * groups_per_row + g];
                int k0 = nib & 3, k1 = (nib >> 2) & 3;
                int k_base = g * 4;
                int comp_base = n * (K / 2) + g * 2;
                acc += (int32_t)w_comp[comp_base]     * (int32_t)x[m * K + k_base + k0];
                acc += (int32_t)w_comp[comp_base + 1] * (int32_t)x[m * K + k_base + k1];
            }
            float val = (float)acc * act_scale[m] * w_scale[n];
            if (bias) val += bias[n];
            output[m * N + n] = val;
        }
    }
}

int main() {
    std::printf("=== Test: INT8 Sparse GEMM ===\n");
    // Use dimensions compatible with kernel requirements: K%64==0, N%16==0
    const int M = 32, K = 128, N = 64;

    // Generate random INT8 weight and activation
    std::vector<int8_t> h_weight(N * K), h_x(M * K);
    test::fill_random_int8(h_weight, 42);
    test::fill_random_int8(h_x, 43);

    // CPU: prune + compress
    std::vector<int8_t> h_comp(N * K / 2);
    std::vector<uint8_t> h_meta_raw(N * K / 4);
    // Reuse the same pruning logic
    for (int row = 0; row < N; row++) {
        for (int g = 0; g < K / 4; g++) {
            int base = row * K + g * 4;
            int8_t vals[4] = {h_weight[base], h_weight[base+1],
                              h_weight[base+2], h_weight[base+3]};
            int abs_v[4];
            for (int i = 0; i < 4; i++) abs_v[i] = std::abs((int)vals[i]);
            int min1 = 0, min2 = 1;
            if (abs_v[0] > abs_v[1]) { min1 = 1; min2 = 0; }
            for (int i = 2; i < 4; i++) {
                if (abs_v[i] < abs_v[min1]) { min2 = min1; min1 = i; }
                else if (abs_v[i] < abs_v[min2]) { min2 = i; }
            }
            int keep[2], ki = 0;
            for (int i = 0; i < 4; i++)
                if (i != min1 && i != min2) keep[ki++] = i;
            h_comp[row * (K/2) + g * 2]     = vals[keep[0]];
            h_comp[row * (K/2) + g * 2 + 1] = vals[keep[1]];
            h_meta_raw[row * (K/4) + g] = (uint8_t)(keep[0] | (keep[1] << 2));
        }
    }

    // Scales and bias
    std::vector<float> h_act_scale(M), h_w_scale_f(N), h_bias_f(N);
    test::fill_random(h_act_scale, 0.001f, 0.05f, 44);
    test::fill_random(h_w_scale_f, 0.001f, 0.05f, 45);
    test::fill_random(h_bias_f, -0.5f, 0.5f, 46);

    // CPU reference
    std::vector<float> ref_output(M * N);
    sparse_gemm_cpu(h_comp.data(), h_meta_raw.data(), h_x.data(),
                    h_act_scale.data(), h_w_scale_f.data(), h_bias_f.data(),
                    ref_output.data(), M, K, N);

    // GPU pipeline: prune -> pack -> gemm
    int8_t* d_weight = test::to_device(h_weight);
    int8_t* d_comp = test::to_device_zeroed<int8_t>(N * K / 2);
    uint8_t* d_meta_raw = test::to_device_zeroed<uint8_t>(N * K / 4);
    uint32_t* d_meta_packed = test::to_device_zeroed<uint32_t>(N * K / 32);

    sparsity::prune_compress_24(d_weight, d_comp, d_meta_raw, N, K);
    sparsity::pack_metadata(d_meta_raw, d_meta_packed, N, K);
    cudaDeviceSynchronize();

    int8_t* d_x = test::to_device(h_x);
    float* d_act_scale = test::to_device(h_act_scale);
    std::vector<__half> h_w_scale(N), h_bias(N);
    for (int i = 0; i < N; i++) {
        h_w_scale[i] = __float2half(h_w_scale_f[i]);
        h_bias[i] = __float2half(h_bias_f[i]);
    }
    __half* d_w_scale;
    cudaMalloc(&d_w_scale, N * sizeof(__half));
    cudaMemcpy(d_w_scale, h_w_scale.data(), N * sizeof(__half), cudaMemcpyHostToDevice);
    __half* d_bias;
    cudaMalloc(&d_bias, N * sizeof(__half));
    cudaMemcpy(d_bias, h_bias.data(), N * sizeof(__half), cudaMemcpyHostToDevice);
    __half* d_output;
    cudaMalloc(&d_output, M * N * sizeof(__half));

    gemm::sparse_int8_gemm(d_comp, d_x, d_meta_packed, d_act_scale,
                           d_w_scale, d_bias, d_output, M, K, N,
                           nullptr, 1);
    cudaDeviceSynchronize();

    // Read back and compare
    std::vector<__half> gpu_output_h(M * N);
    cudaMemcpy(gpu_output_h.data(), d_output, M * N * sizeof(__half), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    float mean_rel = 0.0f;
    int fails = 0;
    for (int i = 0; i < M * N; i++) {
        float gpu_val = __half2float(gpu_output_h[i]);
        float diff = std::abs(gpu_val - ref_output[i]);
        float rel = diff / (std::abs(ref_output[i]) + 1e-4f);
        if (diff > max_diff) max_diff = diff;
        mean_rel += rel;
        if (rel > 0.1f && diff > 0.5f) fails++;
    }
    mean_rel /= (M * N);

    std::printf("  max_diff=%.4f mean_rel=%.4f fails=%d/%d\n",
                max_diff, mean_rel, fails, M * N);
    FORGE_ASSERT(fails == 0, "sparse GEMM output mismatch");
    FORGE_ASSERT(mean_rel < 0.05f, "sparse GEMM mean relative error too high");
    std::printf("  PASS [sparse_int8_gemm]\n");

    test::free_device(d_weight);
    test::free_device(d_comp);
    test::free_device(d_meta_raw);
    test::free_device(d_meta_packed);
    test::free_device(d_x);
    test::free_device(d_act_scale);
    cudaFree(d_w_scale);
    cudaFree(d_bias);
    cudaFree(d_output);
    std::printf("ALL PASSED\n");
    return 0;
}
