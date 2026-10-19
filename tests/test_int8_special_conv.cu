// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Tests INT8 depthwise Conv2d and ConvTranspose2d (k=2 s=2).
#include <forge/nn/int8_depthwise_conv2d.cuh>
#include <forge/nn/int8_conv_transpose2d.cuh>
#include "test_helpers.cuh"
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

using namespace forge;

// ---------- Helpers ----------

static void fill_random_fp16(std::vector<__half>& v, float lo, float hi, uint64_t seed) {
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) x = __float2half(dist(rng));
}

static int conv_out_size(int in_size, int pad, int dil, int kernel, int stride) {
    return (in_size + 2 * pad - dil * (kernel - 1) - 1) / stride + 1;
}

// =====================================================================
// Depthwise Conv2d tests
// =====================================================================

static void cpu_depthwise_conv2d(
    const __half* input,   // [C, H, W]
    const int8_t* weight,  // [C, kH*kW]
    const __half* w_scale, // [C]
    const __half* bias,    // [C] or nullptr
    float*        output,  // [C, H_out, W_out]
    int C, int H, int W,
    int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW)
{
    int H_out = conv_out_size(H, padH, dilH, kH, strideH);
    int W_out = conv_out_size(W, padW, dilW, kW, strideW);

    for (int c = 0; c < C; c++) {
        float scale = __half2float(w_scale[c]);
        float b = bias ? __half2float(bias[c]) : 0.0f;

        for (int ho = 0; ho < H_out; ho++) {
            for (int wo = 0; wo < W_out; wo++) {
                float sum = 0.0f;
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        int h_in = ho * strideH + kh * dilH - padH;
                        int w_in = wo * strideW + kw * dilW - padW;
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            float inp = __half2float(input[c * H * W + h_in * W + w_in]);
                            float w = (float)weight[c * kH * kW + kh * kW + kw] * scale;
                            sum += inp * w;
                        }
                    }
                }
                output[c * H_out * W_out + ho * W_out + wo] = sum + b;
            }
        }
    }
}

static int test_depthwise(int C, int H, int W, int kH, int kW,
                           int padH, int padW, bool use_bias, const char* label) {
    int H_out = conv_out_size(H, padH, 1, kH, 1);
    int W_out = conv_out_size(W, padW, 1, kW, 1);
    std::printf("  %s (C=%d %dx%d k=%dx%d pad=%d bias=%d)...\n",
                label, C, H, W, kH, kW, padH, use_bias);

    std::vector<__half> h_input(C * H * W);
    fill_random_fp16(h_input, -1.0f, 1.0f, 42);

    std::vector<int8_t> h_weight(C * kH * kW);
    test::fill_random_int8(h_weight, 43);

    std::vector<float> h_ws_f(C);
    test::fill_random(h_ws_f, 0.005f, 0.05f, 44);
    std::vector<__half> h_w_scale(C);
    for (int i = 0; i < C; i++) h_w_scale[i] = __float2half(h_ws_f[i]);

    std::vector<__half> h_bias(C, __float2half(0.0f));
    if (use_bias) {
        std::vector<float> tmp(C);
        test::fill_random(tmp, -0.5f, 0.5f, 45);
        for (int i = 0; i < C; i++) h_bias[i] = __float2half(tmp[i]);
    }

    // CPU reference
    int out_size = C * H_out * W_out;
    std::vector<float> ref(out_size);
    cpu_depthwise_conv2d(h_input.data(), h_weight.data(), h_w_scale.data(),
                         use_bias ? h_bias.data() : nullptr, ref.data(),
                         C, H, W, kH, kW, padH, padW, 1, 1, 1, 1);

    // GPU
    __half* d_input;  cudaMalloc(&d_input, h_input.size() * sizeof(__half));
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(__half), cudaMemcpyHostToDevice);
    int8_t* d_weight = test::to_device(h_weight);
    __half* d_w_scale; cudaMalloc(&d_w_scale, C * sizeof(__half));
    cudaMemcpy(d_w_scale, h_w_scale.data(), C * sizeof(__half), cudaMemcpyHostToDevice);
    __half* d_bias = nullptr;
    if (use_bias) {
        cudaMalloc(&d_bias, C * sizeof(__half));
        cudaMemcpy(d_bias, h_bias.data(), C * sizeof(__half), cudaMemcpyHostToDevice);
    }
    __half* d_output; cudaMalloc(&d_output, out_size * sizeof(__half));

    nn::int8_depthwise_conv2d_forward(
        d_input, d_weight, d_w_scale, d_bias, d_output,
        1, C, H, W, kH, kW, padH, padW, 1, 1, 1, 1);
    cudaDeviceSynchronize();

    // Compare
    std::vector<__half> gpu_h(out_size);
    cudaMemcpy(gpu_h.data(), d_output, out_size * sizeof(__half), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f, mean_abs = 0.0f;
    for (int i = 0; i < out_size; i++) {
        float diff = std::abs(__half2float(gpu_h[i]) - ref[i]);
        max_diff = std::max(max_diff, diff);
        mean_abs += diff;
    }
    mean_abs /= out_size;

    // Depthwise uses float arithmetic (no INT8 activation quantization), so errors
    // should be tiny — just FP16 rounding.
    bool pass = (max_diff < 0.01f);
    if (pass) std::printf("    PASS (max_diff=%.6f mean_abs=%.8f)\n", max_diff, mean_abs);
    else std::fprintf(stderr, "    FAIL (max_diff=%.6f mean_abs=%.8f)\n", max_diff, mean_abs);

    cudaFree(d_input); cudaFree(d_weight); cudaFree(d_w_scale);
    if (d_bias) cudaFree(d_bias); cudaFree(d_output);
    return pass ? 0 : 1;
}

// =====================================================================
// ConvTranspose2d k=2 s=2 tests
// =====================================================================

static void cpu_conv_transpose2d_k2s2(
    const __half* input,       // [C_in, H, W]
    const int8_t* weight,      // [4*C_out, C_in] (repacked)
    const __half* w_scale,     // [C_out]
    const __half* bias,        // [C_out] or nullptr
    float*        output,      // [C_out, 2*H, 2*W]
    int C_in, int H, int W, int C_out)
{
    int H_out = 2 * H;
    int W_out = 2 * W;

    // output[co, 2*h+kh, 2*w+kw] = sum_ci input[ci, h, w] * weight[co*4+kh*2+kw, ci] * w_scale[co] + bias[co]
    for (int co = 0; co < C_out; co++) {
        float scale = __half2float(w_scale[co]);
        float b = bias ? __half2float(bias[co]) : 0.0f;

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        float sum = 0.0f;
                        int n = co * 4 + kh * 2 + kw;
                        for (int ci = 0; ci < C_in; ci++) {
                            float inp = __half2float(input[ci * H * W + h * W + w]);
                            float wt = (float)weight[n * C_in + ci] * scale;
                            sum += inp * wt;
                        }
                        int ho = 2 * h + kh;
                        int wo = 2 * w + kw;
                        output[co * H_out * W_out + ho * W_out + wo] = sum + b;
                    }
                }
            }
        }
    }
}

static int test_conv_transpose2d(int C_in, int H, int W, int C_out,
                                  bool use_bias, const char* label) {
    int H_out = 2 * H;
    int W_out = 2 * W;
    std::printf("  %s (C_in=%d %dx%d → C_out=%d %dx%d bias=%d)...\n",
                label, C_in, H, W, C_out, H_out, W_out, use_bias);

    std::vector<__half> h_input(C_in * H * W);
    fill_random_fp16(h_input, -1.0f, 1.0f, 42);

    // Weight: [4*C_out, C_in] (repacked layout)
    std::vector<int8_t> h_weight(4 * C_out * C_in);
    test::fill_random_int8(h_weight, 43);

    std::vector<float> h_ws_f(C_out);
    test::fill_random(h_ws_f, 0.005f, 0.05f, 44);
    std::vector<__half> h_w_scale(C_out);
    for (int i = 0; i < C_out; i++) h_w_scale[i] = __float2half(h_ws_f[i]);

    std::vector<__half> h_bias(C_out, __float2half(0.0f));
    if (use_bias) {
        std::vector<float> tmp(C_out);
        test::fill_random(tmp, -0.5f, 0.5f, 45);
        for (int i = 0; i < C_out; i++) h_bias[i] = __float2half(tmp[i]);
    }

    // CPU reference
    int out_size = C_out * H_out * W_out;
    std::vector<float> ref(out_size);
    cpu_conv_transpose2d_k2s2(h_input.data(), h_weight.data(), h_w_scale.data(),
                              use_bias ? h_bias.data() : nullptr, ref.data(),
                              C_in, H, W, C_out);

    // GPU
    __half* d_input; cudaMalloc(&d_input, h_input.size() * sizeof(__half));
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(__half), cudaMemcpyHostToDevice);
    int8_t* d_weight = test::to_device(h_weight);
    __half* d_w_scale; cudaMalloc(&d_w_scale, C_out * sizeof(__half));
    cudaMemcpy(d_w_scale, h_w_scale.data(), C_out * sizeof(__half), cudaMemcpyHostToDevice);
    __half* d_bias = nullptr;
    if (use_bias) {
        cudaMalloc(&d_bias, C_out * sizeof(__half));
        cudaMemcpy(d_bias, h_bias.data(), C_out * sizeof(__half), cudaMemcpyHostToDevice);
    }
    __half* d_output; cudaMalloc(&d_output, out_size * sizeof(__half));

    size_t ws_size = nn::int8_conv_transpose2d_k2s2_workspace_size(C_in, H, W, C_out);
    void* d_ws; cudaMalloc(&d_ws, ws_size);

    nn::int8_conv_transpose2d_k2s2_forward(
        d_input, d_weight, d_w_scale, d_bias, d_output,
        1, C_in, H, W, C_out, d_ws);
    cudaDeviceSynchronize();

    // Compare
    std::vector<__half> gpu_h(out_size);
    cudaMemcpy(gpu_h.data(), d_output, out_size * sizeof(__half), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f, mean_abs = 0.0f;
    for (int i = 0; i < out_size; i++) {
        float diff = std::abs(__half2float(gpu_h[i]) - ref[i]);
        max_diff = std::max(max_diff, diff);
        mean_abs += diff;
    }
    mean_abs /= out_size;

    // INT8 activation quantization error (same as conv2d)
    bool pass = (max_diff < 0.5f) && (mean_abs < 0.1f);
    if (pass)
        std::printf("    PASS (max_diff=%.4f mean_abs=%.6f ws=%.1f KB)\n",
                    max_diff, mean_abs, ws_size / 1024.0f);
    else
        std::fprintf(stderr, "    FAIL (max_diff=%.4f mean_abs=%.6f)\n",
                     max_diff, mean_abs);

    cudaFree(d_input); cudaFree(d_weight); cudaFree(d_w_scale);
    if (d_bias) cudaFree(d_bias); cudaFree(d_output); cudaFree(d_ws);
    return pass ? 0 : 1;
}

// =====================================================================
// Main
// =====================================================================

int main() {
    std::printf("=== INT8 Special Conv Tests ===\n\n");
    int failures = 0;

    // --- Depthwise Conv2d ---
    std::printf("-- Depthwise Conv2d --\n");

    failures += test_depthwise(16, 16, 16, 3, 3, 1, 1, true, "dw_3x3_C16");
    failures += test_depthwise(32, 32, 32, 3, 3, 1, 1, true, "dw_3x3_C32");
    failures += test_depthwise(64, 8, 8, 3, 3, 1, 1, false, "dw_3x3_C64_nobias");

    // --- ConvTranspose2d k=2 s=2 ---
    std::printf("\n-- ConvTranspose2d k=2 s=2 --\n");

    // FSR4-like configs
    failures += test_conv_transpose2d(32, 16, 16, 64, true, "ct_32to64");
    failures += test_conv_transpose2d(64, 8, 8, 32, true, "ct_64to32");
    failures += test_conv_transpose2d(16, 32, 32, 16, false, "ct_16to16_nobias");

    std::printf("\n=== %s (%d failures) ===\n",
                failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures;
}
