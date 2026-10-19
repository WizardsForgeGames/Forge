// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Tests INT8 Conv2d forward (inference-only).
// Validates against a float CPU reference conv2d with tolerance for
// INT8 activation quantization error.
#include <forge/nn/int8_conv2d.cuh>
#include "test_helpers.cuh"
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

using namespace forge;

// ---------- FP16 helpers ----------

static void fill_random_fp16(std::vector<__half>& v, float lo, float hi, uint64_t seed) {
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) x = __float2half(dist(rng));
}

// ---------- CPU reference: float conv2d ----------
// Uses dequantized INT8 weights so the "ideal" answer matches what the
// GPU would compute with perfect (non-quantized) activations.

static int conv_out_size(int in_size, int pad, int dil, int kernel, int stride) {
    return (in_size + 2 * pad - dil * (kernel - 1) - 1) / stride + 1;
}

static void cpu_float_conv2d(
    const __half* input,       // [C_in, H, W]
    const int8_t* weight,      // [C_out, C_in*kH*kW]
    const __half* w_scale,     // [C_out]
    const __half* bias,        // [C_out] or nullptr
    float*        output,      // [C_out, H_out, W_out]
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW)
{
    int H_out = conv_out_size(H, padH, dilH, kH, strideH);
    int W_out = conv_out_size(W, padW, dilW, kW, strideW);
    int K = C_in * kH * kW;

    for (int co = 0; co < C_out; co++) {
        float scale = __half2float(w_scale[co]);
        float b = bias ? __half2float(bias[co]) : 0.0f;

        for (int ho = 0; ho < H_out; ho++) {
            for (int wo = 0; wo < W_out; wo++) {
                float sum = 0.0f;
                for (int ci = 0; ci < C_in; ci++) {
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int h_in = ho * strideH + kh * dilH - padH;
                            int w_in = wo * strideW + kw * dilW - padW;
                            float inp = 0.0f;
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
                                inp = __half2float(input[ci * H * W + h_in * W + w_in]);
                            int w_idx = ci * kH * kW + kh * kW + kw;
                            float w = (float)weight[co * K + w_idx] * scale;
                            sum += inp * w;
                        }
                    }
                }
                output[co * H_out * W_out + ho * W_out + wo] = sum + b;
            }
        }
    }
}

// ---------- Test function ----------

static int test_int8_conv2d(
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW,
    bool use_bias,
    const char* label)
{
    int H_out = conv_out_size(H, padH, dilH, kH, strideH);
    int W_out = conv_out_size(W, padW, dilW, kW, strideW);
    int K = C_in * kH * kW;
    int N = 1;  // single sample for testing

    std::printf("  %s (C_in=%d %dx%d → C_out=%d k=%dx%d s=%dx%d p=%dx%d d=%dx%d bias=%d)...\n",
                label, C_in, H, W, C_out, kH, kW, strideH, strideW,
                padH, padW, dilH, dilW, use_bias);

    // Generate test data
    std::vector<__half> h_input(C_in * H * W);
    fill_random_fp16(h_input, -1.0f, 1.0f, 42);

    std::vector<int8_t> h_weight(C_out * K);
    test::fill_random_int8(h_weight, 43);

    std::vector<float> h_w_scale_f(C_out);
    test::fill_random(h_w_scale_f, 0.005f, 0.05f, 44);
    std::vector<__half> h_w_scale(C_out);
    for (int i = 0; i < C_out; i++) h_w_scale[i] = __float2half(h_w_scale_f[i]);

    std::vector<__half> h_bias(C_out, __float2half(0.0f));
    if (use_bias) {
        std::vector<float> tmp(C_out);
        test::fill_random(tmp, -0.5f, 0.5f, 45);
        for (int i = 0; i < C_out; i++) h_bias[i] = __float2half(tmp[i]);
    }

    // CPU reference
    std::vector<float> ref_output(C_out * H_out * W_out);
    cpu_float_conv2d(h_input.data(), h_weight.data(), h_w_scale.data(),
                     use_bias ? h_bias.data() : nullptr,
                     ref_output.data(),
                     C_in, H, W, C_out, kH, kW,
                     padH, padW, strideH, strideW, dilH, dilW);

    // GPU: allocate and upload
    __half* d_input;
    cudaMalloc(&d_input, h_input.size() * sizeof(__half));
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(__half), cudaMemcpyHostToDevice);

    int8_t* d_weight = test::to_device(h_weight);

    __half* d_w_scale;
    cudaMalloc(&d_w_scale, C_out * sizeof(__half));
    cudaMemcpy(d_w_scale, h_w_scale.data(), C_out * sizeof(__half), cudaMemcpyHostToDevice);

    __half* d_bias = nullptr;
    if (use_bias) {
        cudaMalloc(&d_bias, C_out * sizeof(__half));
        cudaMemcpy(d_bias, h_bias.data(), C_out * sizeof(__half), cudaMemcpyHostToDevice);
    }

    __half* d_output;
    cudaMalloc(&d_output, N * C_out * H_out * W_out * sizeof(__half));

    // Workspace
    size_t ws_size = nn::int8_conv2d_workspace_size(
        C_in, H, W, C_out, kH, kW, padH, padW, strideH, strideW, dilH, dilW);
    void* d_workspace;
    cudaMalloc(&d_workspace, ws_size);

    // Run
    nn::int8_conv2d_forward(
        d_input, d_weight, d_w_scale, d_bias, d_output,
        N, C_in, H, W, C_out, kH, kW,
        padH, padW, strideH, strideW, dilH, dilW,
        d_workspace);
    cudaDeviceSynchronize();

    // Read back
    int out_size = C_out * H_out * W_out;
    std::vector<__half> gpu_output_h(out_size);
    cudaMemcpy(gpu_output_h.data(), d_output, out_size * sizeof(__half), cudaMemcpyDeviceToHost);
    std::vector<float> gpu_output_f(out_size);
    for (int i = 0; i < out_size; i++) gpu_output_f[i] = __half2float(gpu_output_h[i]);

    // Compare with absolute tolerance.
    // INT8 activation quantization introduces ~0.008 per-element error (1/127),
    // which accumulates over K filter taps. Expected max_diff scales with sqrt(K).
    // Relative error is unreliable near zero, so we use absolute tolerance only.
    float max_diff = 0.0f;
    float mean_abs_err = 0.0f;
    for (int i = 0; i < out_size; i++) {
        float diff = std::abs(gpu_output_f[i] - ref_output[i]);
        max_diff = std::max(max_diff, diff);
        mean_abs_err += diff;
    }
    mean_abs_err /= out_size;

    // Tolerance: max_diff < 0.5 (generous for INT8 quant), mean_abs < 0.1
    bool pass = (max_diff < 0.5f) && (mean_abs_err < 0.1f);

    if (pass) {
        std::printf("    PASS (max_diff=%.4f mean_abs=%.6f ws=%.1f KB)\n",
                    max_diff, mean_abs_err, ws_size / 1024.0f);
    } else {
        std::fprintf(stderr, "    FAIL (max_diff=%.4f mean_abs=%.6f)\n",
                     max_diff, mean_abs_err);
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_w_scale);
    if (d_bias) cudaFree(d_bias);
    cudaFree(d_output);
    cudaFree(d_workspace);

    return pass ? 0 : 1;
}

// ---------- Main ----------

int main() {
    std::printf("=== INT8 Conv2d Forward Tests ===\n\n");
    int failures = 0;

    // Test 1: Standard 3×3 conv with padding
    failures += test_int8_conv2d(
        /*C_in=*/8, /*H=*/16, /*W=*/16,
        /*C_out=*/16, /*kH=*/3, /*kW=*/3,
        /*padH=*/1, /*padW=*/1,
        /*strideH=*/1, /*strideW=*/1,
        /*dilH=*/1, /*dilW=*/1,
        /*bias=*/true, "3x3_pad1_bias");

    // Test 2: Pointwise 1×1 conv (special-cased in implementation)
    failures += test_int8_conv2d(
        /*C_in=*/16, /*H=*/8, /*W=*/8,
        /*C_out=*/32, /*kH=*/1, /*kW=*/1,
        /*padH=*/0, /*padW=*/0,
        /*strideH=*/1, /*strideW=*/1,
        /*dilH=*/1, /*dilW=*/1,
        /*bias=*/true, "1x1_pointwise");

    // Test 3: Strided 2×2 conv (FSR4 first conv pattern)
    failures += test_int8_conv2d(
        /*C_in=*/7, /*H=*/32, /*W=*/32,
        /*C_out=*/16, /*kH=*/2, /*kW=*/2,
        /*padH=*/0, /*padW=*/0,
        /*strideH=*/2, /*strideW=*/2,
        /*dilH=*/1, /*dilW=*/1,
        /*bias=*/true, "2x2_stride2_fsr4");

    // Test 4: No bias
    failures += test_int8_conv2d(
        /*C_in=*/4, /*H=*/12, /*W=*/12,
        /*C_out=*/8, /*kH=*/3, /*kW=*/3,
        /*padH=*/1, /*padW=*/1,
        /*strideH=*/1, /*strideW=*/1,
        /*dilH=*/1, /*dilW=*/1,
        /*bias=*/false, "3x3_no_bias");

    // Test 5: Larger spatial (closer to real use)
    failures += test_int8_conv2d(
        /*C_in=*/16, /*H=*/64, /*W=*/64,
        /*C_out=*/32, /*kH=*/3, /*kW=*/3,
        /*padH=*/1, /*padW=*/1,
        /*strideH=*/1, /*strideW=*/1,
        /*dilH=*/1, /*dilW=*/1,
        /*bias=*/true, "larger_spatial");

    // Test 6: Dilated convolution
    failures += test_int8_conv2d(
        /*C_in=*/8, /*H=*/16, /*W=*/16,
        /*C_out=*/8, /*kH=*/3, /*kW=*/3,
        /*padH=*/2, /*padW=*/2,
        /*strideH=*/1, /*strideW=*/1,
        /*dilH=*/2, /*dilW=*/2,
        /*bias=*/true, "3x3_dilated2");

    std::printf("\n=== %s (%d failures) ===\n",
                failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures;
}
