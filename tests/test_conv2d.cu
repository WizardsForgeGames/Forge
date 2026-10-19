// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/conv2d.cuh>
#include <forge/autograd.h>
#include <forge/tensor.h>
#include <forge/tensor_ops.h>
#include <forge/reduction_ops.h>
#include "test_helpers.cuh"
#include <cmath>
#include <vector>

using namespace forge;

// =====================================================================
// CPU reference implementations
// =====================================================================

static int conv_out_size(int in_size, int pad, int dil, int kernel, int stride) {
    return (in_size + 2 * pad - dil * (kernel - 1) - 1) / stride + 1;
}

// CPU conv2d forward (NCHW, groups support)
void conv2d_fwd_cpu(
    const float* input,   // [N, C_in, H, W]
    const float* weight,  // [C_out, C_in/groups, kH, kW]
    const float* bias,    // [C_out] or nullptr
    float*       output,  // [N, C_out, H_out, W_out]
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW,
    int groups)
{
    int H_out = conv_out_size(H, padH, dilH, kH, strideH);
    int W_out = conv_out_size(W, padW, dilW, kW, strideW);
    int C_in_per_group  = C_in  / groups;
    int C_out_per_group = C_out / groups;

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            for (int co = 0; co < C_out_per_group; ++co) {
                int co_abs = g * C_out_per_group + co;
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        float sum = (bias) ? bias[co_abs] : 0.0f;
                        for (int ci = 0; ci < C_in_per_group; ++ci) {
                            int ci_abs = g * C_in_per_group + ci;
                            for (int kh = 0; kh < kH; ++kh) {
                                for (int kw = 0; kw < kW; ++kw) {
                                    int h_in = h * strideH + kh * dilH - padH;
                                    int w_in = w * strideW + kw * dilW - padW;
                                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                        float iv = input[((n * C_in + ci_abs) * H + h_in) * W + w_in];
                                        float wv = weight[((co_abs * C_in_per_group + ci) * kH + kh) * kW + kw];
                                        sum += iv * wv;
                                    }
                                }
                            }
                        }
                        output[((n * C_out + co_abs) * H_out + h) * W_out + w] = sum;
                    }
                }
            }
        }
    }
}

// CPU conv2d backward data
void conv2d_bwd_data_cpu(
    const float* d_output, // [N, C_out, H_out, W_out]
    const float* weight,   // [C_out, C_in/groups, kH, kW]
    float*       d_input,  // [N, C_in, H, W]
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW,
    int groups)
{
    int H_out = conv_out_size(H, padH, dilH, kH, strideH);
    int W_out = conv_out_size(W, padW, dilW, kW, strideW);
    int C_in_per_group  = C_in  / groups;
    int C_out_per_group = C_out / groups;

    // Zero d_input
    for (int i = 0; i < N * C_in * H * W; ++i) d_input[i] = 0.0f;

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            for (int co = 0; co < C_out_per_group; ++co) {
                int co_abs = g * C_out_per_group + co;
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        float dout = d_output[((n * C_out + co_abs) * H_out + h) * W_out + w];
                        for (int ci = 0; ci < C_in_per_group; ++ci) {
                            int ci_abs = g * C_in_per_group + ci;
                            for (int kh = 0; kh < kH; ++kh) {
                                for (int kw = 0; kw < kW; ++kw) {
                                    int h_in = h * strideH + kh * dilH - padH;
                                    int w_in = w * strideW + kw * dilW - padW;
                                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                        float wv = weight[((co_abs * C_in_per_group + ci) * kH + kh) * kW + kw];
                                        d_input[((n * C_in + ci_abs) * H + h_in) * W + w_in] += dout * wv;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// CPU conv2d backward weight
void conv2d_bwd_weight_cpu(
    const float* d_output, // [N, C_out, H_out, W_out]
    const float* input,    // [N, C_in, H, W]
    float*       d_weight, // [C_out, C_in/groups, kH, kW]
    float*       d_bias,   // [C_out] or nullptr
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW,
    int groups,
    float beta)
{
    int H_out = conv_out_size(H, padH, dilH, kH, strideH);
    int W_out = conv_out_size(W, padW, dilW, kW, strideW);
    int C_in_per_group  = C_in  / groups;
    int C_out_per_group = C_out / groups;

    int weight_size = C_out * C_in_per_group * kH * kW;

    // Apply beta
    if (beta == 0.0f) {
        for (int i = 0; i < weight_size; ++i) d_weight[i] = 0.0f;
    } else {
        for (int i = 0; i < weight_size; ++i) d_weight[i] *= beta;
    }

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            for (int co = 0; co < C_out_per_group; ++co) {
                int co_abs = g * C_out_per_group + co;
                for (int ci = 0; ci < C_in_per_group; ++ci) {
                    int ci_abs = g * C_in_per_group + ci;
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            float sum = 0.0f;
                            for (int h = 0; h < H_out; ++h) {
                                for (int w = 0; w < W_out; ++w) {
                                    int h_in = h * strideH + kh * dilH - padH;
                                    int w_in = w * strideW + kw * dilW - padW;
                                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                        float iv = input[((n * C_in + ci_abs) * H + h_in) * W + w_in];
                                        float dout = d_output[((n * C_out + co_abs) * H_out + h) * W_out + w];
                                        sum += dout * iv;
                                    }
                                }
                            }
                            d_weight[((co_abs * C_in_per_group + ci) * kH + kh) * kW + kw] += sum;
                        }
                    }
                }
            }
        }
    }

    // Bias gradient
    if (d_bias) {
        for (int c = 0; c < C_out; ++c) {
            float sum = 0.0f;
            for (int n = 0; n < N; ++n) {
                for (int s = 0; s < H_out * W_out; ++s) {
                    sum += d_output[(n * C_out + c) * H_out * W_out + s];
                }
            }
            d_bias[c] = sum;
        }
    }
}

// =====================================================================
// Test runner
// =====================================================================

struct ConvTestCase {
    const char* name;
    int N, C_in, H, W;
    int C_out, kH, kW;
    int padH, padW;
    int strideH, strideW;
    int dilH, dilW;
    int groups;
    bool use_bias;
};

static bool run_forward_test(const ConvTestCase& tc) {
    int H_out = conv_out_size(tc.H, tc.padH, tc.dilH, tc.kH, tc.strideH);
    int W_out = conv_out_size(tc.W, tc.padW, tc.dilW, tc.kW, tc.strideW);

    int C_in_per_group = tc.C_in / tc.groups;
    int input_size  = tc.N * tc.C_in * tc.H * tc.W;
    int weight_size = tc.C_out * C_in_per_group * tc.kH * tc.kW;
    int output_size = tc.N * tc.C_out * H_out * W_out;

    std::vector<float> h_input(input_size), h_weight(weight_size), h_bias(tc.C_out);
    test::fill_random(h_input,  -1.0f, 1.0f, 42);
    test::fill_random(h_weight, -0.5f, 0.5f, 43);
    test::fill_random(h_bias,   -0.1f, 0.1f, 44);

    float* d_input  = test::to_device(h_input);
    float* d_weight = test::to_device(h_weight);
    float* d_bias   = tc.use_bias ? test::to_device(h_bias) : nullptr;
    float* d_output = test::to_device_zeroed<float>(output_size);

    nn::conv2d_forward(d_input, d_weight, d_bias, d_output,
                       tc.N, tc.C_in, tc.H, tc.W,
                       tc.C_out, tc.kH, tc.kW,
                       tc.padH, tc.padW, tc.strideH, tc.strideW,
                       tc.dilH, tc.dilW, tc.groups);
    cudaDeviceSynchronize();

    auto gpu = test::to_host(d_output, output_size);

    std::vector<float> ref(output_size);
    conv2d_fwd_cpu(h_input.data(), h_weight.data(),
                   tc.use_bias ? h_bias.data() : nullptr,
                   ref.data(),
                   tc.N, tc.C_in, tc.H, tc.W,
                   tc.C_out, tc.kH, tc.kW,
                   tc.padH, tc.padW, tc.strideH, tc.strideW,
                   tc.dilH, tc.dilW, tc.groups);

    bool ok = test::check_arrays(gpu, ref, 1e-3f, 1e-2f, tc.name);

    test::free_device(d_input);
    test::free_device(d_weight);
    if (d_bias) test::free_device(d_bias);
    test::free_device(d_output);
    return ok;
}

static bool run_backward_data_test(const ConvTestCase& tc) {
    int H_out = conv_out_size(tc.H, tc.padH, tc.dilH, tc.kH, tc.strideH);
    int W_out = conv_out_size(tc.W, tc.padW, tc.dilW, tc.kW, tc.strideW);

    int C_in_per_group = tc.C_in / tc.groups;
    int input_size  = tc.N * tc.C_in * tc.H * tc.W;
    int weight_size = tc.C_out * C_in_per_group * tc.kH * tc.kW;
    int output_size = tc.N * tc.C_out * H_out * W_out;

    std::vector<float> h_d_output(output_size), h_weight(weight_size);
    test::fill_random(h_d_output, -1.0f, 1.0f, 45);
    test::fill_random(h_weight,   -0.5f, 0.5f, 43);

    float* d_d_output = test::to_device(h_d_output);
    float* d_weight   = test::to_device(h_weight);
    float* d_d_input  = test::to_device_zeroed<float>(input_size);

    nn::conv2d_backward_data(d_d_output, d_weight, d_d_input,
                             tc.N, tc.C_in, tc.H, tc.W,
                             tc.C_out, tc.kH, tc.kW,
                             tc.padH, tc.padW, tc.strideH, tc.strideW,
                             tc.dilH, tc.dilW, tc.groups);
    cudaDeviceSynchronize();

    auto gpu = test::to_host(d_d_input, input_size);

    std::vector<float> ref(input_size);
    conv2d_bwd_data_cpu(h_d_output.data(), h_weight.data(), ref.data(),
                        tc.N, tc.C_in, tc.H, tc.W,
                        tc.C_out, tc.kH, tc.kW,
                        tc.padH, tc.padW, tc.strideH, tc.strideW,
                        tc.dilH, tc.dilW, tc.groups);

    std::string label = std::string(tc.name) + "_bwd_data";
    bool ok = test::check_arrays(gpu, ref, 1e-3f, 1e-2f, label.c_str());

    test::free_device(d_d_output);
    test::free_device(d_weight);
    test::free_device(d_d_input);
    return ok;
}

static bool run_backward_weight_test(const ConvTestCase& tc, float beta = 0.0f) {
    int H_out = conv_out_size(tc.H, tc.padH, tc.dilH, tc.kH, tc.strideH);
    int W_out = conv_out_size(tc.W, tc.padW, tc.dilW, tc.kW, tc.strideW);

    int C_in_per_group = tc.C_in / tc.groups;
    int input_size  = tc.N * tc.C_in * tc.H * tc.W;
    int weight_size = tc.C_out * C_in_per_group * tc.kH * tc.kW;
    int output_size = tc.N * tc.C_out * H_out * W_out;

    std::vector<float> h_input(input_size), h_d_output(output_size);
    std::vector<float> h_d_weight(weight_size, 1.0f);  // init to 1 for accumulate test
    test::fill_random(h_input,    -1.0f, 1.0f, 42);
    test::fill_random(h_d_output, -1.0f, 1.0f, 45);

    float* d_input    = test::to_device(h_input);
    float* d_d_output = test::to_device(h_d_output);
    float* d_d_weight = test::to_device(h_d_weight);
    float* d_d_bias   = tc.use_bias ? test::to_device_zeroed<float>(tc.C_out) : nullptr;

    nn::conv2d_backward_weight(d_d_output, d_input, d_d_weight, d_d_bias,
                               tc.N, tc.C_in, tc.H, tc.W,
                               tc.C_out, tc.kH, tc.kW,
                               tc.padH, tc.padW, tc.strideH, tc.strideW,
                               tc.dilH, tc.dilW, tc.groups, beta);
    cudaDeviceSynchronize();

    auto gpu_weight = test::to_host(d_d_weight, weight_size);

    std::vector<float> ref_weight(h_d_weight);  // copy initial values for beta
    conv2d_bwd_weight_cpu(h_d_output.data(), h_input.data(), ref_weight.data(),
                          nullptr,
                          tc.N, tc.C_in, tc.H, tc.W,
                          tc.C_out, tc.kH, tc.kW,
                          tc.padH, tc.padW, tc.strideH, tc.strideW,
                          tc.dilH, tc.dilW, tc.groups, beta);

    std::string label = std::string(tc.name) + "_bwd_weight";
    if (beta > 0.0f) label += "_acc";
    bool ok = test::check_arrays(gpu_weight, ref_weight, 1e-2f, 1e-1f, label.c_str());

    if (tc.use_bias && d_d_bias) {
        auto gpu_bias = test::to_host(d_d_bias, tc.C_out);
        std::vector<float> ref_bias(tc.C_out);
        // Use a dummy weight buffer since we only want d_bias
        std::vector<float> dummy_weight(weight_size, 0.0f);
        conv2d_bwd_weight_cpu(h_d_output.data(), h_input.data(), dummy_weight.data(),
                              ref_bias.data(),
                              tc.N, tc.C_in, tc.H, tc.W,
                              tc.C_out, tc.kH, tc.kW,
                              tc.padH, tc.padW, tc.strideH, tc.strideW,
                              tc.dilH, tc.dilW, tc.groups, 0.0f);
        std::string bias_label = std::string(tc.name) + "_bwd_bias";
        ok = test::check_arrays(gpu_bias, ref_bias, 1e-3f, 1e-2f, bias_label.c_str()) && ok;
    }

    test::free_device(d_input);
    test::free_device(d_d_output);
    test::free_device(d_d_weight);
    if (d_d_bias) test::free_device(d_d_bias);
    return ok;
}

int main() {
    std::printf("=== Test: Conv2d ===\n");
    int failures = 0;

    // Forward test cases
    ConvTestCase fwd_cases[] = {
        // name, N, C_in, H, W, C_out, kH, kW, padH, padW, strideH, strideW, dilH, dilW, groups, bias
        {"conv2d_fwd_1x1",       2, 4,  8,  8,  8, 1, 1, 0, 0, 1, 1, 1, 1, 1, true},
        {"conv2d_fwd_3x3",       2, 4,  8,  8,  8, 3, 3, 1, 1, 1, 1, 1, 1, 1, true},
        {"conv2d_fwd_3x3_nobias",2, 4,  8,  8,  8, 3, 3, 1, 1, 1, 1, 1, 1, 1, false},
        {"conv2d_fwd_stride2",   2, 4, 16, 16,  8, 3, 3, 1, 1, 2, 2, 1, 1, 1, true},
        {"conv2d_fwd_dilation",  2, 4, 16, 16,  8, 3, 3, 2, 2, 1, 1, 2, 2, 1, true},
        {"conv2d_fwd_groups",    2, 8,  8,  8, 16, 3, 3, 1, 1, 1, 1, 1, 1, 2, true},
        {"conv2d_fwd_depthwise", 2, 4,  8,  8,  4, 3, 3, 1, 1, 1, 1, 1, 1, 4, true},
    };

    for (const auto& tc : fwd_cases) {
        if (!run_forward_test(tc)) failures++;
    }

    // Backward data test
    ConvTestCase bwd_data_case = {"conv2d_3x3", 2, 4, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, false};
    if (!run_backward_data_test(bwd_data_case)) failures++;

    // Backward data with stride
    ConvTestCase bwd_data_stride = {"conv2d_stride2", 2, 4, 16, 16, 8, 3, 3, 1, 1, 2, 2, 1, 1, 1, false};
    if (!run_backward_data_test(bwd_data_stride)) failures++;

    // Backward data with groups
    ConvTestCase bwd_data_groups = {"conv2d_groups", 2, 8, 8, 8, 16, 3, 3, 1, 1, 1, 1, 1, 1, 2, false};
    if (!run_backward_data_test(bwd_data_groups)) failures++;

    // Backward weight test (overwrite)
    ConvTestCase bwd_weight_case = {"conv2d_3x3", 2, 4, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, true};
    if (!run_backward_weight_test(bwd_weight_case, 0.0f)) failures++;

    // Backward weight test (accumulate)
    if (!run_backward_weight_test(bwd_weight_case, 1.0f)) failures++;

    // Backward weight with groups
    ConvTestCase bwd_weight_groups = {"conv2d_groups", 2, 8, 8, 8, 16, 3, 3, 1, 1, 1, 1, 1, 1, 2, true};
    if (!run_backward_weight_test(bwd_weight_groups, 0.0f)) failures++;

    // ================================================================
    // Autograd integration test
    // ================================================================
    {
        // N=1, C_in=2, H=4, W=4, C_out=3, 3x3 kernel, pad=1
        int ag_Cin = 2, ag_Cout = 3;
        std::vector<float> h_in(1 * ag_Cin * 4 * 4), h_w(ag_Cout * ag_Cin * 3 * 3);
        test::fill_random(h_in, -1.0f, 1.0f, 100);
        test::fill_random(h_w, -0.5f, 0.5f, 101);

        Tensor input  = Tensor::zeros({1, (int64_t)ag_Cin, 4, 4});
        Tensor weight = Tensor::zeros({(int64_t)ag_Cout, (int64_t)ag_Cin, 3, 3});
        // Tensor::zeros memsets on Forge's non-blocking stream, which the
        // legacy-stream cudaMemcpy below does not order against: without the
        // first sync the memset can land after the copy and zero the input;
        // without the second, conv2d can read before the pageable copy's DMA
        // has landed.
        FORGE_CHECK(cudaDeviceSynchronize());
        cudaMemcpy(input.data<float>(), h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(weight.data<float>(), h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice);
        FORGE_CHECK(cudaDeviceSynchronize());
        input.set_requires_grad_(true);
        weight.set_requires_grad_(true);

        Tensor output = conv2d(input, weight, 1, 1, 1, 1);
        Tensor loss = sum(output);
        backward(loss);

        Tensor ig = input.grad();
        Tensor wg = weight.grad();

        bool ok = true;
        if (!ig.defined() || ig.numel() != input.numel()) {
            std::fprintf(stderr, "FAIL [conv2d_autograd]: input.grad() bad\n");
            ok = false;
        }
        if (!wg.defined() || wg.numel() != weight.numel()) {
            std::fprintf(stderr, "FAIL [conv2d_autograd]: weight.grad() bad\n");
            ok = false;
        }

        // Verify values: loss = sum(output) => d_output = ones.
        // Forge runs on a non-blocking stream, which the legacy-stream
        // cudaMemcpy in to_host() does not wait for -- sync first.
        if (ok) {
            FORGE_CHECK(cudaDeviceSynchronize());
            auto ig_h  = test::to_host(ig.data<float>(), ig.numel());
            auto wg_h  = test::to_host(wg.data<float>(), wg.numel());
            auto out_h = test::to_host(output.data<float>(), output.numel());
            std::vector<float> ones(out_h.size(), 1.0f);
            std::vector<float> ref_out(out_h.size()), ref_ig(ig_h.size()), ref_wg(wg_h.size());
            conv2d_fwd_cpu(h_in.data(), h_w.data(), nullptr, ref_out.data(),
                           1, ag_Cin, 4, 4, ag_Cout, 3, 3, 1, 1, 1, 1, 1, 1, 1);
            conv2d_bwd_data_cpu(ones.data(), h_w.data(), ref_ig.data(),
                                1, ag_Cin, 4, 4, ag_Cout, 3, 3, 1, 1, 1, 1, 1, 1, 1);
            conv2d_bwd_weight_cpu(ones.data(), h_in.data(), ref_wg.data(), nullptr,
                                  1, ag_Cin, 4, 4, ag_Cout, 3, 3, 1, 1, 1, 1, 1, 1, 1, 0.0f);
            ok &= test::check_arrays(out_h, ref_out, 1e-4f, 1e-3f, "conv2d_autograd_fwd");
            ok &= test::check_arrays(ig_h, ref_ig, 1e-4f, 1e-3f, "conv2d_autograd_dinput");
            ok &= test::check_arrays(wg_h, ref_wg, 1e-4f, 1e-3f, "conv2d_autograd_dweight");
        }

        if (ok) std::printf("  PASS [conv2d_autograd]\n");
        else failures++;
    }

    // ================================================================
    // Stream ordering: conv2d consumes a tensor produced on Forge's
    // current stream, with no host sync in between. The forward and
    // backward kernels must run on that stream, not the legacy default
    // stream (which a non-blocking stream does not order against).
    // ================================================================
    for (int rep = 0; rep < 3; ++rep) {  // later reps run with warm allocator pools
        const int N = 8, C_in = 32, H = 64, W = 64, C_out = 32;
        std::vector<float> h_in((size_t)N * C_in * H * W), h_w((size_t)C_out * C_in * 9);
        test::fill_random(h_in, -1.0f, 1.0f, 200);
        test::fill_random(h_w, -0.2f, 0.2f, 201);
        float* d_in = test::to_device(h_in);
        float* d_w  = test::to_device(h_w);
        Tensor x0 = Tensor::from_ptr(d_in, {N, C_in, H, W});
        Tensor weight = Tensor::from_ptr(d_w, {C_out, C_in, 3, 3});
        weight.set_requires_grad_(true);

        // Producers on the Forge stream, immediately followed by conv2d.
        Tensor x = mul(add(x0, 1.0f), 2.0f);       // x = 2 * (x0 + 1)
        x.set_requires_grad_(true);
        Tensor y = conv2d(x, weight, 1, 1, 1, 1);
        Tensor loss = sum(mul(y, 3.0f));           // d_output = 3
        backward(loss);
        FORGE_CHECK(cudaDeviceSynchronize());

        std::vector<float> hx(h_in.size());
        for (size_t i = 0; i < hx.size(); ++i) hx[i] = 2.0f * (h_in[i] + 1.0f);
        std::vector<float> ref_y((size_t)N * C_out * H * W), ref_dx(hx.size()), ref_dw(h_w.size());
        std::vector<float> dout(ref_y.size(), 3.0f);
        conv2d_fwd_cpu(hx.data(), h_w.data(), nullptr, ref_y.data(),
                       N, C_in, H, W, C_out, 3, 3, 1, 1, 1, 1, 1, 1, 1);
        conv2d_bwd_data_cpu(dout.data(), h_w.data(), ref_dx.data(),
                            N, C_in, H, W, C_out, 3, 3, 1, 1, 1, 1, 1, 1, 1);
        conv2d_bwd_weight_cpu(dout.data(), hx.data(), ref_dw.data(), nullptr,
                              N, C_in, H, W, C_out, 3, 3, 1, 1, 1, 1, 1, 1, 1, 0.0f);

        bool ok = x.grad().defined() && weight.grad().defined();
        if (ok) {
            auto gy  = test::to_host(y.data<float>(), y.numel());
            auto gdx = test::to_host(x.grad().data<float>(), x.numel());
            auto gdw = test::to_host(weight.grad().data<float>(), weight.numel());
            // dW sums N*H*W = 32768 products of magnitude ~12: allow FP32
            // accumulation-order error on |dW| up to ~1e3.
            ok &= test::check_arrays(gy,  ref_y,  1e-3f, 1e-3f, "conv2d_stream_fwd");
            ok &= test::check_arrays(gdx, ref_dx, 1e-3f, 1e-3f, "conv2d_stream_dinput");
            ok &= test::check_arrays(gdw, ref_dw, 1e-1f, 1e-3f, "conv2d_stream_dweight");
        } else {
            std::fprintf(stderr, "FAIL [conv2d_stream]: missing gradients\n");
        }
        if (!ok) failures++;
        y = Tensor(); loss = Tensor(); x = Tensor(); weight = Tensor(); x0 = Tensor();
        test::free_device(d_in);
        test::free_device(d_w);
    }

    if (failures > 0) {
        std::printf("FAILED (%d failures)\n", failures);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
