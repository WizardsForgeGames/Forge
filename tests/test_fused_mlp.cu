// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/fused_mlp.cuh>
#include "test_helpers.cuh"
#include <cmath>
#include <vector>

using namespace forge;

// ============================================================
//  CPU reference: naive MLP forward
// ============================================================

static float cpu_celu(float x, float alpha) {
    return (x > 0.0f) ? x : alpha * (std::exp(x / alpha) - 1.0f);
}

static void cpu_mlp_forward(
    int n_inputs, int n_outputs, int hidden, int n_layers,
    float act_param, float output_scale, bool normalize,
    const float* input, const float* weights, float* output,
    float* acts_save,  // [B, hidden * n_layers + n_outputs] or nullptr
    int B)
{
    int w_off = 0;
    int acts_per_sample = hidden * n_layers + n_outputs;

    // Normalization bounds
    const float* lb = nullptr;
    const float* ub = nullptr;
    if (normalize) {
        // Compute total weights to find bounds at end
        int total = hidden * n_inputs + hidden; // layer 0
        for (int l = 1; l < n_layers; l++)
            total += hidden * hidden + hidden;
        total += n_outputs * hidden + n_outputs;
        lb = weights + total;
        ub = lb + n_inputs;
    }

    // Working buffers per sample
    std::vector<float> cur(n_inputs > hidden ? n_inputs : hidden);
    std::vector<float> next(hidden);

    for (int b = 0; b < B; b++) {
        // Load and normalize input
        for (int d = 0; d < n_inputs; d++) {
            float val = input[b * n_inputs + d];
            if (normalize)
                val = 2.0f * (val - lb[d]) / (ub[d] - lb[d]) - 1.0f;
            cur[d] = val;
        }

        w_off = 0;
        for (int l = 0; l < n_layers; l++) {
            int in_dim = (l == 0) ? n_inputs : hidden;
            const float* W = weights + w_off;
            const float* bias = W + hidden * in_dim;

            for (int j = 0; j < hidden; j++) {
                float sum = bias[j];
                for (int k = 0; k < in_dim; k++)
                    sum += W[j * in_dim + k] * cur[k];
                if (acts_save)
                    acts_save[b * acts_per_sample + l * hidden + j] = sum;
                next[j] = cpu_celu(sum, act_param);
            }

            for (int j = 0; j < hidden; j++)
                cur[j] = next[j];

            w_off += hidden * in_dim + hidden;
        }

        // Output layer
        const float* W = weights + w_off;
        const float* bias = W + n_outputs * hidden;
        for (int j = 0; j < n_outputs; j++) {
            float sum = bias[j];
            for (int k = 0; k < hidden; k++)
                sum += W[j * hidden + k] * cur[k];
            if (acts_save)
                acts_save[b * acts_per_sample + n_layers * hidden + j] = sum;
            output[b * n_outputs + j] = sum * output_scale;
        }
    }
}

// ============================================================
//  CPU reference: naive MLP backward (numerical gradient check)
// ============================================================

static float cpu_mlp_loss(
    int n_inputs, int n_outputs, int hidden, int n_layers,
    float act_param, float output_scale, bool normalize,
    const float* input, const float* weights, int B)
{
    std::vector<float> output(B * n_outputs);
    cpu_mlp_forward(n_inputs, n_outputs, hidden, n_layers,
                    act_param, output_scale, normalize,
                    input, weights, output.data(), nullptr, B);

    // Sum of all outputs as a simple scalar loss
    float loss = 0.0f;
    for (int i = 0; i < B * n_outputs; i++)
        loss += output[i];
    return loss;
}

int main() {
    std::printf("=== Test: Fused MLP ===\n");

    // ------ Forward test: H=64, L=3, in=9, out=2 ------
    {
        const int B = 16, IN = 9, OUT = 2, H = 64, L = 3;
        float act_param = 1.0f;
        float out_scale = 0.01f;

        nn::MlpDesc desc;
        desc.n_inputs = IN; desc.n_outputs = OUT; desc.hidden = H;
        desc.n_layers = L; desc.act = nn::Activation::CELU;
        desc.act_param = act_param; desc.output_scale = out_scale;
        desc.normalize = false;

        int n_weights = nn::mlp_weight_count(desc);
        int n_acts = nn::mlp_acts_save_count(desc, B);

        std::vector<float> h_input(B * IN);
        std::vector<float> h_weights(n_weights);
        test::fill_random(h_input, -1.0f, 1.0f, 42);
        test::fill_random(h_weights, -0.1f, 0.1f, 43);

        // CPU reference
        std::vector<float> ref_output(B * OUT);
        std::vector<float> ref_acts(n_acts);
        cpu_mlp_forward(IN, OUT, H, L, act_param, out_scale, false,
                        h_input.data(), h_weights.data(),
                        ref_output.data(), ref_acts.data(), B);

        // GPU
        float* d_input   = test::to_device(h_input);
        float* d_weights = test::to_device(h_weights);
        float* d_output  = test::to_device_zeroed<float>(B * OUT);
        float* d_acts    = test::to_device_zeroed<float>(n_acts);

        nn::fused_mlp_forward(desc, d_input, d_weights, d_output, B, d_acts);
        cudaDeviceSynchronize();

        auto gpu_out = test::to_host(d_output, B * OUT);
        auto gpu_acts = test::to_host(d_acts, n_acts);

        FORGE_ASSERT(test::check_arrays(gpu_out, ref_output, 1e-4f, 1e-3f, "fused_fwd_output"),
                     "Fused MLP forward output mismatch");
        FORGE_ASSERT(test::check_arrays(gpu_acts, ref_acts, 1e-4f, 1e-3f, "fused_fwd_acts"),
                     "Fused MLP forward acts_save mismatch");

        test::free_device(d_input);
        test::free_device(d_weights);
        test::free_device(d_output);
        test::free_device(d_acts);
    }

    // ------ Forward test: H=128, L=5 (paper's model architecture) ------
    {
        const int B = 32, IN = 9, OUT = 2, H = 64, L = 5;
        float act_param = 1.0f;
        float out_scale = 0.01f;

        nn::MlpDesc desc;
        desc.n_inputs = IN; desc.n_outputs = OUT; desc.hidden = H;
        desc.n_layers = L; desc.act = nn::Activation::CELU;
        desc.act_param = act_param; desc.output_scale = out_scale;
        desc.normalize = false;

        int n_weights = nn::mlp_weight_count(desc);

        std::vector<float> h_input(B * IN);
        std::vector<float> h_weights(n_weights);
        test::fill_random(h_input, -1.0f, 1.0f, 50);
        test::fill_random(h_weights, -0.05f, 0.05f, 51);

        std::vector<float> ref_output(B * OUT);
        cpu_mlp_forward(IN, OUT, H, L, act_param, out_scale, false,
                        h_input.data(), h_weights.data(),
                        ref_output.data(), nullptr, B);

        float* d_input   = test::to_device(h_input);
        float* d_weights = test::to_device(h_weights);
        float* d_output  = test::to_device_zeroed<float>(B * OUT);

        nn::fused_mlp_forward(desc, d_input, d_weights, d_output, B, nullptr);
        cudaDeviceSynchronize();

        auto gpu_out = test::to_host(d_output, B * OUT);
        FORGE_ASSERT(test::check_arrays(gpu_out, ref_output, 1e-3f, 1e-2f, "fused_fwd_L5"),
                     "Fused MLP forward L=5 mismatch");

        test::free_device(d_input);
        test::free_device(d_weights);
        test::free_device(d_output);
    }

    // ------ Backward test: numerical gradient check ------
    {
        const int B = 8, IN = 4, OUT = 2, H = 16, L = 2;
        float act_param = 1.0f;
        float out_scale = 1.0f;  // no scaling for cleaner gradient check

        nn::MlpDesc desc;
        desc.n_inputs = IN; desc.n_outputs = OUT; desc.hidden = H;
        desc.n_layers = L; desc.act = nn::Activation::CELU;
        desc.act_param = act_param; desc.output_scale = out_scale;
        desc.normalize = false;

        int n_weights = nn::mlp_weight_count(desc);
        int n_acts = nn::mlp_acts_save_count(desc, B);

        std::vector<float> h_input(B * IN);
        std::vector<float> h_weights(n_weights);
        test::fill_random(h_input, -0.5f, 0.5f, 60);
        test::fill_random(h_weights, -0.3f, 0.3f, 61);

        // GPU forward + backward
        float* d_input   = test::to_device(h_input);
        float* d_weights = test::to_device(h_weights);
        float* d_output  = test::to_device_zeroed<float>(B * OUT);
        float* d_acts    = test::to_device_zeroed<float>(n_acts);
        float* d_dout    = test::to_device_zeroed<float>(B * OUT);
        float* d_dinput  = test::to_device_zeroed<float>(B * IN);
        float* d_dw      = test::to_device_zeroed<float>(n_weights);

        // Forward
        nn::fused_mlp_forward(desc, d_input, d_weights, d_output, B, d_acts);

        // Set d_output = all ones (gradient of sum loss)
        std::vector<float> h_dout(B * OUT, 1.0f);
        cudaMemcpy(d_dout, h_dout.data(), h_dout.size() * sizeof(float),
                   cudaMemcpyHostToDevice);

        // Backward
        nn::fused_mlp_backward(desc, d_input, d_weights, d_acts, d_dout,
                               d_dinput, d_dw, B, 0.0f);
        cudaDeviceSynchronize();

        auto gpu_dw = test::to_host(d_dw, n_weights);
        auto gpu_dinput = test::to_host(d_dinput, B * IN);

        // Numerical gradient check for weights
        float eps = 1e-3f;
        int n_check = (n_weights < 50) ? n_weights : 50;  // check first 50
        bool weight_grad_ok = true;
        float max_weight_err = 0.0f;
        for (int i = 0; i < n_check; i++) {
            float orig = h_weights[i];

            h_weights[i] = orig + eps;
            float loss_plus = cpu_mlp_loss(IN, OUT, H, L, act_param, out_scale, false,
                                           h_input.data(), h_weights.data(), B);
            h_weights[i] = orig - eps;
            float loss_minus = cpu_mlp_loss(IN, OUT, H, L, act_param, out_scale, false,
                                            h_input.data(), h_weights.data(), B);
            h_weights[i] = orig;

            float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
            float err = std::abs(gpu_dw[i] - num_grad);
            float denom = std::max(std::abs(num_grad), std::abs(gpu_dw[i])) + 1e-7f;
            float rel_err = err / denom;

            if (rel_err > max_weight_err) max_weight_err = rel_err;
            if (rel_err > 0.05f && err > 1e-4f) {
                std::fprintf(stderr, "  weight grad[%d]: gpu=%.6f num=%.6f rel_err=%.4f\n",
                             i, gpu_dw[i], num_grad, rel_err);
                weight_grad_ok = false;
            }
        }
        std::printf("  weight grad max_rel_err=%.6f (%d checked)\n", max_weight_err, n_check);
        FORGE_ASSERT(weight_grad_ok, "Fused MLP backward weight gradient mismatch");
        std::printf("  PASS [fused_bwd_weights]\n");

        // Numerical gradient check for inputs
        bool input_grad_ok = true;
        float max_input_err = 0.0f;
        for (int i = 0; i < B * IN; i++) {
            float orig = h_input[i];

            h_input[i] = orig + eps;
            float loss_plus = cpu_mlp_loss(IN, OUT, H, L, act_param, out_scale, false,
                                           h_input.data(), h_weights.data(), B);
            h_input[i] = orig - eps;
            float loss_minus = cpu_mlp_loss(IN, OUT, H, L, act_param, out_scale, false,
                                            h_input.data(), h_weights.data(), B);
            h_input[i] = orig;

            float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
            float err = std::abs(gpu_dinput[i] - num_grad);
            float denom = std::max(std::abs(num_grad), std::abs(gpu_dinput[i])) + 1e-7f;
            float rel_err = err / denom;

            if (rel_err > max_input_err) max_input_err = rel_err;
            if (rel_err > 0.05f && err > 1e-4f) {
                std::fprintf(stderr, "  input grad[%d]: gpu=%.6f num=%.6f rel_err=%.4f\n",
                             i, gpu_dinput[i], num_grad, rel_err);
                input_grad_ok = false;
            }
        }
        std::printf("  input grad max_rel_err=%.6f (%d checked)\n", max_input_err, B * IN);
        FORGE_ASSERT(input_grad_ok, "Fused MLP backward input gradient mismatch");
        std::printf("  PASS [fused_bwd_inputs]\n");

        test::free_device(d_input);
        test::free_device(d_weights);
        test::free_device(d_output);
        test::free_device(d_acts);
        test::free_device(d_dout);
        test::free_device(d_dinput);
        test::free_device(d_dw);
    }

    std::printf("ALL PASSED\n");
    return 0;
}
