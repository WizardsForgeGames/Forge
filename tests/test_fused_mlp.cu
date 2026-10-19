// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/fused_mlp.cuh>
#include "test_helpers.cuh"
#include <algorithm>
#include <cmath>
#include <random>
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

// ============================================================
//  Double-precision analytic reference (forward + backward)
// ============================================================
//
// Used by the dispatch-path sweep below. The backward reference consumes the
// GPU's own acts_save (pre-activations), exactly as the backward kernel does,
// so it checks the backward kernel in isolation: a TF32 forward that moves a
// ReLU pre-activation across 0 cannot masquerade as a backward error.

static double ref_act(double x, nn::Activation a, double p) {
    switch (a) {
    case nn::Activation::CELU:    return x > 0 ? x : p * (std::exp(x / p) - 1);
    case nn::Activation::GELU: {
        double in = 0.7978845608 * (x + 0.044715 * x * x * x);
        return 0.5 * x * (1 + std::tanh(in));
    }
    case nn::Activation::ReLU:    return x > 0 ? x : 0;
    case nn::Activation::Sigmoid: return 1 / (1 + std::exp(-x));
    case nn::Activation::SiLU:    return x / (1 + std::exp(-x));
    case nn::Activation::Tanh:    return std::tanh(x);
    }
    return x;
}

static double ref_act_grad(double z, nn::Activation a, double p) {
    switch (a) {
    case nn::Activation::CELU:    return z > 0 ? 1 : std::exp(z / p);
    case nn::Activation::GELU: {
        double in = 0.7978845608 * (z + 0.044715 * z * z * z);
        double t = std::tanh(in);
        return 0.5 * (1 + t)
             + 0.5 * z * (1 - t * t) * 0.7978845608 * (1 + 3 * 0.044715 * z * z);
    }
    case nn::Activation::ReLU:    return z > 0 ? 1 : 0;
    case nn::Activation::Sigmoid: { double s = 1 / (1 + std::exp(-z)); return s * (1 - s); }
    case nn::Activation::SiLU:    { double s = 1 / (1 + std::exp(-z)); return s + z * s * (1 - s); }
    case nn::Activation::Tanh:    { double t = std::tanh(z); return 1 - t * t; }
    }
    return 1;
}

struct MlpRef { std::vector<double> out, acts, dw, din; };

// acts_ref: exact pre-activations (each layer fed by the GPU's previous-layer
// pre-activations from gpu_acts, so forward errors are measured per layer).
// out/dw/din: computed from gpu_acts.
static void ref_mlp(const nn::MlpDesc& d, int B,
                    const std::vector<float>& x, const std::vector<float>& w,
                    const std::vector<float>& dout,
                    const std::vector<float>& gpu_acts, MlpRef& r)
{
    const int IN = d.n_inputs, H = d.hidden, L = d.n_layers, O = d.n_outputs;
    nn::MlpDesc nd = d; nd.normalize = false;
    const int norm_off = nn::mlp_weight_count(nd);
    const int aps = H * L + O;
    r.out.assign((size_t)B * O, 0.0);
    r.acts.assign((size_t)B * aps, 0.0);
    r.dw.assign(nn::mlp_weight_count(d), 0.0);
    r.din.assign((size_t)B * IN, 0.0);

    std::vector<int> woff(L + 1);
    for (int l = 0, off = 0; l <= L; l++) {
        woff[l] = off;
        if (l < L) off += H * (l == 0 ? IN : H) + H;
    }

    std::vector<std::vector<double>> a(L + 1), z(L + 1);
    for (int b = 0; b < B; b++) {
        a[0].assign(IN, 0.0);
        for (int k = 0; k < IN; k++) {
            double v = x[(size_t)b * IN + k];
            if (d.normalize) {
                double lb = w[norm_off + k], ub = w[norm_off + IN + k];
                v = 2.0 * (v - lb) / (ub - lb) - 1.0;
            }
            a[0][k] = v;
        }
        for (int l = 0; l < L; l++) {
            int in = (l == 0) ? IN : H;
            const float* W = &w[woff[l]];
            const float* bias = W + H * in;
            z[l + 1].assign(H, 0.0);
            a[l + 1].assign(H, 0.0);
            for (int j = 0; j < H; j++) {
                double s = bias[j];
                for (int k = 0; k < in; k++) s += (double)W[j * in + k] * a[l][k];
                size_t ai = (size_t)b * aps + l * H + j;
                r.acts[ai] = s;
                z[l + 1][j] = gpu_acts[ai];
                a[l + 1][j] = ref_act(gpu_acts[ai], d.act, d.act_param);
            }
        }
        const float* Wo = &w[woff[L]];
        const float* bo = Wo + O * H;
        std::vector<double> g(O), da(H, 0.0);
        for (int j = 0; j < O; j++) {
            double s = bo[j];
            for (int k = 0; k < H; k++) s += (double)Wo[j * H + k] * a[L][k];
            r.acts[(size_t)b * aps + L * H + j] = s;
            r.out[(size_t)b * O + j] = s * d.output_scale;
            g[j] = (double)dout[(size_t)b * O + j] * d.output_scale;
        }
        for (int j = 0; j < O; j++) {
            for (int k = 0; k < H; k++) {
                r.dw[woff[L] + j * H + k] += g[j] * a[L][k];
                da[k] += Wo[j * H + k] * g[j];
            }
            r.dw[woff[L] + O * H + j] += g[j];
        }
        for (int l = L - 1; l >= 0; l--) {
            int in = (l == 0) ? IN : H;
            const float* W = &w[woff[l]];
            std::vector<double> dz(H), dprev(in, 0.0);
            for (int j = 0; j < H; j++)
                dz[j] = da[j] * ref_act_grad(z[l + 1][j], d.act, d.act_param);
            for (int j = 0; j < H; j++) {
                for (int k = 0; k < in; k++) {
                    r.dw[woff[l] + j * in + k] += dz[j] * a[l][k];
                    dprev[k] += W[j * in + k] * dz[j];
                }
                r.dw[woff[l] + H * in + j] += dz[j];
            }
            da.swap(dprev);
        }
        for (int k = 0; k < IN; k++) {
            double chain = 1.0;  // d(normalized)/d(raw) = 2 / (ub - lb)
            if (d.normalize)
                chain = 2.0 / ((double)w[norm_off + IN + k] - (double)w[norm_off + k]);
            r.din[(size_t)b * IN + k] = da[k] * chain;
        }
    }
}

// max |gpu - ref| / max |ref| over the first n entries (NaN -> +inf).
static double norm_err(const std::vector<float>& gpu, const std::vector<double>& ref, size_t n) {
    double mx = 0.0, me = 0.0;
    for (size_t i = 0; i < n; i++) {
        mx = std::max(mx, std::abs(ref[i]));
        double e = std::abs((double)gpu[i] - ref[i]);
        if (!(e <= me)) me = e;  // also catches NaN
    }
    if (std::isnan(me)) return INFINITY;
    return me / (mx + 1e-30);
}

struct SweepCase {
    const char* name;
    int IN, OUT, H, L, B;
    nn::Activation act;
    bool tf32, normalize, workspace;
    float beta;
};

// Tolerances are normalized errors (max|err| / max|ref|).
//  FP32 path: observed <= 1e-6 across the sweep; 1e-4 leaves ~100x margin
//  for accumulation-order differences (atomics / per-block reduction).
//  TF32 path: wmma TF32 keeps 10 explicit mantissa bits, so each operand
//  carries up to 2^-10 ~ 9.8e-4 relative error (the hardware drops the low
//  13 bits of an unconverted FP32 operand) and each product up to ~2e-3,
//  accumulated in FP32. Observed: per-layer pre-activations <= 8.2e-4,
//  dW <= 7.5e-4, d_input (L TF32 propagations deep) <= 4.6e-3. 1e-2 sits
//  ~2x above the worst observed while any indexing/race bug shows up as
//  O(0.1-1).
static bool run_sweep_case(const SweepCase& c, uint64_t seed) {
    nn::MlpDesc d;
    d.n_inputs = c.IN; d.n_outputs = c.OUT; d.hidden = c.H; d.n_layers = c.L;
    d.act = c.act; d.act_param = 0.7f; d.output_scale = 0.5f;
    d.normalize = c.normalize; d.use_tf32 = c.tf32;
    const int B = c.B;
    const int n_w = nn::mlp_weight_count(d);
    const int n_trained = c.normalize ? n_w - 2 * c.IN : n_w;
    const int n_acts = nn::mlp_acts_save_count(d, B);

    std::mt19937 rng((unsigned)seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> x((size_t)B * c.IN), w(n_w), dout((size_t)B * c.OUT), dw0(n_w);
    for (auto& v : x) v = u(rng);
    for (int l = 0, off = 0; l <= c.L; l++) {
        int in = (l == 0) ? c.IN : c.H;
        int out = (l == c.L) ? c.OUT : c.H;
        float bound = 1.5f / std::sqrt((float)in);
        for (int i = 0; i < out * in + out; i++) w[off + i] = u(rng) * bound;
        off += out * in + out;
        if (l == c.L && c.normalize) {
            for (int k = 0; k < c.IN; k++) {
                w[off + k]        = -1.5f + 0.3f * u(rng);  // lb
                w[off + c.IN + k] =  1.5f + 0.3f * u(rng);  // ub
            }
        }
    }
    for (auto& v : dout) v = u(rng);
    for (auto& v : dw0) v = u(rng);

    float* d_x    = test::to_device(x);
    float* d_w    = test::to_device(w);
    float* d_dout = test::to_device(dout);
    float* d_y    = test::to_device_zeroed<float>((size_t)B * c.OUT);
    float* d_acts = test::to_device_zeroed<float>(n_acts);
    float* d_din  = test::to_device_zeroed<float>((size_t)B * c.IN);
    float* d_dw   = test::to_device(dw0);  // beta=0 must overwrite this
    float* d_ws   = nullptr;
    int ws_count = 0;
    if (c.workspace) {
        ws_count = nn::fused_mlp_backward_workspace_count(d, B);
        if (ws_count > 0) d_ws = test::to_device_zeroed<float>(ws_count);
    }

    nn::fused_mlp_forward(d, d_x, d_w, d_y, B, d_acts);
    nn::fused_mlp_backward(d, d_x, d_w, d_acts, d_dout, d_din, d_dw, B, c.beta,
                           nullptr, d_ws);
    FORGE_CHECK(cudaDeviceSynchronize());

    auto g_y    = test::to_host(d_y, (size_t)B * c.OUT);
    auto g_acts = test::to_host(d_acts, n_acts);
    auto g_dw   = test::to_host(d_dw, n_w);
    auto g_din  = test::to_host(d_din, (size_t)B * c.IN);
    test::free_device(d_x); test::free_device(d_w); test::free_device(d_dout);
    test::free_device(d_y); test::free_device(d_acts); test::free_device(d_din);
    test::free_device(d_dw); test::free_device(d_ws);

    MlpRef r;
    ref_mlp(d, B, x, w, dout, g_acts, r);
    if (c.beta != 0.0f)
        for (int i = 0; i < n_trained; i++) r.dw[i] += c.beta * dw0[i];

    double e_out  = norm_err(g_y, r.out, g_y.size());
    double e_acts = norm_err(g_acts, r.acts, g_acts.size());
    double e_dw   = norm_err(g_dw, r.dw, n_trained);
    double e_din  = norm_err(g_din, r.din, g_din.size());
    double tol = c.tf32 ? 1e-2 : 1e-4;
    bool ok = e_out < tol && e_acts < tol && e_dw < tol && e_din < tol;
    std::printf("  %s [%s] IN=%d H=%d L=%d B=%d tf32=%d norm=%d ws=%d beta=%g: "
                "out %.1e acts %.1e dW %.1e dIn %.1e (tol %.0e)\n",
                ok ? "PASS" : "FAIL", c.name, c.IN, c.H, c.L, B, (int)c.tf32,
                (int)c.normalize, ws_count > 0, c.beta, e_out, e_acts, e_dw, e_din, tol);
    return ok;
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

    // ------ Dispatch-path sweep vs double-precision analytic reference ------
    // Covers what the cases above do not reach: multi-block atomics, the
    // persistent grid-stride loop (B > SMs * tile), L2 mode (H > 128), and
    // the TF32 Tensor Core kernels.
    {
        using A = nn::Activation;
        const SweepCase cases[] = {
            // name               IN OUT  H   L     B  act        tf32   norm   ws     beta
            {"fp32_atomic",        9,  2,  64, 5,  600, A::CELU,    false, false, false, 0.0f},
            {"fp32_persistent",    9,  2,  64, 5, 20000, A::SiLU,   false, false, false, 1.0f},
            {"fp32_l2_mode",       9,  2, 256, 3,  257, A::GELU,    false, false, false, 0.0f},
            {"tf32_single_tile",   9,  2,  64, 3,   16, A::Tanh,    true,  false, false, 0.0f},
            {"tf32_atomic",        9,  2,  64, 5,  600, A::CELU,    true,  false, false, 0.0f},
            {"tf32_h128",          9,  2, 128, 3, 4100, A::ReLU,    true,  false, false, 1.0f},
            {"tf32_h48",           7,  3,  48, 4,  777, A::Sigmoid, true,  false, false, 0.0f},
            {"tf32_in17",         17,  4,  32, 6, 5001, A::SiLU,    true,  false, false, 0.0f},
            {"tf32_in_gt_h",     130,  2,  64, 2,  300, A::GELU,    true,  false, false, 1.0f},
        };
        bool all_ok = true;
        uint64_t seed = 1000;
        for (const auto& c : cases) all_ok &= run_sweep_case(c, seed++);
        FORGE_ASSERT(all_ok, "Fused MLP dispatch-path sweep mismatch");
    }

    std::printf("ALL PASSED\n");
    return 0;
}
