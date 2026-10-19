// Forge GPU Compute Library -- Python Bindings
// Copyright (c) 2026 Darren Sander. MIT License.
//
// pybind11 wrappers for all Forge C++ launchers.
// Each function accepts torch::Tensor, extracts data pointers + dims,
// and calls the corresponding C++ launcher on the current CUDA stream.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <forge/optim/adam.cuh>
#include <forge/optim/grad_clip.cuh>
#include <forge/optim/lr_schedule.h>
#include <forge/nn/activations.cuh>
#include <forge/nn/fused_mlp.cuh>
#include <forge/nn/mse.cuh>
#include <forge/nn/l1.cuh>
#include <forge/nn/rng.cuh>
#include <forge/nn/reduce.cuh>
#include <forge/nn/softmax_ce.cuh>
#include <forge/nn/embedding.cuh>
#include <forge/nn/bias.cuh>
#include <forge/nn/fused_gemm_softmax_ce.cuh>
#include <forge/graph.h>
#include <forge/quant/quantize.cuh>
#include <forge/quant/dequantize.cuh>
#include <forge/gemm/fp32_tiled.cuh>
#include <forge/gemm/int8_dense.cuh>
#include <forge/gemm/int8_sparse.cuh>
#include <forge/sparsity/prune_compress.cuh>
#include <forge/sparsity/pack_metadata.cuh>
#include <forge/nn/conv2d.cuh>
#include <forge/nn/conv3d.cuh>
#include <forge/nn/conv3d_implicit.cuh>
#include <forge/nn/conv3d_winograd.cuh>
#include <forge/nn/group_norm.cuh>
#include <forge/sim/thermal.cuh>
#include <forge/sim/thermal_datagen.cuh>
#include <forge/sim/stefan.cuh>
#include <forge/sim/stefan_datagen.cuh>
#include <vector>
#include <pybind11/stl.h>

namespace py = pybind11;

static cudaStream_t current_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}

// ============================= Optimizer =============================

void adam_step_binding(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor m, torch::Tensor v,
    float base_lr, torch::Tensor step_scalars,
    float beta1, float beta2, float eps, float weight_decay,
    c10::optional<torch::Tensor> grad_scale,
    bool do_clamp, float clamp_lo, float clamp_hi,
    bool zero_grad)
{
    int n = param.numel();
    forge::optim::adam_step(
        param.data_ptr<float>(), grad.data_ptr<float>(),
        m.data_ptr<float>(), v.data_ptr<float>(), n,
        base_lr, step_scalars.data_ptr<float>(),
        beta1, beta2, eps, weight_decay,
        grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr,
        do_clamp, clamp_lo, clamp_hi, current_stream(),
        zero_grad);
}

void update_step_scalars_binding(
    torch::Tensor step_scalars, int adam_t,
    int warmup_steps, int total_steps, float min_lr_frac,
    float beta1, float beta2, int num_restarts)
{
    forge::optim::update_step_scalars(
        step_scalars.data_ptr<float>(), adam_t,
        warmup_steps, total_steps, min_lr_frac,
        beta1, beta2, num_restarts, current_stream());
}

void grad_norm_sq_binding(torch::Tensor grad, torch::Tensor out_norm_sq) {
    forge::optim::grad_norm_sq(
        grad.data_ptr<float>(), grad.numel(),
        out_norm_sq.data_ptr<float>(), current_stream());
}

void compute_clip_scale_binding(
    torch::Tensor out_scale, torch::Tensor norm_sq,
    float max_grad_norm, float inv_B)
{
    forge::optim::compute_clip_scale(
        out_scale.data_ptr<float>(), norm_sq.data_ptr<float>(),
        max_grad_norm, inv_B, current_stream());
}

void grad_scale_binding(torch::Tensor grad, float scale) {
    forge::optim::grad_scale(
        grad.data_ptr<float>(), grad.numel(), scale, current_stream());
}

// ================================ NN ================================

void leaky_relu_backward_binding(
    torch::Tensor z, torch::Tensor da, torch::Tensor dz, float slope)
{
    forge::nn::leaky_relu_backward(
        z.data_ptr<float>(), da.data_ptr<float>(), dz.data_ptr<float>(),
        z.numel(), slope, current_stream());
}

void tanh_backward_binding(
    torch::Tensor da, torch::Tensor a_out, torch::Tensor dz)
{
    forge::nn::tanh_backward(
        da.data_ptr<float>(), a_out.data_ptr<float>(), dz.data_ptr<float>(),
        da.numel(), current_stream());
}

void bias_grad_accumulate_binding(
    torch::Tensor d_upstream, torch::Tensor d_bias, int N, int B)
{
    forge::nn::bias_grad_accumulate(
        d_upstream.data_ptr<float>(), d_bias.data_ptr<float>(),
        N, B, current_stream());
}

void embedding_gather_binding(
    torch::Tensor context, torch::Tensor embeddings, torch::Tensor output,
    int B, int ctx_size, int E)
{
    forge::nn::embedding_gather(
        context.data_ptr<int>(), embeddings.data_ptr<float>(),
        output.data_ptr<float>(), B, ctx_size, E, current_stream());
}

void embedding_scatter_binding(
    torch::Tensor context, torch::Tensor d_output, torch::Tensor d_embeddings,
    int B, int ctx_size, int E)
{
    forge::nn::embedding_scatter(
        context.data_ptr<int>(), d_output.data_ptr<float>(),
        d_embeddings.data_ptr<float>(), B, ctx_size, E, current_stream());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
softmax_ce_forward_binding(torch::Tensor logits, torch::Tensor targets, int V, int B) {
    auto probs   = torch::empty_like(logits);
    auto losses  = torch::empty({B}, logits.options());
    auto correct = torch::empty({B}, logits.options().dtype(torch::kInt32));

    forge::nn::softmax_ce_forward(
        logits.data_ptr<float>(), probs.data_ptr<float>(),
        targets.data_ptr<int>(), losses.data_ptr<float>(),
        correct.data_ptr<int>(), V, B, 256, current_stream());

    return {probs, losses, correct};
}

void softmax_ce_backward_binding(
    torch::Tensor probs, torch::Tensor targets, torch::Tensor d_logits,
    int V, int B)
{
    forge::nn::softmax_ce_backward(
        probs.data_ptr<float>(), targets.data_ptr<int>(),
        d_logits.data_ptr<float>(), V, B, current_stream());
}

void softmax_ce_backward_bias_binding(
    torch::Tensor probs, torch::Tensor targets,
    torch::Tensor d_logits, torch::Tensor d_bias, int V, int B)
{
    forge::nn::softmax_ce_backward_bias(
        probs.data_ptr<float>(), targets.data_ptr<int>(),
        d_logits.data_ptr<float>(), d_bias.data_ptr<float>(),
        V, B, 256, current_stream());
}

void fused_gemm_softmax_ce_forward_binding(
    torch::Tensor W, torch::Tensor X, c10::optional<torch::Tensor> bias,
    torch::Tensor probs, torch::Tensor targets,
    torch::Tensor losses, torch::Tensor correct,
    int V, int D, int B)
{
    forge::nn::fused_gemm_softmax_ce_forward(
        W.data_ptr<float>(), X.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        probs.data_ptr<float>(), targets.data_ptr<int>(),
        losses.data_ptr<float>(), correct.data_ptr<int>(),
        V, D, B, current_stream());
}

// ========================== Fused MLP ================================

// Helper: convert Python string to Activation enum
static forge::nn::Activation parse_activation(const std::string& s) {
    if (s == "celu")    return forge::nn::Activation::CELU;
    if (s == "gelu")    return forge::nn::Activation::GELU;
    if (s == "relu")    return forge::nn::Activation::ReLU;
    if (s == "sigmoid") return forge::nn::Activation::Sigmoid;
    if (s == "silu")    return forge::nn::Activation::SiLU;
    if (s == "tanh")    return forge::nn::Activation::Tanh;
    throw std::runtime_error("Unknown activation: " + s + " (expected: celu, gelu, relu, sigmoid, silu, tanh)");
}

// Helper: build MlpDesc from Python arguments
static forge::nn::MlpDesc make_desc(
    int n_inputs, int n_outputs, int hidden, int n_layers,
    const std::string& activation, float act_param,
    float output_scale, bool normalize, bool use_tf32 = false)
{
    forge::nn::MlpDesc desc;
    desc.n_inputs     = n_inputs;
    desc.n_outputs    = n_outputs;
    desc.hidden       = hidden;
    desc.n_layers     = n_layers;
    desc.act          = parse_activation(activation);
    desc.act_param    = act_param;
    desc.output_scale = output_scale;
    desc.normalize    = normalize;
    desc.use_tf32     = use_tf32;
    return desc;
}

int mlp_weight_count_binding(
    int n_inputs, int n_outputs, int hidden, int n_layers,
    const std::string& activation, float act_param,
    float output_scale, bool normalize)
{
    auto desc = make_desc(n_inputs, n_outputs, hidden, n_layers,
                          activation, act_param, output_scale, normalize);
    return forge::nn::mlp_weight_count(desc);
}

int mlp_acts_save_count_binding(
    int n_inputs, int n_outputs, int hidden, int n_layers,
    const std::string& activation, float act_param,
    float output_scale, bool normalize, int B)
{
    auto desc = make_desc(n_inputs, n_outputs, hidden, n_layers,
                          activation, act_param, output_scale, normalize);
    return forge::nn::mlp_acts_save_count(desc, B);
}

void fused_mlp_forward_binding(
    int n_inputs, int n_outputs, int hidden, int n_layers,
    const std::string& activation, float act_param,
    float output_scale, bool normalize, bool use_tf32,
    torch::Tensor input, torch::Tensor weights, torch::Tensor output,
    int B, c10::optional<torch::Tensor> acts_save)
{
    auto desc = make_desc(n_inputs, n_outputs, hidden, n_layers,
                          activation, act_param, output_scale, normalize, use_tf32);
    forge::nn::fused_mlp_forward(
        desc,
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        acts_save.has_value() ? acts_save->data_ptr<float>() : nullptr,
        current_stream());
}

void fused_mlp_backward_binding(
    int n_inputs, int n_outputs, int hidden, int n_layers,
    const std::string& activation, float act_param,
    float output_scale, bool normalize, bool use_tf32,
    torch::Tensor input, torch::Tensor weights,
    torch::Tensor acts_save, torch::Tensor d_output,
    c10::optional<torch::Tensor> d_input, torch::Tensor d_weights,
    int B, float beta,
    c10::optional<torch::Tensor> workspace)
{
    auto desc = make_desc(n_inputs, n_outputs, hidden, n_layers,
                          activation, act_param, output_scale, normalize, use_tf32);
    forge::nn::fused_mlp_backward(
        desc,
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        acts_save.data_ptr<float>(),
        d_output.data_ptr<float>(),
        d_input.has_value() ? d_input->data_ptr<float>() : nullptr,
        d_weights.data_ptr<float>(),
        B, beta, current_stream(),
        workspace.has_value() ? workspace->data_ptr<float>() : nullptr);
}

int fused_mlp_backward_workspace_count_binding(
    int n_inputs, int n_outputs, int hidden, int n_layers,
    const std::string& activation, float act_param,
    float output_scale, bool normalize, bool use_tf32, int B)
{
    auto desc = make_desc(n_inputs, n_outputs, hidden, n_layers,
                          activation, act_param, output_scale, normalize, use_tf32);
    return forge::nn::fused_mlp_backward_workspace_count(desc, B);
}

void mlp_init_weights_binding(
    int n_inputs, int n_outputs, int hidden, int n_layers,
    const std::string& activation, float act_param,
    float output_scale, bool normalize,
    torch::Tensor weights, int64_t seed,
    c10::optional<torch::Tensor> norm_lb,
    c10::optional<torch::Tensor> norm_ub)
{
    auto desc = make_desc(n_inputs, n_outputs, hidden, n_layers,
                          activation, act_param, output_scale, normalize);
    // norm_lb/norm_ub are host tensors [n_inputs]
    forge::nn::mlp_init_weights(
        desc, weights.data_ptr<float>(), (uint64_t)seed,
        forge::nn::InitStyle::KaimingUniform,
        norm_lb.has_value() ? norm_lb->data_ptr<float>() : nullptr,
        norm_ub.has_value() ? norm_ub->data_ptr<float>() : nullptr);
}

// ============================== MSE ==================================

void mse_forward_binding(
    torch::Tensor pred, torch::Tensor target, torch::Tensor loss)
{
    forge::nn::mse_forward(
        pred.data_ptr<float>(), target.data_ptr<float>(),
        loss.data_ptr<float>(), pred.numel(), current_stream());
}

void mse_backward_binding(
    torch::Tensor pred, torch::Tensor target, torch::Tensor d_pred,
    float scale)
{
    forge::nn::mse_backward(
        pred.data_ptr<float>(), target.data_ptr<float>(),
        d_pred.data_ptr<float>(), pred.numel(), scale, current_stream());
}

// ============================== L1 ===================================

void l1_forward_binding(
    torch::Tensor pred, torch::Tensor target, torch::Tensor loss)
{
    forge::nn::l1_forward(
        pred.data_ptr<float>(), target.data_ptr<float>(),
        loss.data_ptr<float>(), pred.numel(), current_stream());
}

void l1_backward_binding(
    torch::Tensor pred, torch::Tensor target, torch::Tensor d_pred,
    float scale)
{
    forge::nn::l1_backward(
        pred.data_ptr<float>(), target.data_ptr<float>(),
        d_pred.data_ptr<float>(), pred.numel(), scale, current_stream());
}

// ============================== RNG ==================================

void fill_normal_binding(
    torch::Tensor data, float mean, float std,
    int64_t seed, int64_t offset)
{
    forge::nn::fill_normal(
        data.data_ptr<float>(), data.numel(),
        mean, std, (uint64_t)seed, (uint64_t)offset, current_stream());
}

void fill_uniform_binding(
    torch::Tensor data, float lo, float hi,
    int64_t seed, int64_t offset)
{
    forge::nn::fill_uniform(
        data.data_ptr<float>(), data.numel(),
        lo, hi, (uint64_t)seed, (uint64_t)offset, current_stream());
}

// ============================ Reduce =================================

void reduce_sum_binding(torch::Tensor x, torch::Tensor out) {
    forge::nn::reduce_sum(
        x.data_ptr<float>(), out.data_ptr<float>(),
        x.numel(), current_stream());
}

void reduce_mean_binding(torch::Tensor x, torch::Tensor out) {
    forge::nn::reduce_mean(
        x.data_ptr<float>(), out.data_ptr<float>(),
        x.numel(), current_stream());
}

void reduce_sumsq_binding(torch::Tensor x, torch::Tensor out) {
    forge::nn::reduce_sumsq(
        x.data_ptr<float>(), out.data_ptr<float>(),
        x.numel(), current_stream());
}

void reduce_absmax_binding(torch::Tensor x, torch::Tensor out) {
    forge::nn::reduce_absmax(
        x.data_ptr<float>(), out.data_ptr<float>(),
        x.numel(), current_stream());
}

// ======================== Activation Backward =========================

void silu_backward_binding(
    torch::Tensor z, torch::Tensor da, torch::Tensor dz)
{
    forge::nn::silu_backward(
        z.data_ptr<float>(), da.data_ptr<float>(), dz.data_ptr<float>(),
        z.numel(), current_stream());
}

void celu_backward_binding(
    torch::Tensor z, torch::Tensor da, torch::Tensor dz, float alpha)
{
    forge::nn::celu_backward(
        z.data_ptr<float>(), da.data_ptr<float>(), dz.data_ptr<float>(),
        z.numel(), alpha, current_stream());
}

// ============================= Graph =================================

// Graph wrapper: exposes forge::Graph to Python as a managed object.
struct PyGraph {
    forge::Graph graph;
    cudaStream_t stream = nullptr;

    void begin_capture() {
        stream = current_stream();
        graph.begin_capture(stream);
    }

    void end_capture() {
        graph.end_capture();
    }

    void launch() {
        graph.launch(stream ? stream : current_stream());
    }

    bool is_valid() const { return graph.is_valid(); }

    void destroy() {
        graph.destroy();
        stream = nullptr;
    }
};

// ============================= Quant ================================

std::tuple<torch::Tensor, torch::Tensor>
fused_quantize_activation_binding(torch::Tensor input, int M, int K) {
    auto output = torch::empty({M, K}, input.options().dtype(torch::kInt8));
    auto scales = torch::empty({M}, input.options().dtype(torch::kFloat32));

    forge::quant::fused_quantize_activation(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        output.data_ptr<int8_t>(), scales.data_ptr<float>(),
        M, K, current_stream());

    return {output, scales};
}

torch::Tensor fused_dequantize_bias_binding(
    torch::Tensor input, torch::Tensor act_scale, torch::Tensor w_scale,
    c10::optional<torch::Tensor> bias, int M, int N)
{
    auto output = torch::empty({M, N}, act_scale.options().dtype(torch::kFloat16));

    forge::quant::fused_dequantize_bias(
        input.data_ptr<int32_t>(), act_scale.data_ptr<float>(),
        reinterpret_cast<const __half*>(w_scale.data_ptr<at::Half>()),
        bias.has_value() ?
            reinterpret_cast<const __half*>(bias->data_ptr<at::Half>()) : nullptr,
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        M, N, current_stream());

    return output;
}

// ============================== GEMM ================================

void fp32_gemm_binding(
    const std::string& transA_str, const std::string& transB_str,
    int M, int N, int K, float alpha,
    torch::Tensor A, int lda, torch::Tensor B, int ldb,
    float beta, torch::Tensor C, int ldc,
    c10::optional<torch::Tensor> bias)
{
    auto transA = (transA_str == "T") ? forge::gemm::Op::T : forge::gemm::Op::N;
    auto transB = (transB_str == "T") ? forge::gemm::Op::T : forge::gemm::Op::N;

    forge::gemm::fp32_gemm(
        transA, transB, M, N, K, alpha,
        A.data_ptr<float>(), lda, B.data_ptr<float>(), ldb,
        beta, C.data_ptr<float>(), ldc,
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        current_stream());
}

torch::Tensor int8_linear_binding(
    torch::Tensor x, torch::Tensor w_int8,
    torch::Tensor w_scale, c10::optional<torch::Tensor> bias,
    torch::Tensor scratch_a_int8, torch::Tensor scratch_a_scale)
{
    int M = x.size(0), K = x.size(1), N = w_int8.size(0);
    auto output = torch::empty({M, N}, x.options());

    forge::gemm::int8_linear(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        w_int8.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(w_scale.data_ptr<at::Half>()),
        bias.has_value() ?
            reinterpret_cast<const __half*>(bias->data_ptr<at::Half>()) : nullptr,
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        M, K, N,
        scratch_a_int8.data_ptr<int8_t>(),
        scratch_a_scale.data_ptr<float>(),
        current_stream());

    return output;
}

torch::Tensor sparse_int8_linear_binding(
    torch::Tensor x, torch::Tensor W_comp, torch::Tensor w_meta,
    torch::Tensor w_scale, c10::optional<torch::Tensor> bias,
    torch::Tensor scratch_x_int8, torch::Tensor scratch_x_scale,
    c10::optional<torch::Tensor> splitk_workspace, int split_k)
{
    int M = x.size(0), K = x.size(1), N_out = W_comp.size(0);
    auto output = torch::empty({M, N_out}, x.options());

    forge::gemm::sparse_int8_linear(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        W_comp.data_ptr<int8_t>(),
        reinterpret_cast<const uint32_t*>(w_meta.data_ptr<int32_t>()),
        reinterpret_cast<const __half*>(w_scale.data_ptr<at::Half>()),
        bias.has_value() ?
            reinterpret_cast<const __half*>(bias->data_ptr<at::Half>()) : nullptr,
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        M, K, N_out,
        scratch_x_int8.data_ptr<int8_t>(),
        scratch_x_scale.data_ptr<float>(),
        splitk_workspace.has_value() ? splitk_workspace->data_ptr<float>() : nullptr,
        split_k, current_stream());

    return output;
}

// ============================ Sparsity ==============================

void prune_compress_24_binding(
    torch::Tensor weight, torch::Tensor compressed,
    torch::Tensor metadata_raw, int N, int K)
{
    forge::sparsity::prune_compress_24(
        weight.data_ptr<int8_t>(), compressed.data_ptr<int8_t>(),
        reinterpret_cast<uint8_t*>(metadata_raw.data_ptr<uint8_t>()),
        N, K, current_stream());
}

void pack_metadata_binding(
    torch::Tensor meta_raw, torch::Tensor meta_packed, int N, int K)
{
    forge::sparsity::pack_metadata(
        reinterpret_cast<const uint8_t*>(meta_raw.data_ptr<uint8_t>()),
        reinterpret_cast<uint32_t*>(meta_packed.data_ptr<int32_t>()),
        N, K, current_stream());
}

// ======================== Thermal Simulation ========================

void thermal_generate_batch_binding(
    torch::Tensor T_input, torch::Tensor conductivity,
    torch::Tensor msh, torch::Tensor T_target,
    torch::Tensor energy_buf, torch::Tensor scratch_buf,
    torch::Tensor precomp_buf, torch::Tensor type_map_buf,
    int B, int G, int n_steps,
    float dt, float max_alpha, float min_alpha,
    float max_temp_delta, bool hard_mode,
    float temp_scale, std::vector<float> strategy_weights,
    int64_t seed)
{
    forge::sim::generate_training_batch(
        T_input.data_ptr<float>(), conductivity.data_ptr<float>(),
        msh.data_ptr<float>(), T_target.data_ptr<float>(),
        energy_buf.data_ptr<float>(), scratch_buf.data_ptr<float>(),
        precomp_buf.data_ptr<float>(), type_map_buf.data_ptr<int>(),
        B, G, n_steps, dt, max_alpha, min_alpha,
        max_temp_delta, hard_mode, temp_scale, strategy_weights,
        static_cast<uint64_t>(seed), current_stream());
}

void thermal_simulate_binding(
    torch::Tensor energy, torch::Tensor scratch, torch::Tensor precomp,
    torch::Tensor conductivity, torch::Tensor msh,
    int B, int G, int n_steps,
    float dt, float max_temp_delta, float conduction_scale)
{
    forge::sim::thermal_simulate(
        energy.data_ptr<float>(), scratch.data_ptr<float>(),
        precomp.data_ptr<float>(), conductivity.data_ptr<float>(),
        msh.data_ptr<float>(), B, G, n_steps,
        dt, max_temp_delta, conduction_scale, current_stream());
}

void thermal_temp_to_energy_binding(
    torch::Tensor temperature, torch::Tensor msh, torch::Tensor energy)
{
    forge::sim::temperature_to_energy(
        temperature.data_ptr<float>(), msh.data_ptr<float>(),
        energy.data_ptr<float>(), temperature.numel(), current_stream());
}

void thermal_energy_to_temp_binding(
    torch::Tensor energy, torch::Tensor msh, torch::Tensor temperature)
{
    forge::sim::energy_to_temperature(
        energy.data_ptr<float>(), msh.data_ptr<float>(),
        temperature.data_ptr<float>(), energy.numel(), current_stream());
}

// ======================== Stefan Simulation ========================

void stefan_generate_batch_binding(
    torch::Tensor T_input, torch::Tensor f_input,
    torch::Tensor T_target, torch::Tensor f_target,
    torch::Tensor conductivity, torch::Tensor msh,
    torch::Tensor latent_heat, torch::Tensor T_solidus,
    torch::Tensor T_liquidus,
    torch::Tensor enthalpy_buf, torch::Tensor scratch_buf,
    torch::Tensor precomp_buf,
    int B, int G, int horizon_steps,
    float dt, float max_temp_delta, int64_t seed)
{
    forge::sim::generate_stefan_batch(
        T_input.data_ptr<float>(), f_input.data_ptr<float>(),
        T_target.data_ptr<float>(), f_target.data_ptr<float>(),
        conductivity.data_ptr<float>(), msh.data_ptr<float>(),
        latent_heat.data_ptr<float>(), T_solidus.data_ptr<float>(),
        T_liquidus.data_ptr<float>(),
        enthalpy_buf.data_ptr<float>(), scratch_buf.data_ptr<float>(),
        precomp_buf.data_ptr<float>(),
        B, G, horizon_steps, dt, max_temp_delta,
        static_cast<uint64_t>(seed), current_stream());
}

void stefan_simulate_binding(
    torch::Tensor enthalpy, torch::Tensor scratch, torch::Tensor precomp,
    torch::Tensor conductivity, torch::Tensor msh,
    torch::Tensor latent_heat, torch::Tensor T_solidus,
    torch::Tensor T_liquidus,
    int B, int G, int n_steps,
    float dt, float max_temp_delta, float conduction_scale)
{
    forge::sim::stefan_simulate(
        enthalpy.data_ptr<float>(), scratch.data_ptr<float>(),
        precomp.data_ptr<float>(), conductivity.data_ptr<float>(),
        msh.data_ptr<float>(), latent_heat.data_ptr<float>(),
        T_solidus.data_ptr<float>(), T_liquidus.data_ptr<float>(),
        B, G, n_steps, dt, max_temp_delta, conduction_scale,
        current_stream());
}

void stefan_extract_liquid_fraction_binding(
    torch::Tensor enthalpy, torch::Tensor precomp,
    torch::Tensor liquid_fraction, int B, int G)
{
    forge::sim::extract_liquid_fraction(
        enthalpy.data_ptr<float>(), precomp.data_ptr<float>(),
        liquid_fraction.data_ptr<float>(), B, G, current_stream());
}

void stefan_enthalpy_to_temp_binding(
    torch::Tensor enthalpy, torch::Tensor precomp,
    torch::Tensor temperature, int B, int G)
{
    forge::sim::enthalpy_to_temperature(
        enthalpy.data_ptr<float>(), precomp.data_ptr<float>(),
        temperature.data_ptr<float>(), B, G, current_stream());
}

void stefan_convection_generate_batch_binding(
    torch::Tensor T_input, torch::Tensor f_input,
    torch::Tensor T_target, torch::Tensor f_target,
    torch::Tensor conductivity, torch::Tensor msh,
    torch::Tensor latent_heat, torch::Tensor T_solidus,
    torch::Tensor T_liquidus,
    torch::Tensor vx, torch::Tensor vy, torch::Tensor vz,
    torch::Tensor enthalpy_buf, torch::Tensor scratch_buf,
    torch::Tensor precomp_buf,
    int B, int G, int horizon_steps,
    float dt, float max_temp_delta, int64_t seed)
{
    forge::sim::generate_stefan_convection_batch(
        T_input.data_ptr<float>(), f_input.data_ptr<float>(),
        T_target.data_ptr<float>(), f_target.data_ptr<float>(),
        conductivity.data_ptr<float>(), msh.data_ptr<float>(),
        latent_heat.data_ptr<float>(), T_solidus.data_ptr<float>(),
        T_liquidus.data_ptr<float>(),
        vx.data_ptr<float>(), vy.data_ptr<float>(), vz.data_ptr<float>(),
        enthalpy_buf.data_ptr<float>(), scratch_buf.data_ptr<float>(),
        precomp_buf.data_ptr<float>(),
        B, G, horizon_steps, dt, max_temp_delta,
        static_cast<uint64_t>(seed), current_stream());
}

// ======================== Conv2D ========================

void conv2d_forward_binding(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW, int groups)
{
    forge::nn::conv2d_forward(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H, W,
        C_out, kH, kW,
        padH, padW,
        strideH, strideW,
        dilH, dilW, groups,
        current_stream());
}

void conv2d_backward_data_binding(
    torch::Tensor d_output, torch::Tensor weight,
    torch::Tensor d_input,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW, int groups)
{
    forge::nn::conv2d_backward_data(
        d_output.data_ptr<float>(), weight.data_ptr<float>(),
        d_input.data_ptr<float>(),
        N, C_in, H, W,
        C_out, kH, kW,
        padH, padW,
        strideH, strideW,
        dilH, dilW, groups,
        current_stream());
}

void conv2d_backward_weight_binding(
    torch::Tensor d_output, torch::Tensor input,
    torch::Tensor d_weight, torch::Tensor d_bias,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW, int groups,
    float beta)
{
    forge::nn::conv2d_backward_weight(
        d_output.data_ptr<float>(), input.data_ptr<float>(),
        d_weight.data_ptr<float>(),
        d_bias.defined() ? d_bias.data_ptr<float>() : nullptr,
        N, C_in, H, W,
        C_out, kH, kW,
        padH, padW,
        strideH, strideW,
        dilH, dilW, groups,
        beta, current_stream());
}

// ======================== Conv3D ========================

void conv3d_forward_binding(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, torch::Tensor workspace,
    int N, int C_in, int D, int H, int W,
    int C_out, int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD, int dilH, int dilW, int groups)
{
    forge::nn::conv3d_forward(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        workspace.defined() ? workspace.data_ptr<float>() : nullptr,
        N, C_in, D, H, W,
        C_out, kD, kH, kW,
        padD, padH, padW,
        strideD, strideH, strideW,
        dilD, dilH, dilW, groups,
        current_stream());
}

void conv3d_backward_data_binding(
    torch::Tensor d_output, torch::Tensor weight,
    torch::Tensor d_input, torch::Tensor workspace,
    int N, int C_in, int D, int H, int W,
    int C_out, int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD, int dilH, int dilW, int groups)
{
    forge::nn::conv3d_backward_data(
        d_output.data_ptr<float>(), weight.data_ptr<float>(),
        d_input.data_ptr<float>(),
        workspace.defined() ? workspace.data_ptr<float>() : nullptr,
        N, C_in, D, H, W,
        C_out, kD, kH, kW,
        padD, padH, padW,
        strideD, strideH, strideW,
        dilD, dilH, dilW, groups,
        current_stream());
}

void conv3d_backward_weight_binding(
    torch::Tensor d_output, torch::Tensor input,
    torch::Tensor d_weight, torch::Tensor d_bias,
    torch::Tensor workspace,
    int N, int C_in, int D, int H, int W,
    int C_out, int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD, int dilH, int dilW, int groups,
    float beta)
{
    forge::nn::conv3d_backward_weight(
        d_output.data_ptr<float>(), input.data_ptr<float>(),
        d_weight.data_ptr<float>(),
        d_bias.defined() ? d_bias.data_ptr<float>() : nullptr,
        workspace.defined() ? workspace.data_ptr<float>() : nullptr,
        N, C_in, D, H, W,
        C_out, kD, kH, kW,
        padD, padH, padW,
        strideD, strideH, strideW,
        dilD, dilH, dilW, groups,
        beta, current_stream());
}

// Fused GroupNorm + GELU
void group_norm_gelu_forward_binding(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
    torch::Tensor y, torch::Tensor mean, torch::Tensor rstd,
    int N, int C, int spatial, int G, float eps)
{
    forge::nn::group_norm_gelu_forward(
        x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        y.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        N, C, spatial, G, eps, current_stream());
}

void group_norm_gelu_backward_binding(
    torch::Tensor d_output, torch::Tensor x,
    torch::Tensor gamma, torch::Tensor beta,
    torch::Tensor mean, torch::Tensor rstd,
    torch::Tensor d_input, torch::Tensor d_gamma, torch::Tensor d_beta,
    int N, int C, int spatial, int G)
{
    forge::nn::group_norm_gelu_backward(
        d_output.data_ptr<float>(), x.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(),
        d_input.data_ptr<float>(), d_gamma.data_ptr<float>(), d_beta.data_ptr<float>(),
        N, C, spatial, G, current_stream());
}

// Standalone GroupNorm (no GELU fusion)
void group_norm_forward_binding(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
    torch::Tensor y, torch::Tensor mean, torch::Tensor rstd,
    int N, int C, int spatial, int G, float eps)
{
    forge::nn::group_norm_forward(
        x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        y.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        N, C, spatial, G, eps, current_stream());
}

void group_norm_backward_binding(
    torch::Tensor d_output, torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor mean, torch::Tensor rstd,
    torch::Tensor d_input, torch::Tensor d_gamma, torch::Tensor d_beta,
    int N, int C, int spatial, int G)
{
    forge::nn::group_norm_backward(
        d_output.data_ptr<float>(), x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(),
        d_input.data_ptr<float>(), d_gamma.data_ptr<float>(), d_beta.data_ptr<float>(),
        N, C, spatial, G, current_stream());
}

// Implicit GEMM Conv3D (no workspace needed)
void conv3d_implicit_forward_binding(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int D, int H, int W,
    int C_out, int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD, int dilH, int dilW, int groups)
{
    forge::nn::conv3d_implicit_forward(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, D, H, W,
        C_out, kD, kH, kW,
        padD, padH, padW,
        strideD, strideH, strideW,
        dilD, dilH, dilW, groups,
        current_stream());
}

void conv3d_implicit_backward_data_binding(
    torch::Tensor d_output, torch::Tensor weight,
    torch::Tensor d_input,
    int N, int C_in, int D, int H, int W,
    int C_out, int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD, int dilH, int dilW, int groups)
{
    forge::nn::conv3d_implicit_backward_data(
        d_output.data_ptr<float>(), weight.data_ptr<float>(),
        d_input.data_ptr<float>(),
        N, C_in, D, H, W,
        C_out, kD, kH, kW,
        padD, padH, padW,
        strideD, strideH, strideW,
        dilD, dilH, dilW, groups,
        current_stream());
}

// ====================== Winograd Conv3D ======================

void conv3d_winograd_filter_transform_binding(
    torch::Tensor weight, torch::Tensor U, torch::Tensor U_t,
    int C_out, int C_in)
{
    forge::nn::conv3d_winograd_filter_transform(
        weight.data_ptr<float>(), U.data_ptr<float>(),
        U_t.defined() ? U_t.data_ptr<float>() : nullptr,
        C_out, C_in, current_stream());
}

void conv3d_winograd_forward_binding(
    torch::Tensor input, torch::Tensor U, torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int D, int H, int W, int C_out)
{
    forge::nn::conv3d_winograd_forward(
        input.data_ptr<float>(), U.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, D, H, W, C_out,
        current_stream());
}

void conv3d_winograd_backward_data_binding(
    torch::Tensor d_output, torch::Tensor U, torch::Tensor d_input,
    int N, int C_in, int D, int H, int W, int C_out)
{
    forge::nn::conv3d_winograd_backward_data(
        d_output.data_ptr<float>(), U.data_ptr<float>(),
        d_input.data_ptr<float>(),
        N, C_in, D, H, W, C_out,
        current_stream());
}

// ============================ Module ================================

PYBIND11_MODULE(_forge_C, m) {
    m.doc() = "Forge GPU Compute Library - Python bindings";

    // Optimizer
    m.def("adam_step", &adam_step_binding, "Adam optimizer step",
          py::arg("param"), py::arg("grad"), py::arg("m"), py::arg("v"),
          py::arg("base_lr"), py::arg("step_scalars"),
          py::arg("beta1")=0.9f, py::arg("beta2")=0.999f,
          py::arg("eps")=1e-8f, py::arg("weight_decay")=0.0f,
          py::arg("grad_scale")=py::none(),
          py::arg("do_clamp")=false, py::arg("clamp_lo")=0.0f, py::arg("clamp_hi")=0.0f,
          py::arg("zero_grad")=false);

    m.def("update_step_scalars", &update_step_scalars_binding,
          "Update Adam step scalars on device",
          py::arg("step_scalars"), py::arg("adam_t"),
          py::arg("warmup_steps"), py::arg("total_steps"), py::arg("min_lr_frac"),
          py::arg("beta1")=0.9f, py::arg("beta2")=0.999f,
          py::arg("num_restarts")=1);

    m.def("grad_norm_sq", &grad_norm_sq_binding, "Compute squared gradient norm");
    m.def("compute_clip_scale", &compute_clip_scale_binding, "Compute gradient clip scale");
    m.def("grad_scale", &grad_scale_binding, "Scale gradients by scalar");

    // NN
    m.def("leaky_relu_backward", &leaky_relu_backward_binding,
          "Leaky ReLU backward", py::arg("z"), py::arg("da"), py::arg("dz"),
          py::arg("slope")=0.01f);
    m.def("tanh_backward", &tanh_backward_binding, "Tanh backward");
    m.def("bias_grad_accumulate", &bias_grad_accumulate_binding,
          "Bias gradient accumulation");
    m.def("embedding_gather", &embedding_gather_binding, "Embedding forward (gather)");
    m.def("embedding_scatter", &embedding_scatter_binding, "Embedding backward (scatter)");
    m.def("softmax_ce_forward", &softmax_ce_forward_binding,
          "Softmax + cross-entropy forward");
    m.def("softmax_ce_backward", &softmax_ce_backward_binding,
          "Softmax + cross-entropy backward");
    m.def("softmax_ce_backward_bias", &softmax_ce_backward_bias_binding,
          "Fused softmax CE backward + bias grad");
    m.def("fused_gemm_softmax_ce_forward", &fused_gemm_softmax_ce_forward_binding,
          "Fused GEMM + softmax + CE forward (no materialized logits)");

    // Fused MLP
    m.def("mlp_weight_count", &mlp_weight_count_binding,
          "Total floats in packed MLP weight buffer",
          py::arg("n_inputs"), py::arg("n_outputs"), py::arg("hidden"),
          py::arg("n_layers"), py::arg("activation")="celu",
          py::arg("act_param")=1.0f, py::arg("output_scale")=1.0f,
          py::arg("normalize")=false);
    m.def("mlp_acts_save_count", &mlp_acts_save_count_binding,
          "Floats needed for acts_save buffer (for backward)",
          py::arg("n_inputs"), py::arg("n_outputs"), py::arg("hidden"),
          py::arg("n_layers"), py::arg("activation")="celu",
          py::arg("act_param")=1.0f, py::arg("output_scale")=1.0f,
          py::arg("normalize")=false, py::arg("B")=1);
    m.def("fused_mlp_forward", &fused_mlp_forward_binding,
          "Fused MLP forward pass (single kernel)",
          py::arg("n_inputs"), py::arg("n_outputs"), py::arg("hidden"),
          py::arg("n_layers"), py::arg("activation"), py::arg("act_param"),
          py::arg("output_scale"), py::arg("normalize"), py::arg("use_tf32"),
          py::arg("input"), py::arg("weights"), py::arg("output"),
          py::arg("B"), py::arg("acts_save")=py::none());
    m.def("fused_mlp_backward", &fused_mlp_backward_binding,
          "Fused MLP backward pass (single kernel)",
          py::arg("n_inputs"), py::arg("n_outputs"), py::arg("hidden"),
          py::arg("n_layers"), py::arg("activation"), py::arg("act_param"),
          py::arg("output_scale"), py::arg("normalize"), py::arg("use_tf32"),
          py::arg("input"), py::arg("weights"), py::arg("acts_save"),
          py::arg("d_output"), py::arg("d_input"), py::arg("d_weights"),
          py::arg("B"), py::arg("beta")=0.0f,
          py::arg("workspace")=py::none());
    m.def("fused_mlp_backward_workspace_count",
          &fused_mlp_backward_workspace_count_binding,
          "Floats needed for backward workspace (0 if not useful)",
          py::arg("n_inputs"), py::arg("n_outputs"), py::arg("hidden"),
          py::arg("n_layers"), py::arg("activation")="celu",
          py::arg("act_param")=1.0f, py::arg("output_scale")=1.0f,
          py::arg("normalize")=false, py::arg("use_tf32")=false,
          py::arg("B")=1);
    m.def("mlp_init_weights", &mlp_init_weights_binding,
          "Initialize MLP weights (Kaiming uniform)",
          py::arg("n_inputs"), py::arg("n_outputs"), py::arg("hidden"),
          py::arg("n_layers"), py::arg("activation"), py::arg("act_param"),
          py::arg("output_scale"), py::arg("normalize"),
          py::arg("weights"), py::arg("seed"),
          py::arg("norm_lb")=py::none(), py::arg("norm_ub")=py::none());

    // MSE loss
    m.def("mse_forward", &mse_forward_binding,
          "MSE forward: loss[0] = mean((pred - target)^2)");
    m.def("mse_backward", &mse_backward_binding,
          "MSE backward: d_pred = scale * 2/n * (pred - target)",
          py::arg("pred"), py::arg("target"), py::arg("d_pred"),
          py::arg("scale")=1.0f);

    // L1
    m.def("l1_forward", &l1_forward_binding,
          "L1 forward: loss[0] = mean(|pred - target|)");
    m.def("l1_backward", &l1_backward_binding,
          "L1 backward: d_pred = scale * sign(pred - target) / n",
          py::arg("pred"), py::arg("target"), py::arg("d_pred"),
          py::arg("scale")=1.0f);

    // RNG
    m.def("fill_normal", &fill_normal_binding,
          "Fill tensor with N(mean, std) samples",
          py::arg("data"), py::arg("mean")=0.0f, py::arg("std")=1.0f,
          py::arg("seed")=42, py::arg("offset")=0);
    m.def("fill_uniform", &fill_uniform_binding,
          "Fill tensor with U(lo, hi) samples",
          py::arg("data"), py::arg("lo")=0.0f, py::arg("hi")=1.0f,
          py::arg("seed")=42, py::arg("offset")=0);

    // Reduce
    m.def("reduce_sum", &reduce_sum_binding, "Sum to scalar");
    m.def("reduce_mean", &reduce_mean_binding, "Mean to scalar");
    m.def("reduce_sumsq", &reduce_sumsq_binding, "Sum of squares to scalar");
    m.def("reduce_absmax", &reduce_absmax_binding, "Absolute max to scalar");

    // Activation backward (standalone)
    m.def("silu_backward", &silu_backward_binding, "SiLU backward");
    m.def("celu_backward", &celu_backward_binding, "CELU backward",
          py::arg("z"), py::arg("da"), py::arg("dz"), py::arg("alpha")=1.0f);

    // CUDA Graph
    py::class_<PyGraph>(m, "Graph")
        .def(py::init<>())
        .def("begin_capture", &PyGraph::begin_capture,
             "Begin CUDA stream capture for graph recording")
        .def("end_capture", &PyGraph::end_capture,
             "End capture and instantiate the executable graph")
        .def("launch", &PyGraph::launch,
             "Replay the captured kernel sequence")
        .def("is_valid", &PyGraph::is_valid,
             "Returns True if a graph has been captured")
        .def("destroy", &PyGraph::destroy,
             "Explicitly destroy the graph");

    // Quant
    m.def("fused_quantize_activation", &fused_quantize_activation_binding,
          "FP16 -> INT8 per-token quantization");
    m.def("fused_dequantize_bias", &fused_dequantize_bias_binding,
          "INT32 -> FP16 dequantization with optional bias");

    // GEMM
    m.def("fp32_gemm", &fp32_gemm_binding, "FP32 tiled GEMM",
          py::arg("transA"), py::arg("transB"),
          py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha"),
          py::arg("A"), py::arg("lda"), py::arg("B"), py::arg("ldb"),
          py::arg("beta"), py::arg("C"), py::arg("ldc"),
          py::arg("bias")=py::none());
    m.def("int8_linear", &int8_linear_binding,
          "Fused quantize + INT8 GEMM + dequant linear layer");
    m.def("sparse_int8_linear", &sparse_int8_linear_binding,
          "Sparse INT8 linear with 2:4 structured sparsity");

    // Sparsity
    m.def("prune_compress_24", &prune_compress_24_binding,
          "2:4 structured pruning + compression");
    m.def("pack_metadata", &pack_metadata_binding,
          "Pack sparsity metadata to hardware format");

    // Thermal Simulation
    m.def("thermal_generate_batch", &thermal_generate_batch_binding,
          "Generate complete training batch on GPU (materials + temperature + simulation)",
          py::arg("T_input"), py::arg("conductivity"),
          py::arg("msh"), py::arg("T_target"),
          py::arg("energy_buf"), py::arg("scratch_buf"),
          py::arg("precomp_buf"), py::arg("type_map_buf"),
          py::arg("B"), py::arg("G"), py::arg("n_steps"),
          py::arg("dt")=0.05f, py::arg("max_alpha")=0.001f,
          py::arg("min_alpha")=0.0f, py::arg("max_temp_delta")=50.0f,
          py::arg("hard_mode")=false,
          py::arg("temp_scale")=1.0f,
          py::arg("strategy_weights")=std::vector<float>(),
          py::arg("seed")=42);

    m.def("thermal_simulate", &thermal_simulate_binding,
          "Run n_steps of thermal conduction FD simulation (energy in-place)",
          py::arg("energy"), py::arg("scratch"), py::arg("precomp"),
          py::arg("conductivity"), py::arg("msh"),
          py::arg("B"), py::arg("G"), py::arg("n_steps"),
          py::arg("dt")=0.05f, py::arg("max_temp_delta")=50.0f,
          py::arg("conduction_scale")=1.0f);

    m.def("thermal_temp_to_energy", &thermal_temp_to_energy_binding,
          "Convert temperature to energy: E = T * msh",
          py::arg("temperature"), py::arg("msh"), py::arg("energy"));

    m.def("thermal_energy_to_temp", &thermal_energy_to_temp_binding,
          "Convert energy to temperature: T = E / msh",
          py::arg("energy"), py::arg("msh"), py::arg("temperature"));

    // Stefan (Solidification/Melting) Simulation
    m.def("stefan_generate_batch", &stefan_generate_batch_binding,
          "Generate complete Stefan problem training batch on GPU",
          py::arg("T_input"), py::arg("f_input"),
          py::arg("T_target"), py::arg("f_target"),
          py::arg("conductivity"), py::arg("msh"),
          py::arg("latent_heat"), py::arg("T_solidus"),
          py::arg("T_liquidus"),
          py::arg("enthalpy_buf"), py::arg("scratch_buf"),
          py::arg("precomp_buf"),
          py::arg("B"), py::arg("G"),
          py::arg("horizon_steps")=200,
          py::arg("dt")=0.05f, py::arg("max_temp_delta")=50.0f,
          py::arg("seed")=42);

    m.def("stefan_simulate", &stefan_simulate_binding,
          "Run n_steps of Stefan problem simulation (enthalpy in-place)",
          py::arg("enthalpy"), py::arg("scratch"), py::arg("precomp"),
          py::arg("conductivity"), py::arg("msh"),
          py::arg("latent_heat"), py::arg("T_solidus"),
          py::arg("T_liquidus"),
          py::arg("B"), py::arg("G"), py::arg("n_steps"),
          py::arg("dt")=0.05f, py::arg("max_temp_delta")=50.0f,
          py::arg("conduction_scale")=1.0f);

    m.def("stefan_extract_liquid_fraction", &stefan_extract_liquid_fraction_binding,
          "Extract liquid fraction from enthalpy field",
          py::arg("enthalpy"), py::arg("precomp"),
          py::arg("liquid_fraction"), py::arg("B"), py::arg("G"));

    m.def("stefan_enthalpy_to_temp", &stefan_enthalpy_to_temp_binding,
          "Convert enthalpy to temperature using phase change properties",
          py::arg("enthalpy"), py::arg("precomp"),
          py::arg("temperature"), py::arg("B"), py::arg("G"));

    m.def("stefan_convection_generate_batch", &stefan_convection_generate_batch_binding,
          "Generate Stefan+convection training batch with velocity field",
          py::arg("T_input"), py::arg("f_input"),
          py::arg("T_target"), py::arg("f_target"),
          py::arg("conductivity"), py::arg("msh"),
          py::arg("latent_heat"), py::arg("T_solidus"),
          py::arg("T_liquidus"),
          py::arg("vx"), py::arg("vy"), py::arg("vz"),
          py::arg("enthalpy_buf"), py::arg("scratch_buf"),
          py::arg("precomp_buf"),
          py::arg("B"), py::arg("G"),
          py::arg("horizon_steps")=200,
          py::arg("dt")=0.05f, py::arg("max_temp_delta")=50.0f,
          py::arg("seed")=42);

    // Conv2D (im2col + GEMM)
    m.def("conv2d_forward", &conv2d_forward_binding,
          "2D convolution forward: output = conv2d(input, weight) + bias",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("output"),
          py::arg("N"), py::arg("C_in"), py::arg("H"), py::arg("W"),
          py::arg("C_out"), py::arg("kH"), py::arg("kW"),
          py::arg("padH"), py::arg("padW"),
          py::arg("strideH"), py::arg("strideW"),
          py::arg("dilH")=1, py::arg("dilW")=1,
          py::arg("groups")=1);

    m.def("conv2d_backward_data", &conv2d_backward_data_binding,
          "2D convolution backward data: d_input from d_output and weight",
          py::arg("d_output"), py::arg("weight"),
          py::arg("d_input"),
          py::arg("N"), py::arg("C_in"), py::arg("H"), py::arg("W"),
          py::arg("C_out"), py::arg("kH"), py::arg("kW"),
          py::arg("padH"), py::arg("padW"),
          py::arg("strideH"), py::arg("strideW"),
          py::arg("dilH")=1, py::arg("dilW")=1,
          py::arg("groups")=1);

    m.def("conv2d_backward_weight", &conv2d_backward_weight_binding,
          "2D convolution backward weight: d_weight (and optionally d_bias) from d_output and input",
          py::arg("d_output"), py::arg("input"),
          py::arg("d_weight"), py::arg("d_bias"),
          py::arg("N"), py::arg("C_in"), py::arg("H"), py::arg("W"),
          py::arg("C_out"), py::arg("kH"), py::arg("kW"),
          py::arg("padH"), py::arg("padW"),
          py::arg("strideH"), py::arg("strideW"),
          py::arg("dilH")=1, py::arg("dilW")=1,
          py::arg("groups")=1, py::arg("beta")=0.0f);

    // Conv3D (vol2col + GEMM)
    m.def("conv3d_forward", &conv3d_forward_binding,
          "3D convolution forward: output = conv3d(input, weight) + bias",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("output"), py::arg("workspace"),
          py::arg("N"), py::arg("C_in"), py::arg("D"), py::arg("H"), py::arg("W"),
          py::arg("C_out"), py::arg("kD"), py::arg("kH"), py::arg("kW"),
          py::arg("padD"), py::arg("padH"), py::arg("padW"),
          py::arg("strideD"), py::arg("strideH"), py::arg("strideW"),
          py::arg("dilD")=1, py::arg("dilH")=1, py::arg("dilW")=1,
          py::arg("groups")=1);

    m.def("conv3d_backward_data", &conv3d_backward_data_binding,
          "3D convolution backward data: d_input from d_output and weight",
          py::arg("d_output"), py::arg("weight"),
          py::arg("d_input"), py::arg("workspace"),
          py::arg("N"), py::arg("C_in"), py::arg("D"), py::arg("H"), py::arg("W"),
          py::arg("C_out"), py::arg("kD"), py::arg("kH"), py::arg("kW"),
          py::arg("padD"), py::arg("padH"), py::arg("padW"),
          py::arg("strideD"), py::arg("strideH"), py::arg("strideW"),
          py::arg("dilD")=1, py::arg("dilH")=1, py::arg("dilW")=1,
          py::arg("groups")=1);

    m.def("conv3d_backward_weight", &conv3d_backward_weight_binding,
          "3D convolution backward weight: d_weight (and optionally d_bias) from d_output and input",
          py::arg("d_output"), py::arg("input"),
          py::arg("d_weight"), py::arg("d_bias"), py::arg("workspace"),
          py::arg("N"), py::arg("C_in"), py::arg("D"), py::arg("H"), py::arg("W"),
          py::arg("C_out"), py::arg("kD"), py::arg("kH"), py::arg("kW"),
          py::arg("padD"), py::arg("padH"), py::arg("padW"),
          py::arg("strideD"), py::arg("strideH"), py::arg("strideW"),
          py::arg("dilD")=1, py::arg("dilH")=1, py::arg("dilW")=1,
          py::arg("groups")=1, py::arg("beta")=0.0f);

    // Fused GroupNorm + GELU
    m.def("group_norm_gelu_forward", &group_norm_gelu_forward_binding,
          "Fused GroupNorm + GELU forward: y = GELU(gamma * normalize(x) + beta)",
          py::arg("x"), py::arg("gamma"), py::arg("beta"),
          py::arg("y"), py::arg("mean"), py::arg("rstd"),
          py::arg("N"), py::arg("C"), py::arg("spatial"), py::arg("G"),
          py::arg("eps")=1e-5f);

    m.def("group_norm_gelu_backward", &group_norm_gelu_backward_binding,
          "Fused GroupNorm + GELU backward: d_input, d_gamma, d_beta",
          py::arg("d_output"), py::arg("x"),
          py::arg("gamma"), py::arg("beta"),
          py::arg("mean"), py::arg("rstd"),
          py::arg("d_input"), py::arg("d_gamma"), py::arg("d_beta"),
          py::arg("N"), py::arg("C"), py::arg("spatial"), py::arg("G"));

    // Standalone GroupNorm (no GELU)
    m.def("group_norm_forward", &group_norm_forward_binding,
          "GroupNorm forward: y = gamma * normalize(x) + beta",
          py::arg("x"), py::arg("gamma"), py::arg("beta"),
          py::arg("y"), py::arg("mean"), py::arg("rstd"),
          py::arg("N"), py::arg("C"), py::arg("spatial"), py::arg("G"),
          py::arg("eps")=1e-5f);

    m.def("group_norm_backward", &group_norm_backward_binding,
          "GroupNorm backward: d_input, d_gamma, d_beta",
          py::arg("d_output"), py::arg("x"),
          py::arg("gamma"),
          py::arg("mean"), py::arg("rstd"),
          py::arg("d_input"), py::arg("d_gamma"), py::arg("d_beta"),
          py::arg("N"), py::arg("C"), py::arg("spatial"), py::arg("G"));

    // Implicit GEMM Conv3D (no workspace)
    m.def("conv3d_implicit_forward", &conv3d_implicit_forward_binding,
          "Implicit GEMM Conv3D forward (no workspace needed)",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("output"),
          py::arg("N"), py::arg("C_in"), py::arg("D"), py::arg("H"), py::arg("W"),
          py::arg("C_out"), py::arg("kD"), py::arg("kH"), py::arg("kW"),
          py::arg("padD"), py::arg("padH"), py::arg("padW"),
          py::arg("strideD"), py::arg("strideH"), py::arg("strideW"),
          py::arg("dilD")=1, py::arg("dilH")=1, py::arg("dilW")=1,
          py::arg("groups")=1);

    m.def("conv3d_implicit_backward_data", &conv3d_implicit_backward_data_binding,
          "Implicit GEMM Conv3D backward data (d_input from d_output + weight)",
          py::arg("d_output"), py::arg("weight"),
          py::arg("d_input"),
          py::arg("N"), py::arg("C_in"), py::arg("D"), py::arg("H"), py::arg("W"),
          py::arg("C_out"), py::arg("kD"), py::arg("kH"), py::arg("kW"),
          py::arg("padD"), py::arg("padH"), py::arg("padW"),
          py::arg("strideD"), py::arg("strideH"), py::arg("strideW"),
          py::arg("dilD")=1, py::arg("dilH")=1, py::arg("dilW")=1,
          py::arg("groups")=1);

    // Winograd Conv3D (k=3, stride=1, pad=1)
    m.def("conv3d_winograd_filter_transform", &conv3d_winograd_filter_transform_binding,
          "Pre-transform weight to Winograd domain: U[64][C_in][C_out] + U_t[64][C_out][C_in]",
          py::arg("weight"), py::arg("U"), py::arg("U_t"),
          py::arg("C_out"), py::arg("C_in"));

    m.def("conv3d_winograd_forward", &conv3d_winograd_forward_binding,
          "Fused Winograd F(2,3)^3 Conv3D forward (k=3, s=1, p=1)",
          py::arg("input"), py::arg("U"), py::arg("bias"),
          py::arg("output"),
          py::arg("N"), py::arg("C_in"), py::arg("D"), py::arg("H"), py::arg("W"),
          py::arg("C_out"));

    m.def("conv3d_winograd_backward_data", &conv3d_winograd_backward_data_binding,
          "Fused Winograd F(2,3)^3 Conv3D backward data",
          py::arg("d_output"), py::arg("U"), py::arg("d_input"),
          py::arg("N"), py::arg("C_in"), py::arg("D"), py::arg("H"), py::arg("W"),
          py::arg("C_out"));
}
