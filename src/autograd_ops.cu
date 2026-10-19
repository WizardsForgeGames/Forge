// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Backward node classes for all differentiable operations.
//
// Each class captures tensors/scalars from forward and implements apply()
// to compute input gradients. All computations use existing forward ops
// (no new CUDA kernels needed).
#include "forge/autograd.h"
#include "forge/tensor_ops.h"
#include "forge/reduction_ops.h"
#include "forge/comparison_ops.h"
#include "forge/nn/conv2d.cuh"

#include <cstdio>
#include <vector>

namespace forge {

// =====================================================================
// Helper: reduce_grad for broadcast binary ops
// =====================================================================

static Tensor maybe_reduce(const Tensor& grad, const std::vector<int64_t>& shape) {
    return reduce_grad_to_shape(grad, shape.data(), static_cast<int>(shape.size()));
}

// =====================================================================
// Binary tensor-tensor backward nodes
// =====================================================================

class AddBackward : public Node {
public:
    std::vector<int64_t> a_shape_, b_shape_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        const Tensor& g = grads[0];
        return { maybe_reduce(g, a_shape_),
                 maybe_reduce(g, b_shape_) };
    }
};

class SubBackward : public Node {
public:
    std::vector<int64_t> a_shape_, b_shape_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        const Tensor& g = grads[0];
        return { maybe_reduce(g, a_shape_),
                 maybe_reduce(neg(g), b_shape_) };
    }
};

class MulBackward : public Node {
public:
    Tensor saved_a_, saved_b_;
    std::vector<int64_t> a_shape_, b_shape_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        const Tensor& g = grads[0];
        return { maybe_reduce(mul(g, saved_b_), a_shape_),
                 maybe_reduce(mul(g, saved_a_), b_shape_) };
    }
    void release_variables() override { saved_a_ = Tensor(); saved_b_ = Tensor(); }
};

class DivBackward : public Node {
public:
    Tensor saved_a_, saved_b_;
    std::vector<int64_t> a_shape_, b_shape_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        const Tensor& g = grads[0];
        // d_a = g / b
        // d_b = -g * a / (b * b)
        Tensor ga = div(g, saved_b_);
        Tensor gb = neg(div(mul(g, saved_a_), mul(saved_b_, saved_b_)));
        return { maybe_reduce(ga, a_shape_),
                 maybe_reduce(gb, b_shape_) };
    }
    void release_variables() override { saved_a_ = Tensor(); saved_b_ = Tensor(); }
};

// =====================================================================
// Binary tensor-scalar backward nodes
// =====================================================================

class AddScalarBackward : public Node {
public:
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        return { grads[0] };
    }
};

class SubScalarBackward : public Node {
public:
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        return { grads[0] };
    }
};

class MulScalarBackward : public Node {
public:
    float scalar_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        return { mul(grads[0], scalar_) };
    }
};

class DivScalarBackward : public Node {
public:
    float scalar_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        return { div(grads[0], scalar_) };
    }
};

// =====================================================================
// Unary backward nodes
// =====================================================================

class NegBackward : public Node {
public:
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        return { neg(grads[0]) };
    }
};

class ExpBackward : public Node {
public:
    Tensor saved_result_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        return { mul(grads[0], saved_result_) };
    }
    void release_variables() override { saved_result_ = Tensor(); }
};

class LogBackward : public Node {
public:
    Tensor saved_input_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        return { div(grads[0], saved_input_) };
    }
    void release_variables() override { saved_input_ = Tensor(); }
};

class SqrtBackward : public Node {
public:
    Tensor saved_result_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        // d_x = g / (2 * sqrt(x)) = g / (2 * result)
        return { div(grads[0], mul(saved_result_, 2.0f)) };
    }
    void release_variables() override { saved_result_ = Tensor(); }
};

class RsqrtBackward : public Node {
public:
    Tensor saved_input_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        // d_x = -g / (2 * x^(3/2)) = -g * rsqrt(x)^3 / 2
        Tensor rsqrt_x = rsqrt(saved_input_);
        return { mul(neg(grads[0]), mul(mul(rsqrt_x, mul(rsqrt_x, rsqrt_x)), 0.5f)) };
    }
    void release_variables() override { saved_input_ = Tensor(); }
};

class SquareBackward : public Node {
public:
    Tensor saved_input_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        return { mul(grads[0], mul(saved_input_, 2.0f)) };
    }
    void release_variables() override { saved_input_ = Tensor(); }
};

class ReciprocalBackward : public Node {
public:
    Tensor saved_result_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        // d_x = -g * (1/x)^2 = -g * result^2
        return { neg(mul(grads[0], mul(saved_result_, saved_result_))) };
    }
    void release_variables() override { saved_result_ = Tensor(); }
};

class AbsBackward : public Node {
public:
    Tensor saved_input_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        return { mul(grads[0], sign(saved_input_)) };
    }
    void release_variables() override { saved_input_ = Tensor(); }
};

class SinBackward : public Node {
public:
    Tensor saved_input_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        return { mul(grads[0], cos(saved_input_)) };
    }
    void release_variables() override { saved_input_ = Tensor(); }
};

class CosBackward : public Node {
public:
    Tensor saved_input_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        return { neg(mul(grads[0], sin(saved_input_))) };
    }
    void release_variables() override { saved_input_ = Tensor(); }
};

class TanhBackward : public Node {
public:
    Tensor saved_result_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        // d_x = g * (1 - tanh(x)^2) = g * (1 - result^2)
        Tensor one_minus_sq = sub(Tensor::ones({1}), mul(saved_result_, saved_result_));
        return { mul(grads[0], one_minus_sq) };
    }
    void release_variables() override { saved_result_ = Tensor(); }
};

class SigmoidBackward : public Node {
public:
    Tensor saved_result_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        // d_x = g * sigmoid(x) * (1 - sigmoid(x)) = g * result * (1 - result)
        Tensor one_minus = sub(Tensor::ones({1}), saved_result_);
        return { mul(grads[0], mul(saved_result_, one_minus)) };
    }
    void release_variables() override { saved_result_ = Tensor(); }
};

class ClampBackward : public Node {
public:
    Tensor saved_input_;
    float lo_, hi_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        // d_x = g where lo <= x <= hi, else 0
        Tensor mask_lo = ge(saved_input_, lo_);
        Tensor mask_hi = le(saved_input_, hi_);
        Tensor mask = logical_and(mask_lo, mask_hi);
        // Convert Bool mask to float: where(mask, grad, zeros)
        Tensor zeros = Tensor::zeros(grads[0].sizes(), grads[0].ndim());
        return { where(mask, grads[0], zeros) };
    }
    void release_variables() override { saved_input_ = Tensor(); }
};

class PowBackward : public Node {
public:
    Tensor saved_input_;
    float exponent_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        // d_x = g * e * x^(e-1)
        return { mul(mul(grads[0], exponent_), pow(saved_input_, exponent_ - 1.0f)) };
    }
    void release_variables() override { saved_input_ = Tensor(); }
};

// =====================================================================
// Reduction backward nodes
// =====================================================================

class SumBackward : public Node {
public:
    std::vector<int64_t> input_shape_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        // Gradient of sum is ones broadcast to input shape
        return { grads[0].expand(input_shape_.data(),
                                 static_cast<int>(input_shape_.size())).contiguous() };
    }
};

class SumDimBackward : public Node {
public:
    std::vector<int64_t> input_shape_;
    int dim_;
    bool keepdim_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        Tensor g = grads[0];
        // If keepdim was false, we need to unsqueeze to get back the reduced dim
        if (!keepdim_) {
            g = g.unsqueeze(dim_);
        }
        return { g.expand(input_shape_.data(),
                          static_cast<int>(input_shape_.size())).contiguous() };
    }
};

class MeanBackward : public Node {
public:
    std::vector<int64_t> input_shape_;
    int64_t numel_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        float scale = 1.0f / static_cast<float>(numel_);
        Tensor scaled = mul(grads[0], scale);
        return { scaled.expand(input_shape_.data(),
                               static_cast<int>(input_shape_.size())).contiguous() };
    }
};

class MeanDimBackward : public Node {
public:
    std::vector<int64_t> input_shape_;
    int dim_;
    bool keepdim_;
    int64_t reduce_size_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        Tensor g = grads[0];
        if (!keepdim_) {
            g = g.unsqueeze(dim_);
        }
        float scale = 1.0f / static_cast<float>(reduce_size_);
        Tensor scaled = mul(g, scale);
        return { scaled.expand(input_shape_.data(),
                               static_cast<int>(input_shape_.size())).contiguous() };
    }
};

// =====================================================================
// Graph recording functions (called from forward ops)
// =====================================================================

// ----- Binary tensor-tensor -----

void record_add(Tensor& out, const Tensor& a, const Tensor& b) {
    auto node = std::make_shared<AddBackward>();
    node->a_shape_.assign(a.sizes(), a.sizes() + a.ndim());
    node->b_shape_.assign(b.sizes(), b.sizes() + b.ndim());
    node->next_edges_ = {get_gradient_edge(a), get_gradient_edge(b)};
    set_grad_fn(out, node);
}

void record_sub(Tensor& out, const Tensor& a, const Tensor& b) {
    auto node = std::make_shared<SubBackward>();
    node->a_shape_.assign(a.sizes(), a.sizes() + a.ndim());
    node->b_shape_.assign(b.sizes(), b.sizes() + b.ndim());
    node->next_edges_ = {get_gradient_edge(a), get_gradient_edge(b)};
    set_grad_fn(out, node);
}

void record_mul(Tensor& out, const Tensor& a, const Tensor& b) {
    auto node = std::make_shared<MulBackward>();
    node->saved_a_ = a;
    node->saved_b_ = b;
    node->a_shape_.assign(a.sizes(), a.sizes() + a.ndim());
    node->b_shape_.assign(b.sizes(), b.sizes() + b.ndim());
    node->next_edges_ = {get_gradient_edge(a), get_gradient_edge(b)};
    set_grad_fn(out, node);
}

void record_div(Tensor& out, const Tensor& a, const Tensor& b) {
    auto node = std::make_shared<DivBackward>();
    node->saved_a_ = a;
    node->saved_b_ = b;
    node->a_shape_.assign(a.sizes(), a.sizes() + a.ndim());
    node->b_shape_.assign(b.sizes(), b.sizes() + b.ndim());
    node->next_edges_ = {get_gradient_edge(a), get_gradient_edge(b)};
    set_grad_fn(out, node);
}

// ----- Binary tensor-scalar -----

void record_add_scalar(Tensor& out, const Tensor& a) {
    auto node = std::make_shared<AddScalarBackward>();
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_sub_scalar(Tensor& out, const Tensor& a) {
    auto node = std::make_shared<SubScalarBackward>();
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_mul_scalar(Tensor& out, const Tensor& a, float s) {
    auto node = std::make_shared<MulScalarBackward>();
    node->scalar_ = s;
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_div_scalar(Tensor& out, const Tensor& a, float s) {
    auto node = std::make_shared<DivScalarBackward>();
    node->scalar_ = s;
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

// ----- Unary -----

void record_neg(Tensor& out, const Tensor& a) {
    auto node = std::make_shared<NegBackward>();
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_exp(Tensor& out, const Tensor& a) {
    auto node = std::make_shared<ExpBackward>();
    node->saved_result_ = out;  // save result for backward
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_log(Tensor& out, const Tensor& a) {
    auto node = std::make_shared<LogBackward>();
    node->saved_input_ = a;
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_sqrt(Tensor& out, const Tensor& a) {
    auto node = std::make_shared<SqrtBackward>();
    node->saved_result_ = out;
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_rsqrt(Tensor& out, const Tensor& a) {
    auto node = std::make_shared<RsqrtBackward>();
    node->saved_input_ = a;
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_square(Tensor& out, const Tensor& a) {
    auto node = std::make_shared<SquareBackward>();
    node->saved_input_ = a;
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_reciprocal(Tensor& out, const Tensor& a) {
    auto node = std::make_shared<ReciprocalBackward>();
    node->saved_result_ = out;
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_abs(Tensor& out, const Tensor& a) {
    auto node = std::make_shared<AbsBackward>();
    node->saved_input_ = a;
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_sin(Tensor& out, const Tensor& a) {
    auto node = std::make_shared<SinBackward>();
    node->saved_input_ = a;
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_cos(Tensor& out, const Tensor& a) {
    auto node = std::make_shared<CosBackward>();
    node->saved_input_ = a;
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_tanh(Tensor& out, const Tensor& a) {
    auto node = std::make_shared<TanhBackward>();
    node->saved_result_ = out;
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_sigmoid(Tensor& out, const Tensor& a) {
    auto node = std::make_shared<SigmoidBackward>();
    node->saved_result_ = out;
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_clamp(Tensor& out, const Tensor& a, float lo, float hi) {
    auto node = std::make_shared<ClampBackward>();
    node->saved_input_ = a;
    node->lo_ = lo;
    node->hi_ = hi;
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_pow(Tensor& out, const Tensor& a, float exponent) {
    auto node = std::make_shared<PowBackward>();
    node->saved_input_ = a;
    node->exponent_ = exponent;
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

// ----- Reductions -----

void record_sum(Tensor& out, const Tensor& a) {
    auto node = std::make_shared<SumBackward>();
    node->input_shape_.assign(a.sizes(), a.sizes() + a.ndim());
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_sum_dim(Tensor& out, const Tensor& a, int dim, bool keepdim) {
    auto node = std::make_shared<SumDimBackward>();
    node->input_shape_.assign(a.sizes(), a.sizes() + a.ndim());
    node->dim_ = dim;
    node->keepdim_ = keepdim;
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_mean(Tensor& out, const Tensor& a) {
    auto node = std::make_shared<MeanBackward>();
    node->input_shape_.assign(a.sizes(), a.sizes() + a.ndim());
    node->numel_ = a.numel();
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

void record_mean_dim(Tensor& out, const Tensor& a, int dim, bool keepdim) {
    auto node = std::make_shared<MeanDimBackward>();
    node->input_shape_.assign(a.sizes(), a.sizes() + a.ndim());
    node->dim_ = dim;
    node->keepdim_ = keepdim;
    node->reduce_size_ = a.size(dim);
    node->next_edges_ = {get_gradient_edge(a)};
    set_grad_fn(out, node);
}

// =====================================================================
// Conv2d backward node
// =====================================================================

class Conv2dBackward : public Node {
public:
    Tensor saved_input_, saved_weight_;
    int C_in_, H_, W_, C_out_, kH_, kW_;
    int padH_, padW_, strideH_, strideW_, dilH_, dilW_, groups_;
    bool has_bias_;

    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override {
        const Tensor& grad_output = grads[0];

        int H_out = (H_ + 2 * padH_ - dilH_ * (kH_ - 1) - 1) / strideH_ + 1;
        int W_out = (W_ + 2 * padW_ - dilW_ * (kW_ - 1) - 1) / strideW_ + 1;
        int N = static_cast<int>(grad_output.size(0));

        // d_input
        Tensor d_input = Tensor::zeros({N, C_in_, H_, W_});
        nn::conv2d_backward_data(
            grad_output.data<float>(), saved_weight_.data<float>(),
            d_input.data<float>(),
            N, C_in_, H_, W_, C_out_, kH_, kW_,
            padH_, padW_, strideH_, strideW_, dilH_, dilW_, groups_);

        // d_weight
        int C_in_per_group = C_in_ / groups_;
        Tensor d_weight = Tensor::zeros({C_out_, C_in_per_group, kH_, kW_});
        float* d_bias_ptr = nullptr;
        Tensor d_bias;
        if (has_bias_) {
            d_bias = Tensor::zeros({C_out_});
            d_bias_ptr = d_bias.data<float>();
        }
        nn::conv2d_backward_weight(
            grad_output.data<float>(), saved_input_.data<float>(),
            d_weight.data<float>(), d_bias_ptr,
            N, C_in_, H_, W_, C_out_, kH_, kW_,
            padH_, padW_, strideH_, strideW_, dilH_, dilW_, groups_, 0.0f);

        cudaDeviceSynchronize();

        if (has_bias_) {
            return { d_input, d_weight, d_bias };
        }
        return { d_input, d_weight };
    }

    void release_variables() override {
        saved_input_ = Tensor();
        saved_weight_ = Tensor();
    }
};

void record_conv2d(Tensor& out, const Tensor& input, const Tensor& weight,
                   const Tensor& bias,
                   int C_in, int H, int W,
                   int C_out, int kH, int kW,
                   int padH, int padW,
                   int strideH, int strideW,
                   int dilH, int dilW, int groups,
                   bool has_bias) {
    auto node = std::make_shared<Conv2dBackward>();
    node->saved_input_ = input;
    node->saved_weight_ = weight;
    node->C_in_ = C_in; node->H_ = H; node->W_ = W;
    node->C_out_ = C_out; node->kH_ = kH; node->kW_ = kW;
    node->padH_ = padH; node->padW_ = padW;
    node->strideH_ = strideH; node->strideW_ = strideW;
    node->dilH_ = dilH; node->dilW_ = dilW;
    node->groups_ = groups;
    node->has_bias_ = has_bias;

    node->next_edges_.push_back(get_gradient_edge(input));
    node->next_edges_.push_back(get_gradient_edge(weight));
    if (has_bias && bias.defined()) {
        node->next_edges_.push_back(get_gradient_edge(bias));
    }

    set_grad_fn(out, node);
}

} // namespace forge
