// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Autograd engine: computation graph, backward traversal, gradient mode.
//
// Architecture:
//   Node: abstract base class for backward functions
//   Edge: {shared_ptr<Node>, input_nr} — link between nodes
//   AccumulateGrad: terminal leaf node that accumulates into tensor.grad
//   GradMode: thread-local flag controlling graph recording
//   NoGradGuard: RAII scope that disables gradient recording
//
// Usage:
//   Tensor x = Tensor::ones({3});
//   x.set_requires_grad_(true);
//   Tensor y = sum(x * x);
//   backward(y);
//   // x.grad() now contains 2*x
#pragma once

#include "forge/autograd_fwd.h"
#include "forge/tensor.h"

#include <memory>
#include <vector>

namespace forge {

// =====================================================================
// Edge: connection between nodes in the computation graph
// =====================================================================

struct Edge {
    std::shared_ptr<Node> node;
    int input_nr;

    Edge() : input_nr(0) {}
    Edge(std::shared_ptr<Node> n, int nr) : node(std::move(n)), input_nr(nr) {}

    bool defined() const { return node != nullptr; }
};

// =====================================================================
// Node: abstract base class for backward functions
// =====================================================================

class Node : public std::enable_shared_from_this<Node> {
public:
    virtual ~Node() = default;

    /// Compute gradients for inputs given gradients of outputs.
    virtual std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) = 0;

    /// Release saved tensors after backward to break circular references.
    /// Nodes that save the forward output (e.g., ExpBackward, TanhBackward)
    /// create a cycle: TensorImpl -> grad_fn -> saved_result_ -> TensorImpl.
    virtual void release_variables() {}

    /// Edges to the nodes that produced this node's inputs.
    std::vector<Edge> next_edges_;

    /// Number of outputs this node produces in forward.
    int num_outputs_ = 1;

    void set_next_edge(int index, Edge edge) {
        if (static_cast<int>(next_edges_.size()) <= index)
            next_edges_.resize(index + 1);
        next_edges_[index] = std::move(edge);
    }
};

// =====================================================================
// AccumulateGrad: terminal node for leaf tensors
// =====================================================================

class AccumulateGrad : public Node {
public:
    TensorImpl* variable_;  // weak ref — owned by user's Tensor handle

    explicit AccumulateGrad(TensorImpl* var) : variable_(var) {
        num_outputs_ = 1;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override;
};

// =====================================================================
// GradMode: thread-local gradient recording toggle
// =====================================================================

struct GradMode {
    static bool is_enabled();
    static void set_enabled(bool val);
};

// =====================================================================
// NoGradGuard: RAII scope that disables gradient recording
// =====================================================================

class NoGradGuard {
    bool prev_;
public:
    NoGradGuard() : prev_(GradMode::is_enabled()) { GradMode::set_enabled(false); }
    ~NoGradGuard() { GradMode::set_enabled(prev_); }

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
};

// =====================================================================
// Backward engine
// =====================================================================

/// Run backward pass from a scalar loss tensor.
/// grad_output defaults to ones({1}) if not provided.
void backward(const Tensor& loss, const Tensor& grad_output = Tensor());

// =====================================================================
// Autograd helpers (used by forward ops to build graph)
// =====================================================================

/// Should we record this operation in the graph?
inline bool should_record(const Tensor& a) {
    return GradMode::is_enabled() && a.requires_grad();
}
inline bool should_record(const Tensor& a, const Tensor& b) {
    return GradMode::is_enabled() && (a.requires_grad() || b.requires_grad());
}

/// Get or create the gradient edge for a tensor.
/// For leaf tensors: returns Edge to AccumulateGrad node.
/// For intermediate tensors: returns Edge to grad_fn.
Edge get_gradient_edge(const Tensor& t);

/// Set the grad_fn on an output tensor (marks it as non-leaf).
void set_grad_fn(Tensor& output, std::shared_ptr<Node> node, int output_nr = 0);

/// Reduce a gradient tensor to match a target shape (for broadcast).
/// Sums over dimensions that were broadcast in the forward pass.
Tensor reduce_grad_to_shape(const Tensor& grad,
                            const int64_t* target_sizes, int target_ndim);

} // namespace forge
