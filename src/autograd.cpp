// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Autograd backward engine implementation.
//
// backward() traverses the computation graph in reverse topological order
// using BFS with dependency counting. Each Node::apply() computes input
// gradients from output gradients, which are then propagated to child nodes.
#include "forge/autograd.h"
#include "forge/reduction_ops.h"
#include "forge/tensor_ops.h"
#include "forge/error.h"

#include <queue>
#include <unordered_map>
#include <cstdio>

namespace forge {

// =====================================================================
// GradMode: thread-local gradient recording flag
// =====================================================================

static thread_local bool grad_mode_enabled = true;

bool GradMode::is_enabled() { return grad_mode_enabled; }
void GradMode::set_enabled(bool val) { grad_mode_enabled = val; }

// =====================================================================
// AccumulateGrad: accumulate gradient into leaf tensor
// =====================================================================

std::vector<Tensor> AccumulateGrad::apply(const std::vector<Tensor>& grad_outputs) {
    if (grad_outputs.empty() || !grad_outputs[0].defined()) return {};

    const Tensor& grad = grad_outputs[0];

    if (!variable_->autograd_meta_) return {};

    AutogradMeta* meta = variable_->autograd_meta_;
    if (meta->grad == nullptr) {
        // First gradient — clone it
        meta->grad = new Tensor(grad.clone());
    } else {
        // Accumulate — add in-place
        meta->grad->add_(grad);
    }

    return {};  // no further propagation
}

// =====================================================================
// get_gradient_edge: get/create edge for a tensor's gradient
// =====================================================================

Edge get_gradient_edge(const Tensor& t) {
    if (!t.defined()) return Edge();

    TensorImpl* impl = t.unsafeGetTensorImpl();
    if (!impl->requires_grad()) return Edge();

    // Non-leaf tensor: use its grad_fn
    if (impl->autograd_meta_ && impl->autograd_meta_->grad_fn) {
        return Edge(impl->autograd_meta_->grad_fn,
                    impl->autograd_meta_->output_nr);
    }

    // Leaf tensor: get or create AccumulateGrad node
    if (!impl->autograd_meta_) {
        impl->autograd_meta_ = new AutogradMeta();
    }

    AutogradMeta* meta = impl->autograd_meta_;
    if (!meta->accum_grad) {
        meta->accum_grad = std::make_shared<AccumulateGrad>(impl);
    }

    return Edge(meta->accum_grad, 0);
}

// =====================================================================
// set_grad_fn: mark output tensor as non-leaf with given grad_fn
// =====================================================================

void set_grad_fn(Tensor& output, std::shared_ptr<Node> node, int output_nr) {
    TensorImpl* impl = output.unsafeGetTensorImpl();

    // Ensure autograd meta exists
    if (!impl->autograd_meta_) {
        impl->autograd_meta_ = new AutogradMeta();
    }

    impl->autograd_meta_->grad_fn = std::move(node);
    impl->autograd_meta_->output_nr = output_nr;

    // Mark as requires_grad + non-leaf
    impl->requires_grad_.store(true, std::memory_order_relaxed);
    impl->flags_ &= ~kIsLeaf;
}

// =====================================================================
// reduce_grad_to_shape: sum over broadcast dimensions
// =====================================================================

Tensor reduce_grad_to_shape(const Tensor& grad,
                            const int64_t* target_sizes, int target_ndim) {
    // If shapes already match, return as-is
    int grad_ndim = grad.ndim();
    if (grad_ndim == target_ndim) {
        bool same = true;
        for (int i = 0; i < grad_ndim; ++i) {
            if (grad.size(i) != target_sizes[i]) { same = false; break; }
        }
        if (same) return grad;
    }

    // Disable gradient recording for backward computations
    NoGradGuard no_grad;

    Tensor result = grad;

    // Sum over leading dimensions that were added by broadcast
    int extra_dims = grad_ndim - target_ndim;
    for (int i = 0; i < extra_dims; ++i) {
        result = sum(result, 0, /*keepdim=*/false);
    }

    // Sum over dimensions where target has size 1 but grad doesn't
    for (int i = 0; i < target_ndim; ++i) {
        if (target_sizes[i] == 1 && result.size(i) != 1) {
            result = sum(result, i, /*keepdim=*/true);
        }
    }

    return result;
}

// =====================================================================
// backward: reverse-mode automatic differentiation
// =====================================================================

void backward(const Tensor& loss, const Tensor& grad_output) {
    if (!loss.defined()) {
        std::fprintf(stderr, "Forge: backward() called on undefined tensor\n");
        std::abort();
    }

    TensorImpl* impl = loss.unsafeGetTensorImpl();
    if (!impl->autograd_meta_ || !impl->autograd_meta_->grad_fn) {
        std::fprintf(stderr, "Forge: backward() called on tensor without grad_fn "
                     "(is it a leaf or was it created outside autograd?)\n");
        std::abort();
    }

    // Default grad_output = ones({1})
    Tensor grad_out = grad_output.defined()
        ? grad_output
        : Tensor::ones({1});

    // Disable gradient recording during backward
    NoGradGuard no_grad;

    // BFS-based reverse topological traversal
    // 1. Count dependencies for each node
    std::shared_ptr<Node> root = impl->autograd_meta_->grad_fn;

    // Map from Node* to accumulated gradient for each output
    std::unordered_map<Node*, std::vector<Tensor>> node_grads;
    // Map from Node* to dependency count (how many parents still need to send grads)
    std::unordered_map<Node*, int> deps;

    // BFS to count dependencies
    std::queue<Node*> bfs_queue;
    std::unordered_map<Node*, bool> visited;

    bfs_queue.push(root.get());
    visited[root.get()] = true;

    while (!bfs_queue.empty()) {
        Node* n = bfs_queue.front();
        bfs_queue.pop();

        for (const Edge& edge : n->next_edges_) {
            if (!edge.defined()) continue;
            Node* child = edge.node.get();
            deps[child]++;
            if (!visited[child]) {
                visited[child] = true;
                bfs_queue.push(child);
            }
        }
    }

    // 2. Process nodes in topological order (BFS from root)
    // Initialize root with grad_output
    node_grads[root.get()].resize(root->num_outputs_);
    node_grads[root.get()][0] = grad_out;

    // Keep shared_ptrs alive during traversal
    std::unordered_map<Node*, std::shared_ptr<Node>> alive;
    alive[root.get()] = root;

    std::queue<Node*> ready;
    ready.push(root.get());

    while (!ready.empty()) {
        Node* current = ready.front();
        ready.pop();

        // Get accumulated grad_outputs for this node
        auto& grads = node_grads[current];

        // Call apply to compute input gradients
        std::vector<Tensor> input_grads = current->apply(grads);

        // Release saved tensors to break circular references
        // (e.g., TensorImpl -> grad_fn -> saved_result_ -> TensorImpl)
        current->release_variables();

        // Propagate gradients to child nodes
        for (int i = 0; i < static_cast<int>(current->next_edges_.size()); ++i) {
            const Edge& edge = current->next_edges_[i];
            if (!edge.defined()) continue;
            if (i >= static_cast<int>(input_grads.size())) continue;
            if (!input_grads[i].defined()) continue;

            Node* child = edge.node.get();

            // Keep child alive
            alive[child] = edge.node;

            // Accumulate gradient at child's input_nr
            auto& child_grads = node_grads[child];
            if (child_grads.empty()) {
                child_grads.resize(child->num_outputs_);
            }

            int nr = edge.input_nr;
            if (nr >= static_cast<int>(child_grads.size())) {
                child_grads.resize(nr + 1);
            }

            if (!child_grads[nr].defined()) {
                child_grads[nr] = input_grads[i];
            } else {
                // Accumulate
                child_grads[nr] = add(child_grads[nr], input_grads[i]);
            }

            // Decrement dependency count
            deps[child]--;
            if (deps[child] == 0) {
                ready.push(child);
            }
        }

        // Free gradients for processed node
        node_grads.erase(current);
    }
}

} // namespace forge
