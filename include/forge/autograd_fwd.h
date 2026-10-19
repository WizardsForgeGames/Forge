// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Forward declarations for the autograd system (FEAT-10).
// Included by tensor_impl.h so TensorImpl can hold a lazy pointer
// to AutogradMeta without pulling in the full autograd headers.
#pragma once

#include <memory>

namespace forge {

class Tensor;  // forward declaration
class Node;    // autograd computation graph node

/// Autograd metadata attached to tensors that require gradients.
/// Lazy-allocated: nullptr for inference-only tensors (zero overhead).
struct AutogradMeta {
    Tensor*                   grad;        // accumulated gradient (owned)
    std::shared_ptr<Node>     grad_fn;     // op that produced this tensor
    std::shared_ptr<Node>     accum_grad;  // AccumulateGrad node for leaves
    int                       output_nr;   // which output of grad_fn
    bool                      retain_grad; // keep grad for non-leaf tensors

    AutogradMeta()
        : grad(nullptr), output_nr(0), retain_grad(false) {}
    ~AutogradMeta();  // defined in tensor.cpp to break circular include
};

} // namespace forge
