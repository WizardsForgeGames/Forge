// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// CUDA Graph wrapper for zero-overhead kernel replay.
//
// Usage:
//   forge::Graph graph;
//   // First iteration: capture the kernel sequence
//   graph.begin_capture(stream);
//   my_kernel_a<<<...>>>();
//   my_kernel_b<<<...>>>();
//   graph.end_capture();
//
//   // Subsequent iterations: replay with zero launch overhead
//   graph.launch(stream);
//
// All captured kernels must use device pointers for dynamic data.
// Host-side scalar arguments are baked at capture time.
// Dynamic data flows through device memory (pointer is fixed, data changes).
//
// Operations that change host-side arguments each iteration (RNG offset,
// adam_t counter) must stay OUTSIDE the capture and be launched before
// graph.launch() on the same stream for correct ordering.
#pragma once

#include <cuda_runtime.h>

namespace forge {

class Graph {
public:
    Graph() = default;
    ~Graph();

    /// Begin stream capture. All CUDA operations on the stream after this
    /// call are recorded into the graph (not executed).
    void begin_capture(cudaStream_t stream = nullptr);

    /// End stream capture and instantiate the executable graph.
    /// After this call, launch() can be used to replay the captured sequence.
    void end_capture();

    /// Replay the captured kernel sequence on the given stream.
    /// All preceding operations on the stream complete before the graph starts.
    void launch(cudaStream_t stream = nullptr);

    /// Returns true if a graph has been captured and is ready to launch.
    bool is_valid() const { return exec_ != nullptr; }

    /// Explicitly destroy the graph. Called automatically by destructor.
    void destroy();

    // Move-only
    Graph(Graph&& o) noexcept;
    Graph& operator=(Graph&& o) noexcept;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

private:
    cudaGraphExec_t exec_   = nullptr;
    cudaStream_t    stream_ = nullptr;  // stream used during capture
};

} // namespace forge
