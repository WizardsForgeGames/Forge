// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/graph.h>
#include <forge/error.h>

namespace forge {

Graph::~Graph() { destroy(); }

Graph::Graph(Graph&& o) noexcept
    : exec_(o.exec_), stream_(o.stream_)
{
    o.exec_ = nullptr;
    o.stream_ = nullptr;
}

Graph& Graph::operator=(Graph&& o) noexcept {
    if (this != &o) {
        destroy();
        exec_ = o.exec_;
        stream_ = o.stream_;
        o.exec_ = nullptr;
        o.stream_ = nullptr;
    }
    return *this;
}

void Graph::begin_capture(cudaStream_t stream) {
    destroy();  // drop any previous graph
    stream_ = stream;
    FORGE_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
}

void Graph::end_capture() {
    cudaGraph_t graph = nullptr;
    FORGE_CHECK(cudaStreamEndCapture(stream_, &graph));
    FORGE_CHECK(cudaGraphInstantiate(&exec_, graph, 0));
    cudaGraphDestroy(graph);  // exec_ holds the instantiated copy
    stream_ = nullptr;
}

void Graph::launch(cudaStream_t stream) {
    FORGE_CHECK(cudaGraphLaunch(exec_, stream));
}

void Graph::destroy() {
    if (exec_) {
        cudaGraphExecDestroy(exec_);
        exec_ = nullptr;
    }
    stream_ = nullptr;
}

} // namespace forge
