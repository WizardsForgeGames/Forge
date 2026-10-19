// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include "forge/stream.h"
#include "forge/error.h"

namespace forge {

namespace {

/// Creates a non-default CUDA stream for this thread.
cudaStream_t create_default_stream() {
    cudaStream_t s = nullptr;
    FORGE_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    return s;
}

thread_local cudaStream_t tls_stream = nullptr;
thread_local bool         tls_initialized = false;

} // anonymous namespace

cudaStream_t getCurrentStream() {
    if (!tls_initialized) {
        tls_stream = create_default_stream();
        tls_initialized = true;
    }
    return tls_stream;
}

void setCurrentStream(cudaStream_t stream) {
    tls_stream = stream;
    tls_initialized = true;
}

StreamGuard::StreamGuard(cudaStream_t stream)
    : prev_(getCurrentStream()) {
    setCurrentStream(stream);
}

StreamGuard::~StreamGuard() {
    setCurrentStream(prev_);
}

} // namespace forge
