// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/embedding.cuh>
#include "test_helpers.cuh"
#include <cmath>

using namespace forge;

int main() {
    std::printf("=== Test: Embedding ===\n");
    const int V = 64;        // vocab size
    const int E = 32;        // embed dim
    const int B = 4;         // batch
    const int ctx = 3;       // context size
    const int I = ctx * E;   // output rows

    // Random embedding table [V, E] row-major
    std::vector<float> h_embed(V * E);
    test::fill_random(h_embed, -1.0f, 1.0f, 42);

    // Random context tokens [B * ctx]
    std::vector<int> h_ctx(B * ctx);
    std::mt19937 rng(99);
    for (auto& t : h_ctx) t = rng() % V;

    // CPU gather: output[I, B] col-major
    // context layout: context[b * ctx + c] = token
    // output[(c * E + e) * B + b] = embed[token * E + e]
    std::vector<float> ref_output(I * B, 0.0f);
    for (int b = 0; b < B; b++)
        for (int c = 0; c < ctx; c++) {
            int tok = h_ctx[b * ctx + c];
            for (int e = 0; e < E; e++)
                ref_output[(c * E + e) * B + b] = h_embed[tok * E + e];
        }

    // GPU gather
    int* d_ctx     = test::to_device(h_ctx);
    float* d_embed = test::to_device(h_embed);
    float* d_out   = test::to_device_zeroed<float>(I * B);

    nn::embedding_gather(d_ctx, d_embed, d_out, B, ctx, E);
    cudaDeviceSynchronize();

    auto gpu_out = test::to_host(d_out, I * B);
    FORGE_ASSERT(test::check_arrays(gpu_out, ref_output, 1e-6f, 1e-6f, "gather"),
                 "embedding_gather mismatch");

    // CPU scatter: d_embeddings[tok * E + e] += d_output[(c * E + e) * B + b]
    std::vector<float> h_d_output(I * B);
    test::fill_random(h_d_output, -0.5f, 0.5f, 55);
    std::vector<float> ref_d_embed(V * E, 0.0f);
    for (int b = 0; b < B; b++)
        for (int c = 0; c < ctx; c++) {
            int tok = h_ctx[b * ctx + c];
            for (int e = 0; e < E; e++)
                ref_d_embed[tok * E + e] += h_d_output[(c * E + e) * B + b];
        }

    // GPU scatter
    float* d_d_output = test::to_device(h_d_output);
    float* d_d_embed  = test::to_device_zeroed<float>(V * E);

    nn::embedding_scatter(d_ctx, d_d_output, d_d_embed, B, ctx, E);
    cudaDeviceSynchronize();

    auto gpu_d_embed = test::to_host(d_d_embed, V * E);
    FORGE_ASSERT(test::check_arrays(gpu_d_embed, ref_d_embed, 1e-5f, 1e-4f, "scatter"),
                 "embedding_scatter mismatch");

    test::free_device(d_ctx);
    test::free_device(d_embed);
    test::free_device(d_out);
    test::free_device(d_d_output);
    test::free_device(d_d_embed);
    std::printf("ALL PASSED\n");
    return 0;
}
