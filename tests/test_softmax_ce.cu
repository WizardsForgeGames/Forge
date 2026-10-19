// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/softmax_ce.cuh>
#include "test_helpers.cuh"
#include <cmath>
#include <algorithm>

using namespace forge;

// CPU reference: softmax + cross-entropy
void softmax_ce_cpu(const float* logits, float* probs, const int* targets,
                    float* losses, int* correct, int V, int B) {
    for (int b = 0; b < B; b++) {
        // Find max (col-major: logits[v * B + b])
        float maxval = -1e30f;
        for (int v = 0; v < V; v++)
            maxval = std::max(maxval, logits[v * B + b]);

        // Exp and sum
        float sum = 0.0f;
        for (int v = 0; v < V; v++) {
            probs[v * B + b] = std::exp(logits[v * B + b] - maxval);
            sum += probs[v * B + b];
        }

        // Normalize
        for (int v = 0; v < V; v++)
            probs[v * B + b] /= sum;

        // Loss and accuracy
        int t = targets[b];
        losses[b] = -std::log(std::max(probs[t * B + b], 1e-10f));

        int argmax = 0;
        float argmax_val = probs[0 * B + b];
        for (int v = 1; v < V; v++) {
            if (probs[v * B + b] > argmax_val) {
                argmax_val = probs[v * B + b];
                argmax = v;
            }
        }
        correct[b] = (argmax == t) ? 1 : 0;
    }
}

int main() {
    std::printf("=== Test: Softmax CE ===\n");
    const int V = 128, B = 16;

    // Generate random logits and targets
    std::vector<float> h_logits(V * B);
    test::fill_random(h_logits, -5.0f, 5.0f, 42);
    std::vector<int> h_targets(B);
    std::mt19937 rng(99);
    for (int b = 0; b < B; b++)
        h_targets[b] = rng() % V;

    // CPU reference
    std::vector<float> ref_probs(V * B);
    std::vector<float> ref_losses(B);
    std::vector<int> ref_correct(B);
    // Copy logits since GPU kernel modifies them in-place
    auto logits_copy = h_logits;
    softmax_ce_cpu(logits_copy.data(), ref_probs.data(), h_targets.data(),
                   ref_losses.data(), ref_correct.data(), V, B);

    // GPU: forward
    float* d_logits  = test::to_device(h_logits);
    float* d_probs   = test::to_device_zeroed<float>(V * B);
    int*   d_targets = test::to_device(h_targets);
    float* d_losses  = test::to_device_zeroed<float>(B);
    int*   d_correct = test::to_device_zeroed<int>(B);

    nn::softmax_ce_forward(d_logits, d_probs, d_targets, d_losses, d_correct, V, B);
    cudaDeviceSynchronize();

    auto gpu_probs  = test::to_host(d_probs, V * B);
    auto gpu_losses = test::to_host(d_losses, B);
    auto gpu_correct = test::to_host(d_correct, B);

    // Check probs sum to ~1 per sample
    bool probs_ok = true;
    for (int b = 0; b < B; b++) {
        float sum = 0.0f;
        for (int v = 0; v < V; v++) sum += gpu_probs[v * B + b];
        if (std::abs(sum - 1.0f) > 1e-4f) {
            std::fprintf(stderr, "FAIL: probs[b=%d] sum=%.6f\n", b, sum);
            probs_ok = false;
        }
    }
    FORGE_ASSERT(probs_ok, "softmax probs don't sum to 1");
    std::printf("  PASS [probs sum to 1]\n");

    FORGE_ASSERT(test::check_arrays(gpu_probs, ref_probs, 1e-4f, 1e-3f, "probs"),
                 "softmax probs mismatch");
    FORGE_ASSERT(test::check_arrays(gpu_losses, ref_losses, 1e-4f, 1e-3f, "losses"),
                 "CE losses mismatch");

    // Check correct flags
    bool correct_ok = true;
    for (int b = 0; b < B; b++) {
        if (gpu_correct[b] != ref_correct[b]) {
            std::fprintf(stderr, "FAIL: correct[%d] gpu=%d ref=%d\n",
                         b, gpu_correct[b], ref_correct[b]);
            correct_ok = false;
        }
    }
    FORGE_ASSERT(correct_ok, "accuracy flags mismatch");
    std::printf("  PASS [correct flags]\n");

    // GPU: backward
    float* d_dlogits = test::to_device_zeroed<float>(V * B);
    nn::softmax_ce_backward(d_probs, d_targets, d_dlogits, V, B);
    cudaDeviceSynchronize();

    auto gpu_dlogits = test::to_host(d_dlogits, V * B);

    // CPU backward: d_logits = probs - one_hot(target)
    std::vector<float> ref_dlogits(V * B);
    for (int b = 0; b < B; b++) {
        for (int v = 0; v < V; v++) {
            ref_dlogits[v * B + b] = ref_probs[v * B + b]
                - ((v == h_targets[b]) ? 1.0f : 0.0f);
        }
    }
    FORGE_ASSERT(test::check_arrays(gpu_dlogits, ref_dlogits, 1e-5f, 1e-4f, "backward"),
                 "softmax_ce backward mismatch");

    test::free_device(d_logits);
    test::free_device(d_probs);
    test::free_device(d_targets);
    test::free_device(d_losses);
    test::free_device(d_correct);
    test::free_device(d_dlogits);
    std::printf("ALL PASSED\n");
    return 0;
}
