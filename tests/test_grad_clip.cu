// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/optim/grad_clip.cuh>
#include "test_helpers.cuh"
#include <cmath>

using namespace forge;

int main() {
    std::printf("=== Test: Grad Clip ===\n");
    const int N = 4096;

    std::vector<float> h_grad(N);
    test::fill_random(h_grad, -2.0f, 2.0f, 42);

    // CPU: compute norm squared
    float cpu_norm_sq = 0.0f;
    for (int i = 0; i < N; i++)
        cpu_norm_sq += h_grad[i] * h_grad[i];

    // GPU: grad_norm_sq
    float* d_grad = test::to_device(h_grad);
    float* d_norm_sq = test::to_device_zeroed<float>(1);

    optim::grad_norm_sq(d_grad, N, d_norm_sq);
    cudaDeviceSynchronize();

    auto gpu_norm = test::to_host(d_norm_sq, 1);
    float norm_diff = std::abs(gpu_norm[0] - cpu_norm_sq);
    float norm_rtol = norm_diff / (std::abs(cpu_norm_sq) + 1e-8f);
    std::printf("  grad_norm_sq: gpu=%.6f cpu=%.6f diff=%.6e\n",
                gpu_norm[0], cpu_norm_sq, norm_diff);
    FORGE_ASSERT(norm_rtol < 1e-3f, "grad_norm_sq mismatch");
    std::printf("  PASS [grad_norm_sq]\n");

    // GPU: compute_clip_scale
    const float max_norm = 1.0f;
    const float inv_B = 1.0f / 32.0f;
    float* d_clip_scale = test::to_device_zeroed<float>(1);

    optim::compute_clip_scale(d_clip_scale, d_norm_sq, max_norm, inv_B);
    cudaDeviceSynchronize();

    auto gpu_scale = test::to_host(d_clip_scale, 1);
    float actual_norm = std::sqrt(gpu_norm[0]);
    float eff_norm = actual_norm * inv_B;
    float cpu_clip_scale = (eff_norm > max_norm) ? (max_norm / actual_norm) : inv_B;
    float scale_diff = std::abs(gpu_scale[0] - cpu_clip_scale);
    std::printf("  clip_scale: gpu=%.8f cpu=%.8f diff=%.6e\n",
                gpu_scale[0], cpu_clip_scale, scale_diff);
    FORGE_ASSERT(scale_diff < 1e-5f, "compute_clip_scale mismatch");
    std::printf("  PASS [compute_clip_scale]\n");

    // GPU: grad_scale
    float scale_val = 0.5f;
    optim::grad_scale(d_grad, N, scale_val);
    cudaDeviceSynchronize();

    auto gpu_scaled = test::to_host(d_grad, N);
    std::vector<float> ref_scaled(N);
    for (int i = 0; i < N; i++) ref_scaled[i] = h_grad[i] * scale_val;
    FORGE_ASSERT(test::check_arrays(gpu_scaled, ref_scaled, 1e-5f, 1e-5f, "grad_scale"),
                 "grad_scale mismatch");

    test::free_device(d_grad);
    test::free_device(d_norm_sq);
    test::free_device(d_clip_scale);
    std::printf("ALL PASSED\n");
    return 0;
}
