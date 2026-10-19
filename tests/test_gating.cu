// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/gating.cuh>
#include "test_helpers.cuh"
#include <cmath>

using namespace forge;

// CPU reference: sigmoid gate forward
void gate_fwd_cpu(const float* x, const float* w, const float* b,
                  float* gated, int N, int B) {
    for (int i = 0; i < N * B; i++) {
        int unit = i / B;
        float gate = 1.0f / (1.0f + std::exp(-(w[unit] * x[i] + b[unit])));
        gated[i] = gate * x[i];
    }
}

// CPU reference: sigmoid gate backward
void gate_bwd_cpu(const float* d_gated, const float* x, const float* w,
                  const float* b, float* dx, float* dw, float* db,
                  int N, int B) {
    for (int j = 0; j < N; j++) { dw[j] = 0.0f; db[j] = 0.0f; }
    for (int i = 0; i < N * B; i++) {
        int unit = i / B;
        float xi = x[i];
        float gate = 1.0f / (1.0f + std::exp(-(w[unit] * xi + b[unit])));
        float gd = gate * (1.0f - gate);
        float dg = d_gated[i];
        dx[i] = dg * (gate + xi * gd * w[unit]);
        dw[unit] += dg * xi * xi * gd;
        db[unit] += dg * xi * gd;
    }
}

int main() {
    FORGE_TEST_BEGIN("Sigmoid Gating");

    const int N = 128;
    const int B = 64;
    const int total = N * B;

    // Setup host data
    std::vector<float> h_x(total), h_w(N), h_b(N), h_d_gated(total);
    test::fill_random(h_x, -2.0f, 2.0f, 42);
    test::fill_random(h_w, -0.5f, 0.5f, 43);
    test::fill_random(h_b, -0.3f, 0.3f, 44);
    test::fill_random(h_d_gated, -1.0f, 1.0f, 45);

    // Device buffers
    float* d_x       = test::to_device(h_x);
    float* d_w       = test::to_device(h_w);
    float* d_b       = test::to_device(h_b);
    float* d_gated   = test::to_device_zeroed<float>(total);
    float* d_d_gated = test::to_device(h_d_gated);
    float* d_dx      = test::to_device_zeroed<float>(total);
    float* d_dw      = test::to_device_zeroed<float>(N);
    float* d_db      = test::to_device_zeroed<float>(N);

    // --- Test forward ---
    {
        nn::sigmoid_gate_forward(d_x, d_w, d_b, d_gated, N, B);
        cudaDeviceSynchronize();
        auto gpu = test::to_host(d_gated, total);
        std::vector<float> ref(total);
        gate_fwd_cpu(h_x.data(), h_w.data(), h_b.data(), ref.data(), N, B);
        FORGE_ASSERT(test::check_arrays(gpu, ref, 1e-5f, 1e-4f, "gate_forward"),
                     "sigmoid_gate_forward mismatch");
    }

    // --- Test backward ---
    {
        nn::sigmoid_gate_backward(d_d_gated, d_x, d_w, d_b, d_dx, d_dw, d_db, N, B);
        cudaDeviceSynchronize();

        // CPU reference
        std::vector<float> ref_dx(total), ref_dw(N), ref_db(N);
        gate_bwd_cpu(h_d_gated.data(), h_x.data(), h_w.data(), h_b.data(),
                     ref_dx.data(), ref_dw.data(), ref_db.data(), N, B);

        auto gpu_dx = test::to_host(d_dx, total);
        auto gpu_dw = test::to_host(d_dw, N);
        auto gpu_db = test::to_host(d_db, N);

        FORGE_ASSERT(test::check_arrays(gpu_dx, ref_dx, 1e-5f, 1e-4f, "gate_bwd_dx"),
                     "sigmoid_gate_backward dx mismatch");
        // Relaxed tolerance for atomicAdd accumulation over B elements
        FORGE_ASSERT(test::check_arrays(gpu_dw, ref_dw, 1e-3f, 1e-2f, "gate_bwd_dw"),
                     "sigmoid_gate_backward dw mismatch");
        FORGE_ASSERT(test::check_arrays(gpu_db, ref_db, 1e-3f, 1e-2f, "gate_bwd_db"),
                     "sigmoid_gate_backward db mismatch");
    }

    test::free_device(d_x);
    test::free_device(d_w);
    test::free_device(d_b);
    test::free_device(d_gated);
    test::free_device(d_d_gated);
    test::free_device(d_dx);
    test::free_device(d_dw);
    test::free_device(d_db);

    FORGE_TEST_END();
}
