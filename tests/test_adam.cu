// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/optim/adam.cuh>
#include <forge/optim/lr_schedule.h>
#include "test_helpers.cuh"
#include <cmath>

using namespace forge;

void adam_cpu_ref(float* param, const float* grad, float* m, float* v,
                 int n, float lr, float lr_scale, float bc1, float bc2,
                 float beta1, float beta2, float eps, float wd, float gs) {
    for (int i = 0; i < n; i++) {
        float g = grad[i] * gs;
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float mh = m[i] / bc1;
        float vh = v[i] / bc2;
        float update = mh / (std::sqrt(vh) + eps);
        if (wd > 0.0f) update += wd * param[i];
        param[i] -= lr * lr_scale * update;
    }
}

int main() {
    std::printf("=== Test: Adam Optimizer ===\n");
    const int N = 2048;
    const float lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

    std::vector<float> h_param(N), h_grad(N), h_m(N, 0.0f), h_v(N, 0.0f);
    test::fill_random(h_param, -1.0f, 1.0f, 10);
    test::fill_random(h_grad, -0.5f, 0.5f, 11);

    // GPU copies
    auto h_param_gpu = h_param;
    float* d_param = test::to_device(h_param_gpu);
    float* d_grad  = test::to_device(h_grad);
    float* d_m     = test::to_device_zeroed<float>(N);
    float* d_v     = test::to_device_zeroed<float>(N);
    float* d_scalars = test::to_device_zeroed<float>(3);
    float* d_gs    = test::to_device_zeroed<float>(1);

    // Set grad_scale = 1.0
    float one = 1.0f;
    cudaMemcpy(d_gs, &one, sizeof(float), cudaMemcpyHostToDevice);

    // Run 10 steps
    for (int t = 1; t <= 10; t++) {
        float lr_scale = 1.0f;
        float bc1 = 1.0f - std::pow(beta1, (float)t);
        float bc2 = 1.0f - std::pow(beta2, (float)t);

        // CPU reference
        adam_cpu_ref(h_param.data(), h_grad.data(), h_m.data(), h_v.data(),
                     N, lr, lr_scale, bc1, bc2, beta1, beta2, eps, 0.0f, 1.0f);

        // GPU
        float scalars[3] = {lr_scale, bc1, bc2};
        cudaMemcpy(d_scalars, scalars, 3 * sizeof(float), cudaMemcpyHostToDevice);
        optim::adam_step(d_param, d_grad, d_m, d_v, N, lr, d_scalars,
                         beta1, beta2, eps, 0.0f, d_gs);
        cudaDeviceSynchronize();
    }

    auto gpu_param = test::to_host(d_param, N);
    FORGE_ASSERT(test::check_arrays(gpu_param, h_param, 1e-4f, 1e-3f, "adam_10steps"),
                 "Adam 10-step mismatch");

    test::free_device(d_param);
    test::free_device(d_grad);
    test::free_device(d_m);
    test::free_device(d_v);
    test::free_device(d_scalars);
    test::free_device(d_gs);
    std::printf("ALL PASSED\n");
    return 0;
}
