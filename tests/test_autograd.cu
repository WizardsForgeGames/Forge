// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Tests for the autograd engine: backward pass, gradient computation,
// GradMode, NoGradGuard, and backward rules for core operations.
#include <forge/tensor.h>
#include <forge/tensor_ops.h>
#include <forge/reduction_ops.h>
#include <forge/autograd.h>
#include "test_helpers.cuh"
#include <cmath>

using namespace forge;

int main() {
    std::printf("=== Test: Autograd Engine ===\n");

    // ------ basic_add_backward ------
    {
        // c = a + b (scalars); sum(c).backward() → a.grad=1, b.grad=1
        std::vector<float> ha = {3.0f};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {1});
        a.set_requires_grad_(true);

        std::vector<float> hb = {5.0f};
        float* db = test::to_device(hb);
        Tensor b = Tensor::from_ptr(db, {1});
        b.set_requires_grad_(true);

        Tensor c = add(a, b);
        Tensor loss = sum(c);
        loss.backward();
        cudaDeviceSynchronize();

        auto ga = test::to_host(a.grad().data<float>(), 1);
        auto gb = test::to_host(b.grad().data<float>(), 1);
        FORGE_ASSERT(test::approx_eq(ga[0], 1.0f), "add_backward: a.grad should be 1");
        FORGE_ASSERT(test::approx_eq(gb[0], 1.0f), "add_backward: b.grad should be 1");
        std::printf("  PASS [basic_add_backward]\n");

        loss = Tensor(); c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ mul_backward ------
    {
        std::vector<float> ha = {3.0f};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {1});
        a.set_requires_grad_(true);

        std::vector<float> hb = {5.0f};
        float* db = test::to_device(hb);
        Tensor b = Tensor::from_ptr(db, {1});
        b.set_requires_grad_(true);

        Tensor c = mul(a, b);
        Tensor loss = sum(c);
        loss.backward();
        cudaDeviceSynchronize();

        auto ga = test::to_host(a.grad().data<float>(), 1);
        auto gb = test::to_host(b.grad().data<float>(), 1);
        FORGE_ASSERT(test::approx_eq(ga[0], 5.0f), "mul_backward: a.grad should be b=5");
        FORGE_ASSERT(test::approx_eq(gb[0], 3.0f), "mul_backward: b.grad should be a=3");
        std::printf("  PASS [mul_backward]\n");

        loss = Tensor(); c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ chain_rule ------
    {
        // y = exp(x*2 + 1); dy/dx = exp(x*2+1) * 2
        std::vector<float> hx = {1.0f};
        float* dx = test::to_device(hx);
        Tensor x = Tensor::from_ptr(dx, {1});
        x.set_requires_grad_(true);

        Tensor y = exp(add(mul(x, 2.0f), 1.0f));
        Tensor loss = sum(y);
        loss.backward();
        cudaDeviceSynchronize();

        float expected = std::exp(1.0f * 2.0f + 1.0f) * 2.0f; // exp(3)*2
        auto gx = test::to_host(x.grad().data<float>(), 1);
        FORGE_ASSERT(test::approx_eq(gx[0], expected, 1e-3f, 1e-3f),
                     "chain_rule: gradient mismatch");
        std::printf("  PASS [chain_rule]\n");

        loss = Tensor(); y = Tensor(); x = Tensor();
        cudaFree(dx);
    }

    // ------ sum_backward ------
    {
        // y = sum(x) → x.grad = ones
        std::vector<float> hx = {1.0f, 2.0f, 3.0f, 4.0f};
        float* dx = test::to_device(hx);
        Tensor x = Tensor::from_ptr(dx, {4});
        x.set_requires_grad_(true);

        Tensor y = sum(x);
        y.backward();
        cudaDeviceSynchronize();

        auto gx = test::to_host(x.grad().data<float>(), 4);
        for (int i = 0; i < 4; i++) {
            FORGE_ASSERT(test::approx_eq(gx[i], 1.0f), "sum_backward: grad should be 1");
        }
        std::printf("  PASS [sum_backward]\n");

        y = Tensor(); x = Tensor();
        cudaFree(dx);
    }

    // ------ mean_backward ------
    {
        std::vector<float> hx = {2.0f, 4.0f, 6.0f, 8.0f};
        float* dx = test::to_device(hx);
        Tensor x = Tensor::from_ptr(dx, {4});
        x.set_requires_grad_(true);

        Tensor y = mean(x);
        y.backward();
        cudaDeviceSynchronize();

        auto gx = test::to_host(x.grad().data<float>(), 4);
        float expected = 1.0f / 4.0f;
        for (int i = 0; i < 4; i++) {
            FORGE_ASSERT(test::approx_eq(gx[i], expected),
                         "mean_backward: grad should be 1/n");
        }
        std::printf("  PASS [mean_backward]\n");

        y = Tensor(); x = Tensor();
        cudaFree(dx);
    }

    // ------ sum_dim_backward ------
    {
        // x is {2,3}, sum along dim=1 → {2}
        // Gradient should be ones({2,3})
        std::vector<float> hx = {1,2,3, 4,5,6};
        float* dx = test::to_device(hx);
        Tensor x = Tensor::from_ptr(dx, {2, 3});
        x.set_requires_grad_(true);

        Tensor y = sum(x, 1);  // {2}
        Tensor loss = sum(y);
        loss.backward();
        cudaDeviceSynchronize();

        auto gx = test::to_host(x.grad().data<float>(), 6);
        for (int i = 0; i < 6; i++) {
            FORGE_ASSERT(test::approx_eq(gx[i], 1.0f),
                         "sum_dim_backward: grad should be 1");
        }
        std::printf("  PASS [sum_dim_backward]\n");

        loss = Tensor(); y = Tensor(); x = Tensor();
        cudaFree(dx);
    }

    // ------ broadcast_backward ------
    {
        // a is {3,4}, b is {4}; c = a + b; sum(c).backward()
        // a.grad should be ones({3,4})
        // b.grad should be {3,3,3,3} (summed over dim 0)
        std::vector<float> ha(12, 1.0f);
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {3, 4});
        a.set_requires_grad_(true);

        std::vector<float> hb = {1,2,3,4};
        float* db = test::to_device(hb);
        Tensor b = Tensor::from_ptr(db, {4});
        b.set_requires_grad_(true);

        Tensor c = add(a, b);
        Tensor loss = sum(c);
        loss.backward();
        cudaDeviceSynchronize();

        auto ga = test::to_host(a.grad().data<float>(), 12);
        for (int i = 0; i < 12; i++) {
            FORGE_ASSERT(test::approx_eq(ga[i], 1.0f),
                         "broadcast_backward: a.grad should be 1");
        }

        auto gb = test::to_host(b.grad().data<float>(), 4);
        for (int i = 0; i < 4; i++) {
            FORGE_ASSERT(test::approx_eq(gb[i], 3.0f),
                         "broadcast_backward: b.grad should be 3");
        }
        std::printf("  PASS [broadcast_backward]\n");

        loss = Tensor(); c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ no_grad_guard ------
    {
        std::vector<float> hx = {2.0f};
        float* dx = test::to_device(hx);
        Tensor x = Tensor::from_ptr(dx, {1});
        x.set_requires_grad_(true);

        Tensor y;
        {
            NoGradGuard guard;
            y = mul(x, 3.0f);
        }
        // y should not have a grad_fn since it was computed under NoGradGuard
        FORGE_ASSERT(!y.has_grad_fn(), "no_grad_guard: y should not have grad_fn");
        std::printf("  PASS [no_grad_guard]\n");

        y = Tensor(); x = Tensor();
        cudaFree(dx);
    }

    // ------ detach_stops_grad ------
    {
        std::vector<float> hx = {2.0f};
        float* dx = test::to_device(hx);
        Tensor x = Tensor::from_ptr(dx, {1});
        x.set_requires_grad_(true);

        Tensor d = x.detach();
        Tensor y = mul(d, 3.0f);
        Tensor loss = sum(y);
        // d doesn't require grad, so loss has no grad_fn linking back to x
        FORGE_ASSERT(!loss.has_grad_fn(), "detach_stops_grad: loss should not track x");
        FORGE_ASSERT(!x.grad().defined(), "detach_stops_grad: x.grad should be empty");
        std::printf("  PASS [detach_stops_grad]\n");

        loss = Tensor(); y = Tensor(); d = Tensor(); x = Tensor();
        cudaFree(dx);
    }

    // ------ requires_grad_false ------
    {
        std::vector<float> ha = {1.0f};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {1});
        // No requires_grad

        Tensor y = mul(a, 5.0f);
        FORGE_ASSERT(!y.has_grad_fn(), "requires_grad_false: no grad_fn when no input requires grad");
        std::printf("  PASS [requires_grad_false]\n");

        y = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ multi_use_accumulate ------
    {
        // y = x + x; sum(y).backward() → x.grad == 2
        std::vector<float> hx = {5.0f};
        float* dx = test::to_device(hx);
        Tensor x = Tensor::from_ptr(dx, {1});
        x.set_requires_grad_(true);

        Tensor y = add(x, x);
        Tensor loss = sum(y);
        loss.backward();
        cudaDeviceSynchronize();

        auto gx = test::to_host(x.grad().data<float>(), 1);
        FORGE_ASSERT(test::approx_eq(gx[0], 2.0f),
                     "multi_use: x.grad should be 2");
        std::printf("  PASS [multi_use_accumulate]\n");

        loss = Tensor(); y = Tensor(); x = Tensor();
        cudaFree(dx);
    }

    // ------ scalar_ops_backward ------
    {
        // y = x * 3 + 1; sum(y).backward() → x.grad == 3
        std::vector<float> hx = {7.0f};
        float* dx = test::to_device(hx);
        Tensor x = Tensor::from_ptr(dx, {1});
        x.set_requires_grad_(true);

        Tensor y = add(mul(x, 3.0f), 1.0f);
        Tensor loss = sum(y);
        loss.backward();
        cudaDeviceSynchronize();

        auto gx = test::to_host(x.grad().data<float>(), 1);
        FORGE_ASSERT(test::approx_eq(gx[0], 3.0f),
                     "scalar_ops_backward: x.grad should be 3");
        std::printf("  PASS [scalar_ops_backward]\n");

        loss = Tensor(); y = Tensor(); x = Tensor();
        cudaFree(dx);
    }

    // ------ tanh_backward ------
    {
        std::vector<float> hx = {0.5f};
        float* dx = test::to_device(hx);
        Tensor x = Tensor::from_ptr(dx, {1});
        x.set_requires_grad_(true);

        Tensor y = tanh(x);
        Tensor loss = sum(y);
        loss.backward();
        cudaDeviceSynchronize();

        float th = std::tanh(0.5f);
        float expected = 1.0f - th * th;
        auto gx = test::to_host(x.grad().data<float>(), 1);
        FORGE_ASSERT(test::approx_eq(gx[0], expected, 1e-4f, 1e-4f),
                     "tanh_backward: gradient mismatch");
        std::printf("  PASS [tanh_backward]\n");

        loss = Tensor(); y = Tensor(); x = Tensor();
        cudaFree(dx);
    }

    // ------ sigmoid_backward ------
    {
        std::vector<float> hx = {1.0f};
        float* dx = test::to_device(hx);
        Tensor x = Tensor::from_ptr(dx, {1});
        x.set_requires_grad_(true);

        Tensor y = sigmoid(x);
        Tensor loss = sum(y);
        loss.backward();
        cudaDeviceSynchronize();

        float sig = 1.0f / (1.0f + std::exp(-1.0f));
        float expected = sig * (1.0f - sig);
        auto gx = test::to_host(x.grad().data<float>(), 1);
        FORGE_ASSERT(test::approx_eq(gx[0], expected, 1e-4f, 1e-4f),
                     "sigmoid_backward: gradient mismatch");
        std::printf("  PASS [sigmoid_backward]\n");

        loss = Tensor(); y = Tensor(); x = Tensor();
        cudaFree(dx);
    }

    // ------ div_backward ------
    {
        std::vector<float> ha = {6.0f};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {1});
        a.set_requires_grad_(true);

        std::vector<float> hb = {3.0f};
        float* db = test::to_device(hb);
        Tensor b = Tensor::from_ptr(db, {1});
        b.set_requires_grad_(true);

        Tensor c = div(a, b);
        Tensor loss = sum(c);
        loss.backward();
        cudaDeviceSynchronize();

        // d_a = 1/b = 1/3
        // d_b = -a/(b^2) = -6/9 = -2/3
        auto ga = test::to_host(a.grad().data<float>(), 1);
        auto gb = test::to_host(b.grad().data<float>(), 1);
        FORGE_ASSERT(test::approx_eq(ga[0], 1.0f/3.0f, 1e-4f, 1e-4f),
                     "div_backward: a.grad should be 1/3");
        FORGE_ASSERT(test::approx_eq(gb[0], -2.0f/3.0f, 1e-4f, 1e-4f),
                     "div_backward: b.grad should be -2/3");
        std::printf("  PASS [div_backward]\n");

        loss = Tensor(); c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ sqrt_backward ------
    {
        std::vector<float> hx = {4.0f};
        float* dx = test::to_device(hx);
        Tensor x = Tensor::from_ptr(dx, {1});
        x.set_requires_grad_(true);

        Tensor y = sqrt(x);
        Tensor loss = sum(y);
        loss.backward();
        cudaDeviceSynchronize();

        // d_x = 1 / (2 * sqrt(4)) = 1/4 = 0.25
        auto gx = test::to_host(x.grad().data<float>(), 1);
        FORGE_ASSERT(test::approx_eq(gx[0], 0.25f, 1e-5f, 1e-5f),
                     "sqrt_backward: gradient mismatch");
        std::printf("  PASS [sqrt_backward]\n");

        loss = Tensor(); y = Tensor(); x = Tensor();
        cudaFree(dx);
    }

    // ------ log_backward ------
    {
        std::vector<float> hx = {2.0f};
        float* dx = test::to_device(hx);
        Tensor x = Tensor::from_ptr(dx, {1});
        x.set_requires_grad_(true);

        Tensor y = log(x);
        Tensor loss = sum(y);
        loss.backward();
        cudaDeviceSynchronize();

        // d_x = 1/x = 0.5
        auto gx = test::to_host(x.grad().data<float>(), 1);
        FORGE_ASSERT(test::approx_eq(gx[0], 0.5f, 1e-5f, 1e-5f),
                     "log_backward: gradient mismatch");
        std::printf("  PASS [log_backward]\n");

        loss = Tensor(); y = Tensor(); x = Tensor();
        cudaFree(dx);
    }

    // ------ is_leaf ------
    {
        std::vector<float> hx = {1.0f};
        float* dx = test::to_device(hx);
        Tensor x = Tensor::from_ptr(dx, {1});
        x.set_requires_grad_(true);

        FORGE_ASSERT(x.is_leaf(), "is_leaf: x should be leaf");

        Tensor y = mul(x, 2.0f);
        FORGE_ASSERT(!y.is_leaf(), "is_leaf: y should not be leaf");

        std::printf("  PASS [is_leaf]\n");

        y = Tensor(); x = Tensor();
        cudaFree(dx);
    }

    // ------ vector_backward ------
    {
        // Multi-element: x = {1,2,3,4}; y = x^2; sum(y).backward() → x.grad = 2*x
        std::vector<float> hx = {1.0f, 2.0f, 3.0f, 4.0f};
        float* dx = test::to_device(hx);
        Tensor x = Tensor::from_ptr(dx, {4});
        x.set_requires_grad_(true);

        Tensor y = square(x);
        Tensor loss = sum(y);
        loss.backward();
        cudaDeviceSynchronize();

        auto gx = test::to_host(x.grad().data<float>(), 4);
        for (int i = 0; i < 4; i++) {
            float expected = 2.0f * hx[i];
            FORGE_ASSERT(test::approx_eq(gx[i], expected, 1e-4f, 1e-4f),
                         "vector_backward: gradient mismatch");
        }
        std::printf("  PASS [vector_backward]\n");

        loss = Tensor(); y = Tensor(); x = Tensor();
        cudaFree(dx);
    }

    std::printf("ALL PASSED\n");
    return 0;
}
