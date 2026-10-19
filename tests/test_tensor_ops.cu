// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Tests all elementwise math operations: arithmetic, unary, broadcasting,
// non-contiguous tensors, in-place ops, and operator overloads.
#include <forge/tensor.h>
#include <forge/tensor_ops.h>
#include "test_helpers.cuh"
#include <cmath>

using namespace forge;

int main() {
    std::printf("=== Test: Tensor Elementwise Ops ===\n");

    // ------ add / sub / mul / div: contiguous ------
    {
        const int N = 100;
        std::vector<float> ha(N), hb(N);
        for (int i = 0; i < N; i++) {
            ha[i] = static_cast<float>(i);
            hb[i] = static_cast<float>(i * 2 + 1);
        }

        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {N});
        Tensor b = Tensor::from_ptr(db, {N});

        // add
        Tensor c = forge::add(a, b);
        cudaDeviceSynchronize();
        auto gc = test::to_host(c.data<float>(), N);
        std::vector<float> ref_add(N);
        for (int i = 0; i < N; i++) ref_add[i] = ha[i] + hb[i];
        FORGE_ASSERT(test::check_arrays(gc, ref_add, 0.0f, 0.0f, "add"),
                     "add: data mismatch");

        // sub
        c = forge::sub(a, b);
        cudaDeviceSynchronize();
        gc = test::to_host(c.data<float>(), N);
        std::vector<float> ref_sub(N);
        for (int i = 0; i < N; i++) ref_sub[i] = ha[i] - hb[i];
        FORGE_ASSERT(test::check_arrays(gc, ref_sub, 0.0f, 0.0f, "sub"),
                     "sub: data mismatch");

        // mul
        c = forge::mul(a, b);
        cudaDeviceSynchronize();
        gc = test::to_host(c.data<float>(), N);
        std::vector<float> ref_mul(N);
        for (int i = 0; i < N; i++) ref_mul[i] = ha[i] * hb[i];
        FORGE_ASSERT(test::check_arrays(gc, ref_mul, 0.0f, 0.0f, "mul"),
                     "mul: data mismatch");

        // div
        c = forge::div(a, b);
        cudaDeviceSynchronize();
        gc = test::to_host(c.data<float>(), N);
        std::vector<float> ref_div(N);
        for (int i = 0; i < N; i++) ref_div[i] = ha[i] / hb[i];
        FORGE_ASSERT(test::check_arrays(gc, ref_div, 1e-6f, 1e-5f, "div"),
                     "div: data mismatch");

        c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ tensor-scalar arithmetic ------
    {
        const int N = 50;
        std::vector<float> ha(N);
        for (int i = 0; i < N; i++) ha[i] = static_cast<float>(i + 1);

        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {N});

        Tensor c = forge::add(a, 10.0f);
        cudaDeviceSynchronize();
        auto gc = test::to_host(c.data<float>(), N);
        std::vector<float> ref(N);
        for (int i = 0; i < N; i++) ref[i] = ha[i] + 10.0f;
        FORGE_ASSERT(test::check_arrays(gc, ref, 0.0f, 0.0f, "add_scalar"),
                     "add_scalar: mismatch");

        c = forge::mul(a, 0.5f);
        cudaDeviceSynchronize();
        gc = test::to_host(c.data<float>(), N);
        for (int i = 0; i < N; i++) ref[i] = ha[i] * 0.5f;
        FORGE_ASSERT(test::check_arrays(gc, ref, 0.0f, 0.0f, "mul_scalar"),
                     "mul_scalar: mismatch");

        c = forge::sub(a, 5.0f);
        cudaDeviceSynchronize();
        gc = test::to_host(c.data<float>(), N);
        for (int i = 0; i < N; i++) ref[i] = ha[i] - 5.0f;
        FORGE_ASSERT(test::check_arrays(gc, ref, 0.0f, 0.0f, "sub_scalar"),
                     "sub_scalar: mismatch");

        c = forge::div(a, 3.0f);
        cudaDeviceSynchronize();
        gc = test::to_host(c.data<float>(), N);
        for (int i = 0; i < N; i++) ref[i] = ha[i] / 3.0f;
        FORGE_ASSERT(test::check_arrays(gc, ref, 1e-6f, 1e-5f, "div_scalar"),
                     "div_scalar: mismatch");

        c = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ unary ops ------
    {
        const int N = 64;
        std::vector<float> ha(N);
        test::fill_random(ha, 0.1f, 5.0f, 42); // positive values for log/sqrt

        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {N});

        // neg
        {
            Tensor c = forge::neg(a);
            cudaDeviceSynchronize();
            auto gc = test::to_host(c.data<float>(), N);
            std::vector<float> ref(N);
            for (int i = 0; i < N; i++) ref[i] = -ha[i];
            FORGE_ASSERT(test::check_arrays(gc, ref, 0.0f, 0.0f, "neg"), "neg: mismatch");
        }

        // abs (test with mixed signs)
        {
            std::vector<float> ha_mix(N);
            for (int i = 0; i < N; i++) ha_mix[i] = (i % 2 == 0) ? ha[i] : -ha[i];
            float* da_mix = test::to_device(ha_mix);
            Tensor a_mix = Tensor::from_ptr(da_mix, {N});

            Tensor c = forge::abs(a_mix);
            cudaDeviceSynchronize();
            auto gc = test::to_host(c.data<float>(), N);
            std::vector<float> ref(N);
            for (int i = 0; i < N; i++) ref[i] = std::abs(ha_mix[i]);
            FORGE_ASSERT(test::check_arrays(gc, ref, 0.0f, 0.0f, "abs"), "abs: mismatch");

            c = Tensor(); a_mix = Tensor();
            cudaFree(da_mix);
        }

        // sqrt
        {
            Tensor c = forge::sqrt(a);
            cudaDeviceSynchronize();
            auto gc = test::to_host(c.data<float>(), N);
            std::vector<float> ref(N);
            for (int i = 0; i < N; i++) ref[i] = std::sqrt(ha[i]);
            FORGE_ASSERT(test::check_arrays(gc, ref, 1e-5f, 1e-4f, "sqrt"), "sqrt: mismatch");
        }

        // rsqrt
        {
            Tensor c = forge::rsqrt(a);
            cudaDeviceSynchronize();
            auto gc = test::to_host(c.data<float>(), N);
            std::vector<float> ref(N);
            for (int i = 0; i < N; i++) ref[i] = 1.0f / std::sqrt(ha[i]);
            FORGE_ASSERT(test::check_arrays(gc, ref, 1e-5f, 1e-4f, "rsqrt"), "rsqrt: mismatch");
        }

        // exp
        {
            // Use smaller values to avoid overflow
            std::vector<float> hx(N);
            for (int i = 0; i < N; i++) hx[i] = ha[i] * 0.5f - 1.0f;
            float* dx = test::to_device(hx);
            Tensor x = Tensor::from_ptr(dx, {N});

            Tensor c = forge::exp(x);
            cudaDeviceSynchronize();
            auto gc = test::to_host(c.data<float>(), N);
            std::vector<float> ref(N);
            for (int i = 0; i < N; i++) ref[i] = std::exp(hx[i]);
            FORGE_ASSERT(test::check_arrays(gc, ref, 1e-4f, 1e-4f, "exp"), "exp: mismatch");

            c = Tensor(); x = Tensor();
            cudaFree(dx);
        }

        // log
        {
            Tensor c = forge::log(a);
            cudaDeviceSynchronize();
            auto gc = test::to_host(c.data<float>(), N);
            std::vector<float> ref(N);
            for (int i = 0; i < N; i++) ref[i] = std::log(ha[i]);
            FORGE_ASSERT(test::check_arrays(gc, ref, 1e-5f, 1e-4f, "log"), "log: mismatch");
        }

        // square
        {
            Tensor c = forge::square(a);
            cudaDeviceSynchronize();
            auto gc = test::to_host(c.data<float>(), N);
            std::vector<float> ref(N);
            for (int i = 0; i < N; i++) ref[i] = ha[i] * ha[i];
            FORGE_ASSERT(test::check_arrays(gc, ref, 1e-4f, 1e-4f, "square"), "square: mismatch");
        }

        // reciprocal
        {
            Tensor c = forge::reciprocal(a);
            cudaDeviceSynchronize();
            auto gc = test::to_host(c.data<float>(), N);
            std::vector<float> ref(N);
            for (int i = 0; i < N; i++) ref[i] = 1.0f / ha[i];
            FORGE_ASSERT(test::check_arrays(gc, ref, 1e-5f, 1e-4f, "reciprocal"), "reciprocal: mismatch");
        }

        // sign
        {
            std::vector<float> hs = {-3.0f, -0.5f, 0.0f, 0.5f, 3.0f};
            float* ds = test::to_device(hs);
            Tensor s = Tensor::from_ptr(ds, {5});
            Tensor c = forge::sign(s);
            cudaDeviceSynchronize();
            auto gc = test::to_host(c.data<float>(), 5);
            std::vector<float> ref = {-1.0f, -1.0f, 0.0f, 1.0f, 1.0f};
            FORGE_ASSERT(test::check_arrays(gc, ref, 0.0f, 0.0f, "sign"), "sign: mismatch");
            c = Tensor(); s = Tensor();
            cudaFree(ds);
        }

        // sin / cos
        {
            Tensor c = forge::sin(a);
            cudaDeviceSynchronize();
            auto gc = test::to_host(c.data<float>(), N);
            std::vector<float> ref(N);
            for (int i = 0; i < N; i++) ref[i] = std::sin(ha[i]);
            FORGE_ASSERT(test::check_arrays(gc, ref, 1e-5f, 1e-4f, "sin"), "sin: mismatch");

            c = forge::cos(a);
            cudaDeviceSynchronize();
            gc = test::to_host(c.data<float>(), N);
            for (int i = 0; i < N; i++) ref[i] = std::cos(ha[i]);
            FORGE_ASSERT(test::check_arrays(gc, ref, 1e-5f, 1e-4f, "cos"), "cos: mismatch");
        }

        // tanh
        {
            Tensor c = forge::tanh(a);
            cudaDeviceSynchronize();
            auto gc = test::to_host(c.data<float>(), N);
            std::vector<float> ref(N);
            for (int i = 0; i < N; i++) ref[i] = std::tanh(ha[i]);
            FORGE_ASSERT(test::check_arrays(gc, ref, 1e-5f, 1e-4f, "tanh"), "tanh: mismatch");
        }

        // sigmoid
        {
            Tensor c = forge::sigmoid(a);
            cudaDeviceSynchronize();
            auto gc = test::to_host(c.data<float>(), N);
            std::vector<float> ref(N);
            for (int i = 0; i < N; i++) ref[i] = 1.0f / (1.0f + std::exp(-ha[i]));
            FORGE_ASSERT(test::check_arrays(gc, ref, 1e-5f, 1e-4f, "sigmoid"), "sigmoid: mismatch");
        }

        a = Tensor();
        cudaFree(da);
    }

    // ------ clamp / pow ------
    {
        std::vector<float> ha = {-5.0f, -1.0f, 0.0f, 1.5f, 3.0f, 10.0f};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {6});

        Tensor c = forge::clamp(a, -1.0f, 3.0f);
        cudaDeviceSynchronize();
        auto gc = test::to_host(c.data<float>(), 6);
        std::vector<float> ref = {-1.0f, -1.0f, 0.0f, 1.5f, 3.0f, 3.0f};
        FORGE_ASSERT(test::check_arrays(gc, ref, 0.0f, 0.0f, "clamp"), "clamp: mismatch");

        // pow
        std::vector<float> hp = {1.0f, 2.0f, 3.0f, 4.0f};
        float* dp = test::to_device(hp);
        Tensor p = Tensor::from_ptr(dp, {4});
        c = forge::pow(p, 3.0f);
        cudaDeviceSynchronize();
        gc = test::to_host(c.data<float>(), 4);
        std::vector<float> ref_pow = {1.0f, 8.0f, 27.0f, 64.0f};
        FORGE_ASSERT(test::check_arrays(gc, ref_pow, 1e-4f, 1e-4f, "pow"), "pow: mismatch");

        c = Tensor(); p = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(dp);
    }

    // ------ Broadcasting: {3,1} + {1,4} -> {3,4} ------
    {
        std::vector<float> ha = {10.0f, 20.0f, 30.0f};
        std::vector<float> hb = {1.0f, 2.0f, 3.0f, 4.0f};
        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {3, 1});
        Tensor b = Tensor::from_ptr(db, {1, 4});

        Tensor c = forge::add(a, b);
        cudaDeviceSynchronize();
        FORGE_ASSERT(c.ndim() == 2, "broadcast: ndim should be 2");
        FORGE_ASSERT(c.size(0) == 3, "broadcast: size(0) should be 3");
        FORGE_ASSERT(c.size(1) == 4, "broadcast: size(1) should be 4");

        auto gc = test::to_host(c.data<float>(), 12);
        std::vector<float> ref = {
            11, 12, 13, 14,
            21, 22, 23, 24,
            31, 32, 33, 34
        };
        FORGE_ASSERT(test::check_arrays(gc, ref, 0.0f, 0.0f, "broadcast_3x1_1x4"),
                     "broadcast_3x1_1x4: mismatch");

        c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ Broadcasting: {5} + {3,5} -> {3,5} ------
    {
        std::vector<float> ha = {1, 2, 3, 4, 5};
        std::vector<float> hb(15);
        for (int i = 0; i < 15; i++) hb[i] = static_cast<float>(i * 10);

        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {5});
        Tensor b = Tensor::from_ptr(db, {3, 5});

        Tensor c = forge::add(a, b);
        cudaDeviceSynchronize();
        FORGE_ASSERT(c.ndim() == 2, "broadcast_1d_2d: ndim should be 2");
        FORGE_ASSERT(c.size(0) == 3, "broadcast_1d_2d: size(0) should be 3");
        FORGE_ASSERT(c.size(1) == 5, "broadcast_1d_2d: size(1) should be 5");

        auto gc = test::to_host(c.data<float>(), 15);
        std::vector<float> ref(15);
        for (int r = 0; r < 3; r++)
            for (int col = 0; col < 5; col++)
                ref[r * 5 + col] = ha[col] + hb[r * 5 + col];
        FORGE_ASSERT(test::check_arrays(gc, ref, 0.0f, 0.0f, "broadcast_1d_2d"),
                     "broadcast_1d_2d: mismatch");

        c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ Broadcasting: {1} + {100} -> {100} (scalar-like) ------
    {
        std::vector<float> ha = {42.0f};
        std::vector<float> hb(100);
        for (int i = 0; i < 100; i++) hb[i] = static_cast<float>(i);

        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {1});
        Tensor b = Tensor::from_ptr(db, {100});

        Tensor c = forge::add(a, b);
        cudaDeviceSynchronize();
        FORGE_ASSERT(c.size(0) == 100, "broadcast_scalar: size should be 100");

        auto gc = test::to_host(c.data<float>(), 100);
        std::vector<float> ref(100);
        for (int i = 0; i < 100; i++) ref[i] = 42.0f + hb[i];
        FORGE_ASSERT(test::check_arrays(gc, ref, 0.0f, 0.0f, "broadcast_scalar"),
                     "broadcast_scalar: mismatch");

        c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ Non-contiguous: transposed + contiguous ------
    {
        const int R = 4, C = 3;
        std::vector<float> ha(R * C), hb(R * C);
        for (int i = 0; i < R * C; i++) {
            ha[i] = static_cast<float>(i);
            hb[i] = static_cast<float>(i * 10);
        }

        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {R, C});
        Tensor b = Tensor::from_ptr(db, {C, R});

        // Transpose a to {C, R}, then add with b (both {C,R})
        Tensor at = a.transpose(0, 1); // {C, R} non-contiguous
        FORGE_ASSERT(!at.is_contiguous(), "non_contig: transposed should be non-contiguous");

        Tensor c = forge::add(at, b);
        cudaDeviceSynchronize();
        FORGE_ASSERT(c.size(0) == C, "non_contig: size(0) should be C");
        FORGE_ASSERT(c.size(1) == R, "non_contig: size(1) should be R");

        auto gc = test::to_host(c.data<float>(), C * R);
        // CPU reference: at[c][r] = a[r][c] = r*C+c, b[c][r] = c*R+r
        std::vector<float> ref(C * R);
        for (int ci = 0; ci < C; ci++)
            for (int ri = 0; ri < R; ri++)
                ref[ci * R + ri] = static_cast<float>(ri * C + ci) +
                                   static_cast<float>(ci * R + ri) * 10.0f;
        FORGE_ASSERT(test::check_arrays(gc, ref, 0.0f, 0.0f, "non_contig_add"),
                     "non_contig_add: mismatch");

        c = Tensor(); at = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ Unary on non-contiguous ------
    {
        const int R = 3, C = 4;
        std::vector<float> ha(R * C);
        for (int i = 0; i < R * C; i++) ha[i] = static_cast<float>(i + 1);

        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {R, C});
        Tensor at = a.transpose(0, 1); // {C, R} non-contiguous

        Tensor c = forge::neg(at);
        cudaDeviceSynchronize();
        FORGE_ASSERT(c.is_contiguous(), "unary_noncontig: result should be contiguous");
        FORGE_ASSERT(c.size(0) == C, "unary_noncontig: size(0)");
        FORGE_ASSERT(c.size(1) == R, "unary_noncontig: size(1)");

        auto gc = test::to_host(c.data<float>(), C * R);
        // at[c][r] = a[r][c] = r*C+c+1
        std::vector<float> ref(C * R);
        for (int ci = 0; ci < C; ci++)
            for (int ri = 0; ri < R; ri++)
                ref[ci * R + ri] = -static_cast<float>(ri * C + ci + 1);
        FORGE_ASSERT(test::check_arrays(gc, ref, 0.0f, 0.0f, "unary_noncontig"),
                     "unary_noncontig: mismatch");

        c = Tensor(); at = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ In-place: add_, mul_, sub_, div_ ------
    {
        Tensor a = Tensor::ones({10});
        cudaDeviceSynchronize();
        uint32_t v0 = a.version();

        a.add_(5.0f);
        cudaDeviceSynchronize();
        FORGE_ASSERT(a.version() == v0 + 1, "inplace_add: version should bump");
        auto ga = test::to_host(a.data<float>(), 10);
        for (int i = 0; i < 10; i++)
            FORGE_ASSERT(ga[i] == 6.0f, "inplace_add: all should be 6.0");

        a.mul_(2.0f);
        cudaDeviceSynchronize();
        FORGE_ASSERT(a.version() == v0 + 2, "inplace_mul: version should bump again");
        ga = test::to_host(a.data<float>(), 10);
        for (int i = 0; i < 10; i++)
            FORGE_ASSERT(ga[i] == 12.0f, "inplace_mul: all should be 12.0");

        Tensor b = Tensor::ones({10});
        b.fill_(3.0f);
        cudaDeviceSynchronize();
        a.sub_(b);
        cudaDeviceSynchronize();
        ga = test::to_host(a.data<float>(), 10);
        for (int i = 0; i < 10; i++)
            FORGE_ASSERT(ga[i] == 9.0f, "inplace_sub_tensor: all should be 9.0");

        a.div_(3.0f);
        cudaDeviceSynchronize();
        ga = test::to_host(a.data<float>(), 10);
        for (int i = 0; i < 10; i++)
            FORGE_ASSERT(ga[i] == 3.0f, "inplace_div: all should be 3.0");

        a.neg_();
        cudaDeviceSynchronize();
        ga = test::to_host(a.data<float>(), 10);
        for (int i = 0; i < 10; i++)
            FORGE_ASSERT(ga[i] == -3.0f, "inplace_neg: all should be -3.0");

        std::printf("  PASS [in_place_ops]\n");
    }

    // ------ In-place with broadcast: {3,4}.add_({4}) ------
    {
        Tensor a = Tensor::ones({3, 4});
        cudaDeviceSynchronize();

        std::vector<float> hb = {10, 20, 30, 40};
        float* db = test::to_device(hb);
        Tensor b = Tensor::from_ptr(db, {4});

        a.add_(b);
        cudaDeviceSynchronize();
        auto ga = test::to_host(a.data<float>(), 12);
        std::vector<float> ref = {
            11, 21, 31, 41,
            11, 21, 31, 41,
            11, 21, 31, 41
        };
        FORGE_ASSERT(test::check_arrays(ga, ref, 0.0f, 0.0f, "inplace_broadcast"),
                     "inplace_broadcast: mismatch");

        b = Tensor(); a = Tensor();
        cudaFree(db);
    }

    // ------ Operator overloads ------
    {
        Tensor a = Tensor::ones({5});
        a.fill_(3.0f);
        Tensor b = Tensor::ones({5});
        b.fill_(2.0f);
        cudaDeviceSynchronize();

        // a + b
        Tensor c = a + b;
        cudaDeviceSynchronize();
        auto gc = test::to_host(c.data<float>(), 5);
        for (int i = 0; i < 5; i++)
            FORGE_ASSERT(gc[i] == 5.0f, "op+: should be 5.0");

        // a * scalar
        c = a * 4.0f;
        cudaDeviceSynchronize();
        gc = test::to_host(c.data<float>(), 5);
        for (int i = 0; i < 5; i++)
            FORGE_ASSERT(gc[i] == 12.0f, "op*s: should be 12.0");

        // scalar * a
        c = 2.0f * a;
        cudaDeviceSynchronize();
        gc = test::to_host(c.data<float>(), 5);
        for (int i = 0; i < 5; i++)
            FORGE_ASSERT(gc[i] == 6.0f, "s*op: should be 6.0");

        // unary minus
        c = -a;
        cudaDeviceSynchronize();
        gc = test::to_host(c.data<float>(), 5);
        for (int i = 0; i < 5; i++)
            FORGE_ASSERT(gc[i] == -3.0f, "op-unary: should be -3.0");

        // a - scalar
        c = a - 1.0f;
        cudaDeviceSynchronize();
        gc = test::to_host(c.data<float>(), 5);
        for (int i = 0; i < 5; i++)
            FORGE_ASSERT(gc[i] == 2.0f, "op-s: should be 2.0");

        // scalar - a  (= -a + scalar)
        c = 10.0f - a;
        cudaDeviceSynchronize();
        gc = test::to_host(c.data<float>(), 5);
        for (int i = 0; i < 5; i++)
            FORGE_ASSERT(gc[i] == 7.0f, "s-op: should be 7.0");

        std::printf("  PASS [operator_overloads]\n");
    }

    // ------ Zero-element tensor ------
    {
        Tensor a = Tensor::empty({0});
        Tensor b = Tensor::empty({0});
        Tensor c = forge::add(a, b);
        FORGE_ASSERT(c.numel() == 0, "zero_elem: numel should be 0");
        FORGE_ASSERT(c.defined(), "zero_elem: should be defined");

        c = forge::neg(a);
        FORGE_ASSERT(c.numel() == 0, "zero_elem_unary: numel should be 0");

        std::printf("  PASS [zero_element]\n");
    }

    // ------ Scalar tensor {1} ------
    {
        std::vector<float> ha = {7.0f};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {1});

        Tensor c = forge::square(a);
        cudaDeviceSynchronize();
        auto gc = test::to_host(c.data<float>(), 1);
        FORGE_ASSERT(gc[0] == 49.0f, "scalar_tensor: 7^2 should be 49");

        c = Tensor(); a = Tensor();
        cudaFree(da);
        std::printf("  PASS [scalar_tensor]\n");
    }

    std::printf("ALL PASSED\n");
    return 0;
}
