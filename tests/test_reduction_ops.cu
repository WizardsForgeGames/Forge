// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Tests reduction operations: full-tensor, dimension-wise, statistical.
#include <forge/tensor.h>
#include <forge/tensor_ops.h>
#include <forge/reduction_ops.h>
#include "test_helpers.cuh"
#include <cmath>
#include <cfloat>
#include <numeric>
#include <algorithm>
#include <limits>

using namespace forge;

int main() {
    std::printf("=== Test: Reduction Ops ===\n");

    // ------ sum_full ------
    {
        const int N = 10007;
        std::vector<float> h(N);
        test::fill_random(h, -1.0f, 1.0f, 42);
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {N});

        Tensor out = forge::sum(t);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.numel() == 1, "sum_full: output should be scalar {1}");

        float cpu_sum = 0.0f;
        for (auto v : h) cpu_sum += v;

        auto g = test::to_host(out.data<float>(), 1);
        float diff = std::abs(g[0] - cpu_sum);
        FORGE_ASSERT(diff < 0.05f, "sum_full: data mismatch");
        std::printf("  PASS [sum_full] (diff=%.6e)\n", diff);

        out = Tensor(); t = Tensor(); cudaFree(d);
    }

    // ------ mean_full ------
    {
        const int N = 10007;
        std::vector<float> h(N);
        test::fill_random(h, -1.0f, 1.0f, 123);
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {N});

        Tensor out = forge::mean(t);
        cudaDeviceSynchronize();

        float cpu_sum = 0.0f;
        for (auto v : h) cpu_sum += v;
        float cpu_mean = cpu_sum / N;

        auto g = test::to_host(out.data<float>(), 1);
        float diff = std::abs(g[0] - cpu_mean);
        FORGE_ASSERT(diff < 1e-4f, "mean_full: data mismatch");
        std::printf("  PASS [mean_full] (diff=%.6e)\n", diff);

        out = Tensor(); t = Tensor(); cudaFree(d);
    }

    // ------ prod_full ------
    {
        const int N = 20;
        std::vector<float> h(N);
        // Values near 1 to avoid overflow
        test::fill_random(h, 0.9f, 1.1f, 77);
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {N});

        Tensor out = forge::prod(t);
        cudaDeviceSynchronize();

        float cpu_prod = 1.0f;
        for (auto v : h) cpu_prod *= v;

        auto g = test::to_host(out.data<float>(), 1);
        float diff = std::abs(g[0] - cpu_prod);
        FORGE_ASSERT(diff < 1e-3f, "prod_full: data mismatch");
        std::printf("  PASS [prod_full] (diff=%.6e)\n", diff);

        out = Tensor(); t = Tensor(); cudaFree(d);
    }

    // ------ amax / amin ------
    {
        const int N = 1000;
        std::vector<float> h(N);
        test::fill_random(h, -10.0f, 10.0f, 99);
        // Plant known extremes
        h[42] = 999.0f;
        h[777] = -888.0f;
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {N});

        Tensor mx = forge::amax(t);
        Tensor mn = forge::amin(t);
        cudaDeviceSynchronize();

        auto gmx = test::to_host(mx.data<float>(), 1);
        auto gmn = test::to_host(mn.data<float>(), 1);
        FORGE_ASSERT(gmx[0] == 999.0f, "amax: should be 999");
        FORGE_ASSERT(gmn[0] == -888.0f, "amin: should be -888");
        std::printf("  PASS [amax_amin_full]\n");

        mn = Tensor(); mx = Tensor(); t = Tensor(); cudaFree(d);
    }

    // ------ norm: L1, L2, Linf ------
    {
        std::vector<float> h = {3.0f, -4.0f, 0.0f, 5.0f, -12.0f};
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {5});

        // L1: sum of abs = 3+4+0+5+12 = 24
        Tensor n1 = forge::norm(t, 1.0f);
        cudaDeviceSynchronize();
        auto g1 = test::to_host(n1.data<float>(), 1);
        FORGE_ASSERT(std::abs(g1[0] - 24.0f) < 1e-4f, "norm_L1: mismatch");

        // L2: sqrt(9+16+0+25+144) = sqrt(194)
        Tensor n2 = forge::norm(t, 2.0f);
        cudaDeviceSynchronize();
        auto g2 = test::to_host(n2.data<float>(), 1);
        float expected_l2 = std::sqrt(194.0f);
        FORGE_ASSERT(std::abs(g2[0] - expected_l2) < 1e-3f, "norm_L2: mismatch");

        // Linf: max abs = 12
        Tensor ninf = forge::norm(t, std::numeric_limits<float>::infinity());
        cudaDeviceSynchronize();
        auto ginf = test::to_host(ninf.data<float>(), 1);
        FORGE_ASSERT(std::abs(ginf[0] - 12.0f) < 1e-4f, "norm_Linf: mismatch");

        std::printf("  PASS [norm_p1_p2_pinf]\n");
        ninf = Tensor(); n2 = Tensor(); n1 = Tensor();
        t = Tensor(); cudaFree(d);
    }

    // ------ sum dim=0: {20,100} -> {100} ------
    {
        const int D0 = 20, D1 = 100;
        std::vector<float> h(D0 * D1);
        test::fill_random(h, -1.0f, 1.0f, 55);
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {D0, D1});

        Tensor out = forge::sum(t, 0);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.ndim() == 1 && out.size(0) == D1,
                     "sum_dim0: shape should be {100}");

        auto g = test::to_host(out.data<float>(), D1);
        // CPU reference
        std::vector<float> ref(D1, 0.0f);
        for (int i = 0; i < D0; i++)
            for (int j = 0; j < D1; j++)
                ref[j] += h[i * D1 + j];

        float max_diff = 0.0f;
        for (int j = 0; j < D1; j++)
            max_diff = std::max(max_diff, std::abs(g[j] - ref[j]));

        FORGE_ASSERT(max_diff < 1e-3f, "sum_dim0: data mismatch");
        std::printf("  PASS [sum_dim0] (max_diff=%.6e)\n", max_diff);

        out = Tensor(); t = Tensor(); cudaFree(d);
    }

    // ------ sum dim=1: {20,100} -> {20} ------
    {
        const int D0 = 20, D1 = 100;
        std::vector<float> h(D0 * D1);
        test::fill_random(h, -1.0f, 1.0f, 66);
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {D0, D1});

        Tensor out = forge::sum(t, 1);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.ndim() == 1 && out.size(0) == D0,
                     "sum_dim1: shape should be {20}");

        auto g = test::to_host(out.data<float>(), D0);
        std::vector<float> ref(D0, 0.0f);
        for (int i = 0; i < D0; i++)
            for (int j = 0; j < D1; j++)
                ref[i] += h[i * D1 + j];

        float max_diff = 0.0f;
        for (int i = 0; i < D0; i++)
            max_diff = std::max(max_diff, std::abs(g[i] - ref[i]));

        FORGE_ASSERT(max_diff < 1e-3f, "sum_dim1: data mismatch");
        std::printf("  PASS [sum_dim1] (max_diff=%.6e)\n", max_diff);

        out = Tensor(); t = Tensor(); cudaFree(d);
    }

    // ------ sum middle dim: {5,8,10} dim=1 -> {5,10} ------
    {
        const int S0 = 5, S1 = 8, S2 = 10;
        std::vector<float> h(S0 * S1 * S2);
        test::fill_random(h, -1.0f, 1.0f, 88);
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {S0, S1, S2});

        Tensor out = forge::sum(t, 1);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.ndim() == 2 && out.size(0) == S0 && out.size(1) == S2,
                     "sum_middle: shape should be {5,10}");

        auto g = test::to_host(out.data<float>(), S0 * S2);
        float max_diff = 0.0f;
        for (int i = 0; i < S0; i++) {
            for (int k = 0; k < S2; k++) {
                float ref = 0.0f;
                for (int j = 0; j < S1; j++)
                    ref += h[i * S1 * S2 + j * S2 + k];
                float diff = std::abs(g[i * S2 + k] - ref);
                if (diff > max_diff) max_diff = diff;
            }
        }
        FORGE_ASSERT(max_diff < 1e-3f, "sum_middle: data mismatch");
        std::printf("  PASS [sum_middle_dim] (max_diff=%.6e)\n", max_diff);

        out = Tensor(); t = Tensor(); cudaFree(d);
    }

    // ------ mean dim ------
    {
        const int D0 = 10, D1 = 50;
        std::vector<float> h(D0 * D1);
        test::fill_random(h, -2.0f, 2.0f, 33);
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {D0, D1});

        Tensor out = forge::mean(t, 1);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.ndim() == 1 && out.size(0) == D0,
                     "mean_dim: shape should be {10}");

        auto g = test::to_host(out.data<float>(), D0);
        float max_diff = 0.0f;
        for (int i = 0; i < D0; i++) {
            float ref = 0.0f;
            for (int j = 0; j < D1; j++) ref += h[i * D1 + j];
            ref /= D1;
            float diff = std::abs(g[i] - ref);
            if (diff > max_diff) max_diff = diff;
        }
        FORGE_ASSERT(max_diff < 1e-4f, "mean_dim: data mismatch");
        std::printf("  PASS [mean_dim] (max_diff=%.6e)\n", max_diff);

        out = Tensor(); t = Tensor(); cudaFree(d);
    }

    // ------ keepdim=true ------
    {
        std::vector<float> h = {1,2,3, 4,5,6, 7,8,9};
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {3, 3});

        Tensor out = forge::sum(t, 1, /*keepdim=*/true);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.ndim() == 2 && out.size(0) == 3 && out.size(1) == 1,
                     "keepdim: shape should be {3,1}");

        auto g = test::to_host(out.data<float>(), 3);
        FORGE_ASSERT(std::abs(g[0] - 6.0f) < 1e-5f &&
                     std::abs(g[1] - 15.0f) < 1e-5f &&
                     std::abs(g[2] - 24.0f) < 1e-5f,
                     "keepdim: data mismatch");
        std::printf("  PASS [keepdim_true]\n");

        out = Tensor(); t = Tensor(); cudaFree(d);
    }

    // ------ keepdim=false ------
    {
        std::vector<float> h = {1,2,3, 4,5,6, 7,8,9};
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {3, 3});

        Tensor out = forge::sum(t, 0, /*keepdim=*/false);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.ndim() == 1 && out.size(0) == 3,
                     "keepdim_false: shape should be {3}");

        auto g = test::to_host(out.data<float>(), 3);
        // col sums: 1+4+7=12, 2+5+8=15, 3+6+9=18
        FORGE_ASSERT(std::abs(g[0] - 12.0f) < 1e-5f &&
                     std::abs(g[1] - 15.0f) < 1e-5f &&
                     std::abs(g[2] - 18.0f) < 1e-5f,
                     "keepdim_false: data mismatch");
        std::printf("  PASS [keepdim_false]\n");

        out = Tensor(); t = Tensor(); cudaFree(d);
    }

    // ------ max/min with indices ------
    {
        // {3,5} with planted maxes
        std::vector<float> h = {
            1, 9, 3, 4, 5,
            6, 2, 8, 7, 0,
            3, 4, 5, 6, 10
        };
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {3, 5});

        auto [vals, idxs] = forge::max(t, 1);
        cudaDeviceSynchronize();
        FORGE_ASSERT(vals.ndim() == 1 && vals.size(0) == 3,
                     "max_idx: vals shape should be {3}");
        FORGE_ASSERT(idxs.dtype() == DType::Int64,
                     "max_idx: indices should be Int64");

        auto gv = test::to_host(vals.data<float>(), 3);
        auto gi = test::to_host(idxs.data<int64_t>(), 3);
        // Row 0: max=9 at idx=1
        // Row 1: max=8 at idx=2
        // Row 2: max=10 at idx=4
        FORGE_ASSERT(gv[0] == 9.0f && gi[0] == 1, "max_idx: row 0 mismatch");
        FORGE_ASSERT(gv[1] == 8.0f && gi[1] == 2, "max_idx: row 1 mismatch");
        FORGE_ASSERT(gv[2] == 10.0f && gi[2] == 4, "max_idx: row 2 mismatch");

        auto [vmin, imin] = forge::min(t, 1);
        cudaDeviceSynchronize();
        auto gvmin = test::to_host(vmin.data<float>(), 3);
        auto gimin = test::to_host(imin.data<int64_t>(), 3);
        // Row 0: min=1 at idx=0
        // Row 1: min=0 at idx=4
        // Row 2: min=3 at idx=0
        FORGE_ASSERT(gvmin[0] == 1.0f && gimin[0] == 0, "min_idx: row 0 mismatch");
        FORGE_ASSERT(gvmin[1] == 0.0f && gimin[1] == 4, "min_idx: row 1 mismatch");
        FORGE_ASSERT(gvmin[2] == 3.0f && gimin[2] == 0, "min_idx: row 2 mismatch");

        std::printf("  PASS [max_min_with_indices]\n");
        imin = Tensor(); vmin = Tensor();
        idxs = Tensor(); vals = Tensor();
        t = Tensor(); cudaFree(d);
    }

    // ------ argmax / argmin ------
    {
        std::vector<float> h = {
            5, 1, 3,
            2, 8, 4,
            7, 6, 9
        };
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {3, 3});

        Tensor amax_idx = forge::argmax(t, 1);
        Tensor amin_idx = forge::argmin(t, 0);
        cudaDeviceSynchronize();

        FORGE_ASSERT(amax_idx.dtype() == DType::Int64, "argmax: should be Int64");
        auto gmax = test::to_host(amax_idx.data<int64_t>(), 3);
        // Row 0: max=5 at 0, Row 1: max=8 at 1, Row 2: max=9 at 2
        FORGE_ASSERT(gmax[0] == 0 && gmax[1] == 1 && gmax[2] == 2,
                     "argmax: index mismatch");

        auto gmin = test::to_host(amin_idx.data<int64_t>(), 3);
        // Col 0: min=2 at row 1, Col 1: min=1 at row 0, Col 2: min=3 at row 0
        FORGE_ASSERT(gmin[0] == 1 && gmin[1] == 0 && gmin[2] == 0,
                     "argmin: index mismatch");

        std::printf("  PASS [argmax_argmin]\n");
        amin_idx = Tensor(); amax_idx = Tensor();
        t = Tensor(); cudaFree(d);
    }

    // ------ var / std_dev ------
    {
        std::vector<float> h = {
            2.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f, 7.0f, 9.0f
        };
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {2, 4});

        // Row variance (dim=1, correction=1 = sample variance)
        Tensor v = forge::var(t, 1, false, 1);
        cudaDeviceSynchronize();

        auto gv = test::to_host(v.data<float>(), 2);
        // Row 0: [2,4,4,4] mean=3.5, var=(2.25+0.25+0.25+0.25)/3 = 3/3 = 1.0
        // Row 1: [5,5,7,9] mean=6.5, var=(2.25+2.25+0.25+6.25)/3 = 11/3 ≈ 3.667
        FORGE_ASSERT(std::abs(gv[0] - 1.0f) < 0.01f, "var: row 0 mismatch");
        FORGE_ASSERT(std::abs(gv[1] - 11.0f/3.0f) < 0.01f, "var: row 1 mismatch");

        // std_dev
        Tensor s = forge::std_dev(t, 1, false, 1);
        cudaDeviceSynchronize();
        auto gs = test::to_host(s.data<float>(), 2);
        FORGE_ASSERT(std::abs(gs[0] - 1.0f) < 0.01f, "std: row 0 mismatch");
        FORGE_ASSERT(std::abs(gs[1] - std::sqrt(11.0f/3.0f)) < 0.01f, "std: row 1 mismatch");

        std::printf("  PASS [var_std]\n");
        s = Tensor(); v = Tensor(); t = Tensor(); cudaFree(d);
    }

    // ------ logsumexp ------
    {
        // Use large values to test numerical stability
        std::vector<float> h = {
            1000.0f, 1000.1f, 1000.2f,
            -1000.0f, -999.9f, -999.8f
        };
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {2, 3});

        Tensor out = forge::logsumexp(t, 1);
        cudaDeviceSynchronize();
        auto g = test::to_host(out.data<float>(), 2);

        // Row 0: log(exp(1000)+exp(1000.1)+exp(1000.2))
        //       = 1000.2 + log(exp(-0.2)+exp(-0.1)+1)
        float ref0 = 1000.2f + std::log(std::exp(-0.2f) + std::exp(-0.1f) + 1.0f);
        float ref1 = -999.8f + std::log(std::exp(-0.2f) + std::exp(-0.1f) + 1.0f);

        FORGE_ASSERT(std::abs(g[0] - ref0) < 0.01f, "logsumexp: row 0 mismatch");
        FORGE_ASSERT(std::abs(g[1] - ref1) < 0.01f, "logsumexp: row 1 mismatch");
        std::printf("  PASS [logsumexp]\n");

        out = Tensor(); t = Tensor(); cudaFree(d);
    }

    // ------ negative dim ------
    {
        std::vector<float> h = {1,2,3, 4,5,6};
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {2, 3});

        Tensor out_neg = forge::sum(t, -1);
        Tensor out_pos = forge::sum(t, 1);
        cudaDeviceSynchronize();

        auto gn = test::to_host(out_neg.data<float>(), 2);
        auto gp = test::to_host(out_pos.data<float>(), 2);
        FORGE_ASSERT(gn[0] == gp[0] && gn[1] == gp[1],
                     "neg_dim: sum(t,-1) != sum(t,1)");
        std::printf("  PASS [negative_dim]\n");

        out_pos = Tensor(); out_neg = Tensor();
        t = Tensor(); cudaFree(d);
    }

    // ------ non-contiguous ------
    {
        // Transpose a {4,3} tensor to {3,4}, then reduce
        std::vector<float> h = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {4, 3});
        Tensor tt = t.transpose(0, 1); // {3,4}, non-contiguous
        FORGE_ASSERT(!tt.is_contiguous(), "non_contig: should be non-contiguous");

        Tensor out = forge::sum(tt, 1); // reduce last dim -> {3}
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.ndim() == 1 && out.size(0) == 3,
                     "non_contig: shape should be {3}");

        auto g = test::to_host(out.data<float>(), 3);
        // Transposed: col0=[1,4,7,10] col1=[2,5,8,11] col2=[3,6,9,12]
        // After transpose rows become: [1,4,7,10], [2,5,8,11], [3,6,9,12]
        // Sum dim=1: 1+4+7+10=22, 2+5+8+11=26, 3+6+9+12=30
        FORGE_ASSERT(std::abs(g[0] - 22.0f) < 1e-5f &&
                     std::abs(g[1] - 26.0f) < 1e-5f &&
                     std::abs(g[2] - 30.0f) < 1e-5f,
                     "non_contig: data mismatch");
        std::printf("  PASS [non_contiguous]\n");

        out = Tensor(); tt = Tensor(); t = Tensor(); cudaFree(d);
    }

    // ------ zero-element ------
    {
        Tensor t = Tensor::empty({0});
        Tensor out = forge::sum(t);
        cudaDeviceSynchronize();
        auto g = test::to_host(out.data<float>(), 1);
        FORGE_ASSERT(g[0] == 0.0f, "zero_elem: sum should be 0");
        std::printf("  PASS [zero_element]\n");
    }

    // ------ shape verification: {3,4,5} all dims ------
    {
        const int S0 = 3, S1 = 4, S2 = 5;
        std::vector<float> h(S0 * S1 * S2);
        test::fill_random(h, -1.0f, 1.0f, 111);
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {S0, S1, S2});

        // dim=0, keepdim=false -> {4,5}
        Tensor r0 = forge::sum(t, 0, false);
        FORGE_ASSERT(r0.ndim() == 2 && r0.size(0) == S1 && r0.size(1) == S2,
                     "shape: dim0 no keepdim should be {4,5}");

        // dim=0, keepdim=true -> {1,4,5}
        Tensor r0k = forge::sum(t, 0, true);
        FORGE_ASSERT(r0k.ndim() == 3 && r0k.size(0) == 1 && r0k.size(1) == S1 && r0k.size(2) == S2,
                     "shape: dim0 keepdim should be {1,4,5}");

        // dim=1, keepdim=false -> {3,5}
        Tensor r1 = forge::sum(t, 1, false);
        FORGE_ASSERT(r1.ndim() == 2 && r1.size(0) == S0 && r1.size(1) == S2,
                     "shape: dim1 no keepdim should be {3,5}");

        // dim=2, keepdim=false -> {3,4}
        Tensor r2 = forge::sum(t, 2, false);
        FORGE_ASSERT(r2.ndim() == 2 && r2.size(0) == S0 && r2.size(1) == S1,
                     "shape: dim2 no keepdim should be {3,4}");

        // dim=2, keepdim=true -> {3,4,1}
        Tensor r2k = forge::sum(t, 2, true);
        FORGE_ASSERT(r2k.ndim() == 3 && r2k.size(0) == S0 && r2k.size(1) == S1 && r2k.size(2) == 1,
                     "shape: dim2 keepdim should be {3,4,1}");

        std::printf("  PASS [shape_verification]\n");
        r2k = Tensor(); r2 = Tensor(); r1 = Tensor();
        r0k = Tensor(); r0 = Tensor(); t = Tensor(); cudaFree(d);
    }

    // ------ prod_dim ------
    {
        std::vector<float> h = {1,2,3, 4,5,6};
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {2, 3});

        Tensor out = forge::prod(t, 1);
        cudaDeviceSynchronize();
        auto g = test::to_host(out.data<float>(), 2);
        // Row 0: 1*2*3=6, Row 1: 4*5*6=120
        FORGE_ASSERT(std::abs(g[0] - 6.0f) < 1e-5f &&
                     std::abs(g[1] - 120.0f) < 1e-5f,
                     "prod_dim: data mismatch");
        std::printf("  PASS [prod_dim]\n");

        out = Tensor(); t = Tensor(); cudaFree(d);
    }

    // ------ amax/amin dim ------
    {
        std::vector<float> h = {5,1,3, 2,8,4, 7,6,9};
        float* d = test::to_device(h);
        Tensor t = Tensor::from_ptr(d, {3, 3});

        Tensor mx = forge::amax(t, 1);
        Tensor mn = forge::amin(t, 0);
        cudaDeviceSynchronize();

        auto gmx = test::to_host(mx.data<float>(), 3);
        // Row maxes: 5, 8, 9
        FORGE_ASSERT(gmx[0] == 5.0f && gmx[1] == 8.0f && gmx[2] == 9.0f,
                     "amax_dim: data mismatch");

        auto gmn = test::to_host(mn.data<float>(), 3);
        // Col mins: min(5,2,7)=2, min(1,8,6)=1, min(3,4,9)=3
        FORGE_ASSERT(gmn[0] == 2.0f && gmn[1] == 1.0f && gmn[2] == 3.0f,
                     "amin_dim: data mismatch");
        std::printf("  PASS [amax_amin_dim]\n");

        mn = Tensor(); mx = Tensor(); t = Tensor(); cudaFree(d);
    }

    std::printf("ALL PASSED\n");
    return 0;
}
