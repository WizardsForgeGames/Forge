// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Tests the full sparse INT8 GEMM pipeline:
//   prune_compress_24 -> pack_metadata -> sparse_int8_gemm
// at several shapes (ragged M, N_out not a multiple of the tile, K from one 64-deep step to
// 64 steps), for every tile config x split-K factor the dispatcher can run (forced), plus
// the automatic plan with and without a split-K workspace. The reference is an FP64 GEMM
// of the CPU-pruned dense weights (2:4, two largest magnitudes kept per group of 4).
// sparse_int8_linear (fp16 in): the fused quantize + GEMM path must match the two-pass path
// bit for bit, including at a large-M narrow-N shape where the automatic path fuses and the
// two-pass GEMM walks tiles N-fastest (int8 activation larger than L2).
#include <forge/gemm/int8_sparse.cuh>
#include <forge/quant/quantize.cuh>
#include <forge/sparsity/prune_compress.cuh>
#include <forge/sparsity/pack_metadata.cuh>
#include "test_helpers.cuh"
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

using namespace forge;

// 2:4 prune on the CPU: zero the two smallest-magnitude values of every group of 4
// (same tie-breaking as prune_compress_kernel).
static std::vector<int8_t> prune_24_dense(const std::vector<int8_t>& w, int N, int K) {
    std::vector<int8_t> p(w.size(), 0);
    for (int row = 0; row < N; row++) {
        for (int g = 0; g < K / 4; g++) {
            const int base = row * K + g * 4;
            int abs_v[4];
            for (int i = 0; i < 4; i++) abs_v[i] = std::abs((int)w[base + i]);
            int min1 = 0, min2 = 1;
            if (abs_v[0] > abs_v[1]) { min1 = 1; min2 = 0; }
            for (int i = 2; i < 4; i++) {
                if (abs_v[i] < abs_v[min1]) { min2 = min1; min1 = i; }
                else if (abs_v[i] < abs_v[min2]) { min2 = i; }
            }
            for (int i = 0; i < 4; i++)
                if (i != min1 && i != min2) p[base + i] = w[base + i];
        }
    }
    return p;
}

struct Shape { int M, K, N; };

static int run_shape(const Shape& sh, uint64_t seed) {
    const int M = sh.M, K = sh.K, N = sh.N;
    std::printf("  shape M=%d K=%d N=%d\n", M, K, N);

    std::vector<int8_t> h_weight((size_t)N * K), h_x((size_t)M * K);
    test::fill_random_int8(h_weight, seed);
    test::fill_random_int8(h_x, seed + 1);
    std::vector<float> h_act_scale(M), h_w_scale_f(N), h_bias_f(N);
    test::fill_random(h_act_scale, 0.001f, 0.05f, seed + 2);
    test::fill_random(h_w_scale_f, 0.001f, 0.05f, seed + 3);
    test::fill_random(h_bias_f, -0.5f, 0.5f, seed + 4);
    std::vector<__half> h_w_scale(N), h_bias(N);
    for (int i = 0; i < N; i++) {
        h_w_scale[i] = __float2half(h_w_scale_f[i]);
        h_bias[i]    = __float2half(h_bias_f[i]);
    }

    // FP64 reference of the pruned dense weights, without and with bias
    const std::vector<int8_t> w_pruned = prune_24_dense(h_weight, N, K);
    std::vector<double> ref_nb((size_t)M * N), ref_b((size_t)M * N);
    double ref_rms = 0.0;
    for (int m = 0; m < M; m++) {
        const int8_t* xr = &h_x[(size_t)m * K];
        for (int n = 0; n < N; n++) {
            const int8_t* wr = &w_pruned[(size_t)n * K];
            long long acc = 0;
            for (int k = 0; k < K; k++) acc += (int)wr[k] * (int)xr[k];
            const double v = (double)acc * h_act_scale[m] * (double)__half2float(h_w_scale[n]);
            ref_nb[(size_t)m * N + n] = v;
            ref_b[(size_t)m * N + n]  = v + (double)__half2float(h_bias[n]);
            ref_rms += v * v;
        }
    }
    ref_rms = std::sqrt(ref_rms / ((double)M * N));

    // GPU weight preparation
    int8_t*   d_weight = test::to_device(h_weight);
    int8_t*   d_comp   = test::to_device_zeroed<int8_t>((size_t)N * K / 2);
    uint8_t*  d_meta_raw    = test::to_device_zeroed<uint8_t>((size_t)N * K / 4);
    uint32_t* d_meta_packed = test::to_device_zeroed<uint32_t>((size_t)N * K / 32);
    sparsity::prune_compress_24(d_weight, d_comp, d_meta_raw, N, K);
    sparsity::pack_metadata(d_meta_raw, d_meta_packed, N, K);
    FORGE_CHECK(cudaDeviceSynchronize());

    int8_t* d_x         = test::to_device(h_x);
    float*  d_act_scale = test::to_device(h_act_scale);
    __half* d_w_scale   = test::to_device(h_w_scale);
    __half* d_bias      = test::to_device(h_bias);
    __half* d_output    = test::to_device_zeroed<__half>((size_t)M * N);
    float*  d_ws        = test::to_device_zeroed<float>((size_t)M * N);

    int failures = 0;
    auto check = [&](const char* label, int tile, int split, bool use_ws, bool use_bias) {
        // Garbage in the output and the workspace: the call must overwrite / zero them.
        FORGE_CHECK(cudaMemset(d_output, 0x7B, (size_t)M * N * sizeof(__half)));
        FORGE_CHECK(cudaMemset(d_ws, 0x5A, (size_t)M * N * sizeof(float)));
        gemm::sparse_int8_gemm(d_comp, d_x, d_meta_packed, d_act_scale, d_w_scale,
                               use_bias ? d_bias : nullptr, d_output, M, K, N,
                               use_ws ? d_ws : nullptr, split, nullptr, tile);
        FORGE_CHECK(cudaGetLastError());
        FORGE_CHECK(cudaDeviceSynchronize());
        const gemm::SparseGemmPlan plan = gemm::sparse_int8_gemm_plan(M, K, N, split, use_ws, tile);

        const auto out = test::to_host(d_output, (size_t)M * N);
        const std::vector<double>& ref = use_bias ? ref_b : ref_nb;
        double max_err = 0.0;
        int bad = 0;
        for (size_t i = 0; i < out.size(); i++) {
            const double g = (double)__half2float(out[i]);
            const double err = std::abs(g - ref[i]);
            // fp16 output rounding (2^-11 relative) plus fp32 dequant / atomics slack
            const double tol = 2e-3 * std::abs(ref[i]) + 1e-3 * ref_rms + 1e-4;
            if (!(err <= tol)) bad++;   // NaN counts as bad
            max_err = std::max(max_err, std::isfinite(err) ? err : 1e30);
        }
        const bool ok = (bad == 0);
        std::printf("    %-14s tile=%d split=%-2d ws=%d bias=%d -> plan(tile=%d, split=%d) "
                    "max_err=%.3e (rms %.3e) %s\n",
                    label, tile, split, use_ws ? 1 : 0, use_bias ? 1 : 0, plan.tile,
                    plan.split_k, max_err, ref_rms, ok ? "ok" : "FAIL");
        if (!ok) {
            std::fprintf(stderr, "FAIL: %d/%zu outputs out of tolerance\n", bad, out.size());
            failures++;
        }
        // Forced configs must run what was asked for.
        if (tile >= 0 && plan.tile != tile) { std::fprintf(stderr, "FAIL: tile not honored\n"); failures++; }
        if (split >= 1 && use_ws && plan.split_k != split) {
            std::fprintf(stderr, "FAIL: split not honored\n"); failures++;
        }
        if (!use_ws && plan.split_k != 1) { std::fprintf(stderr, "FAIL: split without workspace\n"); failures++; }
    };

    // Every tile x every split-K factor the cost model can pick (1, 2, 3, 4, 6), plus an
    // uneven split outside the candidate list (7) and one larger than the K-step count.
    const int k_iters = K / 64;
    for (int tile = 0; tile < gemm::kSparseNumTiles; tile++) {
        for (int split : {1, 2, 3, 4, 6, 7}) {
            check("forced", tile, split, true, (split % 2) == 1);
        }
        check("split>ksteps", tile, k_iters + 1, true, true);
    }
    // Automatic dispatch: with a workspace (tile + split chosen), without one (split 1),
    // default split_k = 1, and a requested split that is dropped for lack of a workspace.
    check("auto", gemm::kSparseTileAuto, 0, true, true);
    check("auto no-bias", gemm::kSparseTileAuto, 0, true, false);
    check("auto no-ws", gemm::kSparseTileAuto, 0, false, true);
    check("default split", gemm::kSparseTileAuto, 1, false, true);
    check("split no-ws", gemm::kSparseTileAuto, 4, false, true);

    test::free_device(d_weight);
    test::free_device(d_comp);
    test::free_device(d_meta_raw);
    test::free_device(d_meta_packed);
    test::free_device(d_x);
    test::free_device(d_act_scale);
    test::free_device(d_w_scale);
    test::free_device(d_bias);
    test::free_device(d_output);
    test::free_device(d_ws);
    return failures;
}

// sparse_int8_linear: fused vs two-pass (bit-identical), auto path, and a sampled FP64
// reference (rows checked against quantize -> pruned-weight GEMM on the CPU).
static int run_linear(const Shape& sh, uint64_t seed, bool expect_fused) {
    const int M = sh.M, K = sh.K, N = sh.N;
    const bool fused_auto = gemm::sparse_int8_linear_fused(M, K, N);
    std::printf("  linear M=%d K=%d N=%d (auto path: %s)\n", M, K, N, fused_auto ? "fused" : "two-pass");
    int failures = 0;
    if (fused_auto != expect_fused) {
        std::fprintf(stderr, "FAIL: auto path %d, expected %d\n", (int)fused_auto, (int)expect_fused);
        failures++;
    }

    std::vector<int8_t> h_weight((size_t)N * K);
    test::fill_random_int8(h_weight, seed);
    std::vector<float> h_xf((size_t)M * K), h_w_scale_f(N), h_bias_f(N);
    test::fill_random(h_xf, -2.0f, 2.0f, seed + 1);
    test::fill_random(h_w_scale_f, 0.001f, 0.05f, seed + 3);
    test::fill_random(h_bias_f, -0.5f, 0.5f, seed + 4);
    std::vector<__half> h_x((size_t)M * K), h_w_scale(N), h_bias(N);
    for (size_t i = 0; i < h_x.size(); i++) h_x[i] = __float2half(h_xf[i]);
    for (int i = 0; i < N; i++) {
        h_w_scale[i] = __float2half(h_w_scale_f[i]);
        h_bias[i]    = __float2half(h_bias_f[i]);
    }

    int8_t*   d_weight = test::to_device(h_weight);
    int8_t*   d_comp   = test::to_device_zeroed<int8_t>((size_t)N * K / 2);
    uint8_t*  d_meta_raw    = test::to_device_zeroed<uint8_t>((size_t)N * K / 4);
    uint32_t* d_meta_packed = test::to_device_zeroed<uint32_t>((size_t)N * K / 32);
    sparsity::prune_compress_24(d_weight, d_comp, d_meta_raw, N, K);
    sparsity::pack_metadata(d_meta_raw, d_meta_packed, N, K);
    __half* d_x       = test::to_device(h_x);
    __half* d_w_scale = test::to_device(h_w_scale);
    __half* d_bias    = test::to_device(h_bias);
    int8_t* d_xq      = test::to_device_zeroed<int8_t>((size_t)M * K);
    float*  d_xs      = test::to_device_zeroed<float>((size_t)M);
    __half* d_out_two = test::to_device_zeroed<__half>((size_t)M * N);
    __half* d_out_fq  = test::to_device_zeroed<__half>((size_t)M * N);
    __half* d_out_au  = test::to_device_zeroed<__half>((size_t)M * N);

    FORGE_CHECK(cudaMemset(d_out_fq, 0x7B, (size_t)M * N * sizeof(__half)));
    gemm::sparse_int8_linear(d_x, d_comp, d_meta_packed, d_w_scale, d_bias, d_out_two, M, K, N,
                             d_xq, d_xs, nullptr, 1, nullptr, gemm::kSparseLinearTwoPass);
    gemm::sparse_int8_linear(d_x, d_comp, d_meta_packed, d_w_scale, d_bias, d_out_fq, M, K, N,
                             nullptr, nullptr, nullptr, 1, nullptr, gemm::kSparseLinearFused);
    gemm::sparse_int8_linear(d_x, d_comp, d_meta_packed, d_w_scale, d_bias, d_out_au, M, K, N,
                             d_xq, d_xs, nullptr, 0, nullptr);
    FORGE_CHECK(cudaGetLastError());
    FORGE_CHECK(cudaDeviceSynchronize());
    const auto two = test::to_host(d_out_two, (size_t)M * N);
    const auto fq  = test::to_host(d_out_fq, (size_t)M * N);
    const auto au  = test::to_host(d_out_au, (size_t)M * N);
    size_t diff_fq = 0, diff_au = 0;
    for (size_t i = 0; i < two.size(); i++) {
        const uint16_t a = reinterpret_cast<const uint16_t&>(two[i]);
        diff_fq += (reinterpret_cast<const uint16_t&>(fq[i]) != a);
        diff_au += (reinterpret_cast<const uint16_t&>(au[i]) != a);
    }
    std::printf("    fused vs two-pass: %zu/%zu differ, auto vs two-pass: %zu differ\n",
                diff_fq, two.size(), diff_au);
    if (diff_fq || diff_au) { std::fprintf(stderr, "FAIL: paths not bit-identical\n"); failures++; }

    // Sampled FP64 reference: rows quantized on the CPU like the kernel (absmax / 127, round
    // to nearest even; the kernel's fast-math reciprocal can move a tie by one int8 step),
    // times the CPU-pruned weights. Error normalized by the sampled outputs' RMS.
    const std::vector<int8_t> w_pruned = prune_24_dense(h_weight, N, K);
    const auto xs = test::to_host(d_xs, (size_t)M);
    const int rows[] = {0, 1, M / 3, M / 2, M - 2, M - 1};
    double max_err = 0.0, ref_ss = 0.0;
    size_t n_ref = 0;
    for (int m : rows) {
        float amax = 0.f;
        for (int k = 0; k < K; k++) amax = std::max(amax, std::fabs(__half2float(h_x[(size_t)m * K + k])));
        const float scale = std::max(amax, 1e-8f) / 127.0f;
        if (std::abs(scale - xs[m]) > 1e-6f * scale) {
            std::fprintf(stderr, "FAIL: row %d scale %.8g vs %.8g\n", m, scale, xs[m]); failures++;
        }
        for (int n = 0; n < N; n++) {
            long long acc = 0;
            for (int k = 0; k < K; k++) {
                const float v = __half2float(h_x[(size_t)m * K + k]) * (1.0f / scale);
                const int q = std::max(-128, std::min(127, (int)std::nearbyint(v)));
                acc += (long long)q * w_pruned[(size_t)n * K + k];
            }
            const double ref = (double)acc * scale * __half2float(h_w_scale[n]) + __half2float(h_bias[n]);
            const double got = __half2float(fq[(size_t)m * N + n]);
            max_err = std::max(max_err, std::abs(got - ref));
            ref_ss += ref * ref;
            n_ref++;
        }
    }
    const double max_rel = max_err / std::sqrt(ref_ss / (double)n_ref);
    std::printf("    sampled rows vs FP64: max err / rms %.3e %s\n", max_rel, max_rel < 5e-3 ? "ok" : "FAIL");
    if (!(max_rel < 5e-3)) failures++;

    test::free_device(d_weight); test::free_device(d_comp); test::free_device(d_meta_raw);
    test::free_device(d_meta_packed); test::free_device(d_x); test::free_device(d_w_scale);
    test::free_device(d_bias); test::free_device(d_xq); test::free_device(d_xs);
    test::free_device(d_out_two); test::free_device(d_out_fq); test::free_device(d_out_au);
    return failures;
}

int main() {
    std::printf("=== Test: INT8 Sparse GEMM ===\n");
    // Kernel requirements: K % 64 == 0, N % 16 == 0. M is unconstrained.
    const Shape shapes[] = {
        {   1,   64,   16},   // single K step, single row/token
        {  32,  128,   64},   // original test shape
        {  37,  192,  208},   // 3 K steps (partial metadata chunk), ragged M and N
        { 100,  640,  272},   // 10 K steps: uneven split ranges cross metadata chunks
        { 300, 1280,  384},
        {1000, 1536,  512},
        {2050,  512,  400},   // M past several 128-token tiles, ragged
        { 129, 4096,   96},   // deep K, narrow N
    };
    int failures = 0;
    uint64_t seed = 42;
    for (const Shape& s : shapes) {
        failures += run_shape(s, seed);
        seed += 10;
    }

    // sparse_int8_linear: fused vs two-pass. The large shape (1280 -> 320, M = 70000: 269 MB
    // of fp16 activation) takes the fused path automatically on a 72 MB-L2 RTX 4090 and its
    // two-pass GEMM runs N-fastest; the small ones stay two-pass (ragged M / N, one K step).
    int l2 = 0;
    FORGE_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    failures += run_linear({70000, 1280, 320}, 900, 3.0 * 70000 * 1280 > 0.9 * l2);
    failures += run_linear({5000, 1024, 336}, 910, false);
    failures += run_linear({  37,  64,  16}, 920, false);
    failures += run_linear({4096, 1280, 5120}, 930, false);   // headline: never fused

    // The dispatcher's choices at representative shapes (RTX 4090 cost model).
    std::printf("  auto plans (with workspace):\n");
    const Shape plans[] = {{64, 1536, 6144}, {64, 5120, 1280}, {1024, 1536, 6144},
                           {1024, 6144, 1536}, {4096, 1536, 6144}, {8192, 5120, 1280},
                           {512, 1280, 768}, {256, 4096, 14336}};
    for (const Shape& s : plans) {
        const gemm::SparseGemmPlan p = gemm::sparse_int8_gemm_plan(s.M, s.K, s.N, 0, true);
        std::printf("    M=%-5d K=%-5d N=%-6d -> tile=%d split=%d\n", s.M, s.K, s.N, p.tile, p.split_k);
        FORGE_ASSERT(p.tile >= 0 && p.tile < gemm::kSparseNumTiles && p.split_k >= 1, "invalid plan");
    }

    FORGE_ASSERT(failures == 0, "sparse GEMM output mismatch");
    std::printf("  PASS [sparse_int8_gemm]\n");
    std::printf("ALL PASSED\n");
    return 0;
}
