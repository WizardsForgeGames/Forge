// Forge GPU Compute Library -- Benchmark
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Sparse INT8 GEMM (2:4, mma.sp) on pre-quantized inputs, plus the activation quantize
// kernel, at transformer MLP shapes. GPU time only:
//   each call is captured n times into a CUDA graph (n chosen so one replay is >= 0.2 ms),
//   100 warmup replays, then 300 replays each bracketed by CUDA events and enqueued
//   back-to-back; per-call time = median(replay) / n. Inputs stay L2-resident across replays.
// Same protocol as sparsemma tests/bench_kernels.py, so the numbers are comparable.
//
// The GEMM runs with split_k = 0 (automatic) and an fp32 split-K workspace. The source also
// compiles against the pre-cost-model API (FORGE_SPARSE_GEMM_API_VERSION undefined) for
// old-vs-new A/B runs; there the workspace is zeroed by the benchmark when the old heuristic
// splits K, since the old launcher did not zero it.
//
// Usage: bench_int8_sparse [--rounds R] [--csv]
//   --csv prints one machine-readable line per (shape, M): K,N,M,gemm_ms,quant_ms,tile,split
//
// API version >= 3 adds a second table: the full layer (sparse_int8_linear, fp16 in) at
// narrow-N, large-M shapes (paint-UNet 1280->320 at 147456 tokens and relatives), two-pass
// (quantize kernel + GEMM) vs fused quantize + GEMM vs the automatic path. These inputs
// exceed L2, so they stream from DRAM as in a real network.
#include <forge/gemm/int8_sparse.cuh>
#include <forge/quant/quantize.cuh>
#include <forge/sparsity/prune_compress.cuh>
#include <forge/sparsity/pack_metadata.cuh>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

static float time_graph(const std::function<void(cudaStream_t)>& fn, cudaStream_t st,
                        int warmup = 100, int reps = 300, float min_sample_ms = 0.2f)
{
    cudaEvent_t s, e;
    CK(cudaEventCreate(&s));
    CK(cudaEventCreate(&e));
    for (int i = 0; i < 10; i++) fn(st);
    CK(cudaStreamSynchronize(st));
    CK(cudaEventRecord(s, st));
    for (int i = 0; i < 20; i++) fn(st);
    CK(cudaEventRecord(e, st));
    CK(cudaEventSynchronize(e));
    float est = 0.f;
    CK(cudaEventElapsedTime(&est, s, e));
    est /= 20.f;
    const int n_inner = std::max(1, (int)(min_sample_ms / std::max(est, 1e-3f) + 0.999f));

    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < n_inner; i++) fn(st);
    CK(cudaStreamEndCapture(st, &graph));
    CK(cudaGraphInstantiate(&exec, graph, 0));

    for (int i = 0; i < warmup; i++) CK(cudaGraphLaunch(exec, st));
    CK(cudaStreamSynchronize(st));
    std::vector<cudaEvent_t> ev(2 * reps);
    for (auto& x : ev) CK(cudaEventCreate(&x));
    for (int i = 0; i < reps; i++) {
        CK(cudaEventRecord(ev[2 * i], st));
        CK(cudaGraphLaunch(exec, st));
        CK(cudaEventRecord(ev[2 * i + 1], st));
    }
    CK(cudaStreamSynchronize(st));
    std::vector<float> t(reps);
    for (int i = 0; i < reps; i++) CK(cudaEventElapsedTime(&t[i], ev[2 * i], ev[2 * i + 1]));
    std::nth_element(t.begin(), t.begin() + reps / 2, t.end());
    const float med = t[reps / 2];
    for (auto& x : ev) CK(cudaEventDestroy(x));
    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(graph));
    CK(cudaEventDestroy(s));
    CK(cudaEventDestroy(e));
    return med / n_inner;
}

template <typename T>
static T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    CK(cudaMalloc(&d, h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    int rounds = 1;
    bool csv = false;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--csv")) csv = true;
    }

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
#ifdef FORGE_SPARSE_GEMM_API_VERSION
    const char* api = "cost-model dispatch";
#else
    const char* api = "pre-cost-model dispatch";
#endif
    std::printf("# %s (sm_%d%d, %d SMs) | Forge sparse GEMM, %s\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount, api);

    const int shapes[][2] = {{1536, 6144}, {6144, 1536}, {1280, 5120}, {5120, 1280}};
    const int ms[] = {64, 1024, 4096, 8192};
    const int max_m = 8192;

    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> di8(-127, 127);
    std::uniform_real_distribution<float> du(0.f, 1.f);

    bool spun = false;
    if (!csv)
        std::printf("%11s %5s | %10s %6s | %10s | %s\n", "K x N", "M", "GEMM ms", "TOPS",
                    "quant ms", "config");
    for (const auto& kn : shapes) {
        const int K = kn[0], N = kn[1];
        // Weights: random int8 -> 2:4 prune + compress + pack on the GPU
        std::vector<int8_t> hw((size_t)N * K);
        for (auto& v : hw) v = (int8_t)di8(rng);
        int8_t* d_w = upload(hw);
        int8_t* d_comp;  uint8_t* d_meta_raw;  uint32_t* d_meta;
        CK(cudaMalloc(&d_comp, (size_t)N * K / 2));
        CK(cudaMalloc(&d_meta_raw, (size_t)N * K / 4));
        CK(cudaMalloc(&d_meta, (size_t)N * K / 32 * sizeof(uint32_t)));
        forge::sparsity::prune_compress_24(d_w, d_comp, d_meta_raw, N, K);
        forge::sparsity::pack_metadata(d_meta_raw, d_meta, N, K);
        std::vector<__half> hws(N), hb(N);
        for (int i = 0; i < N; i++) {
            hws[i] = __float2half(0.001f + 0.01f * du(rng));
            hb[i]  = __float2half(du(rng) - 0.5f);
        }
        __half* d_ws_scale = upload(hws);
        __half* d_bias = upload(hb);

        // Activations for the largest M; smaller M use a prefix
        std::vector<__half> hx((size_t)max_m * K);
        std::normal_distribution<float> dn(0.f, 1.f);
        for (auto& v : hx) v = __float2half(dn(rng));
        __half* d_xh = upload(hx);
        int8_t* d_xq;  float* d_xs;  __half* d_out;  float* d_split_ws;
        CK(cudaMalloc(&d_xq, (size_t)max_m * K));
        CK(cudaMalloc(&d_xs, (size_t)max_m * sizeof(float)));
        CK(cudaMalloc(&d_out, (size_t)max_m * N * sizeof(__half)));
        CK(cudaMalloc(&d_split_ws, (size_t)max_m * N * sizeof(float)));

        for (int M : ms) {
            forge::quant::fused_quantize_activation(d_xh, d_xq, d_xs, M, K, st);
            CK(cudaStreamSynchronize(st));

#ifdef FORGE_SPARSE_GEMM_API_VERSION
            const forge::gemm::SparseGemmPlan plan =
                forge::gemm::sparse_int8_gemm_plan(M, K, N, 0, true);
            const char* tile_names[] = {"256x128", "128x128", "128x64"};
            char cfg[64];
            std::snprintf(cfg, sizeof(cfg), "%s s%d", tile_names[plan.tile], plan.split_k);
            const int tile_id = plan.tile, split_id = plan.split_k;
            auto gemm = [&](cudaStream_t s) {
                forge::gemm::sparse_int8_gemm(d_comp, d_xq, d_meta, d_xs, d_ws_scale, d_bias,
                                              d_out, M, K, N, d_split_ws, 0, s);
            };
#else
            // Replicates the old launcher's choice to report it and zero the workspace.
            const int bw = (M > 2048 && N >= 2048) ? 256 : 128;
            const int bc = (M > 2048) ? 128 : 64;
            const int spatial = ((N + bw - 1) / bw) * ((M + bc - 1) / bc);
            const int kit = K / 64;
            int split = 1;
            if (spatial < 64 && kit >= 8) {
                split = std::min(std::min(prop.multiProcessorCount / spatial, kit / 4), 32);
                split = std::max(split, 1);
            }
            char cfg[64];
            std::snprintf(cfg, sizeof(cfg), "%dx%d s%d", bw, bc, split);
            const int tile_id = (bw == 256) ? 0 : (bc == 128) ? 1 : 2, split_id = split;
            auto gemm = [&, split](cudaStream_t s) {
                if (split > 1)
                    CK(cudaMemsetAsync(d_split_ws, 0, (size_t)M * N * sizeof(float), s));
                forge::gemm::sparse_int8_gemm(d_comp, d_xq, d_meta, d_xs, d_ws_scale, d_bias,
                                              d_out, M, K, N, d_split_ws, 0, s);
            };
#endif
            auto quant = [&](cudaStream_t s) {
                forge::quant::fused_quantize_activation(d_xh, d_xq, d_xs, M, K, s);
            };

            if (!spun) {   // let the GPU reach its boost clock before the first measurement
                cudaEvent_t a, b;
                CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
                float el = 0.f;
                CK(cudaEventRecord(a, st));
                do {
                    for (int i = 0; i < 200; i++) gemm(st);
                    CK(cudaEventRecord(b, st));
                    CK(cudaEventSynchronize(b));
                    CK(cudaEventElapsedTime(&el, a, b));
                } while (el < 1000.f);
                CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
                spun = true;
            }

            std::vector<float> tg, tq;
            for (int r = 0; r < rounds; r++) {
                tg.push_back(time_graph(gemm, st));
                tq.push_back(time_graph(quant, st));
            }
            std::sort(tg.begin(), tg.end());
            std::sort(tq.begin(), tq.end());
            const float g = tg[tg.size() / 2], q = tq[tq.size() / 2];
            CK(cudaGetLastError());
            if (csv)
                std::printf("%d,%d,%d,%.5f,%.5f,%d,%d\n", K, N, M, g, q, tile_id, split_id);
            else
                std::printf("%5dx%-5d %5d | %8.4fms %5.0fT | %8.4fms | %s\n", K, N, M, g,
                            2.0 * M * K * N / (g * 1e-3) / 1e12, q, cfg);
            std::fflush(stdout);
        }
        cudaFree(d_w); cudaFree(d_comp); cudaFree(d_meta_raw); cudaFree(d_meta);
        cudaFree(d_ws_scale); cudaFree(d_bias); cudaFree(d_xh); cudaFree(d_xq); cudaFree(d_xs);
        cudaFree(d_out); cudaFree(d_split_ws);
    }
#if defined(FORGE_SPARSE_GEMM_API_VERSION) && FORGE_SPARSE_GEMM_API_VERSION >= 3
    // Full layer at narrow-N, large-M shapes: two-pass vs fused vs auto.
    const int lshapes[][3] = {{1280, 320, 147456}, {640, 320, 147456}, {640, 640, 36864},
                              {1024, 640, 37008},  {1024, 320, 37008}, {1280, 1280, 9216}};
    if (!csv)
        std::printf("\n%11s %6s | %10s | %10s | %10s | %s\n", "K x N", "M", "two-pass",
                    "fused", "auto", "auto path");
    for (const auto& s : lshapes) {
        const int K = s[0], N = s[1], M = s[2];
        std::vector<int8_t> hw((size_t)N * K);
        for (auto& v : hw) v = (int8_t)di8(rng);
        int8_t* d_w = upload(hw);
        int8_t* d_comp;  uint8_t* d_meta_raw;  uint32_t* d_meta;
        CK(cudaMalloc(&d_comp, (size_t)N * K / 2));
        CK(cudaMalloc(&d_meta_raw, (size_t)N * K / 4));
        CK(cudaMalloc(&d_meta, (size_t)N * K / 32 * sizeof(uint32_t)));
        forge::sparsity::prune_compress_24(d_w, d_comp, d_meta_raw, N, K);
        forge::sparsity::pack_metadata(d_meta_raw, d_meta, N, K);
        std::vector<__half> hws(N), hb(N);
        for (int i = 0; i < N; i++) {
            hws[i] = __float2half(0.001f + 0.01f * du(rng));
            hb[i]  = __float2half(du(rng) - 0.5f);
        }
        __half* d_ws_scale = upload(hws);
        __half* d_bias = upload(hb);
        std::vector<__half> hx((size_t)M * K);
        uint32_t lcg = 12345u;   // fast fill: uniform in [-2, 2)
        for (auto& v : hx) { lcg = lcg * 1664525u + 1013904223u; v = __float2half((lcg >> 8) * (4.0f / 16777216.0f) - 2.0f); }
        __half* d_xh = upload(hx);
        int8_t* d_xq;  float* d_xs;  __half* d_out;
        CK(cudaMalloc(&d_xq, (size_t)M * K));
        CK(cudaMalloc(&d_xs, (size_t)M * sizeof(float)));
        CK(cudaMalloc(&d_out, (size_t)M * N * sizeof(__half)));
        auto layer = [&](int path) {
            return [&, path](cudaStream_t s) {
                forge::gemm::sparse_int8_linear(d_xh, d_comp, d_meta, d_ws_scale, d_bias, d_out,
                                                M, K, N, d_xq, d_xs, nullptr, 1, s, path);
            };
        };
        std::vector<float> t2, tf, ta;
        for (int r = 0; r < rounds; r++) {
            t2.push_back(time_graph(layer(forge::gemm::kSparseLinearTwoPass), st));
            tf.push_back(time_graph(layer(forge::gemm::kSparseLinearFused), st));
            ta.push_back(time_graph(layer(forge::gemm::kSparseLinearAuto), st));
        }
        for (auto* v : {&t2, &tf, &ta}) std::sort(v->begin(), v->end());
        const bool fused = forge::gemm::sparse_int8_linear_fused(M, K, N);
        CK(cudaGetLastError());
        if (csv)
            std::printf("layer,%d,%d,%d,%.5f,%.5f,%.5f,%d\n", K, N, M, t2[t2.size() / 2],
                        tf[tf.size() / 2], ta[ta.size() / 2], fused ? 1 : 0);
        else
            std::printf("%5dx%-5d %6d | %8.4fms | %8.4fms | %8.4fms | %s\n", K, N, M,
                        t2[t2.size() / 2], tf[tf.size() / 2], ta[ta.size() / 2],
                        fused ? "fused" : "two-pass");
        std::fflush(stdout);
        cudaFree(d_w); cudaFree(d_comp); cudaFree(d_meta_raw); cudaFree(d_meta);
        cudaFree(d_ws_scale); cudaFree(d_bias); cudaFree(d_xh); cudaFree(d_xq); cudaFree(d_xs);
        cudaFree(d_out);
    }
#endif
    CK(cudaStreamDestroy(st));
    return 0;
}
