// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Tests all public device primitives via thin wrapper kernels.
#include <forge/primitives/reduce.cuh>
#include <forge/primitives/distance.cuh>
#include <forge/primitives/activation.cuh>
#include <forge/primitives/atomic.cuh>
#include <forge/primitives/geometry.cuh>
#include <forge/primitives/sparse.cuh>
#include "test_helpers.cuh"
#include <cmath>

using namespace forge;

// ================================================================
// Wrapper kernels -- call device primitives, store results to gmem
// ================================================================

// --- Reductions ---

__global__ void k_warp_reduce(float* out_sum, float* out_max, float* out_min) {
    float val = (float)(threadIdx.x + 1);
    float s  = primitives::warp_reduce_sum(val);
    float mx = primitives::warp_reduce_max(val);
    float mn = primitives::warp_reduce_min(val);
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(out_sum, s);
        primitives::atomic_max_float(out_max, mx);
        primitives::atomic_min_float(out_min, mn);
    }
}

__global__ void k_block_reduce(float* out_sum, float* out_max, float* out_min) {
    extern __shared__ float smem[];
    float val = (float)(threadIdx.x + 1);
    float s  = primitives::block_reduce_sum(val, smem);
    float mx = primitives::block_reduce_max(val, smem);
    float mn = primitives::block_reduce_min(val, smem);
    if (threadIdx.x == 0) {
        *out_sum = s;
        *out_max = mx;
        *out_min = mn;
    }
}

__global__ void k_block_broadcast(float* out_sum, float* out_min, int n) {
    extern __shared__ float smem[];
    float val = (float)(threadIdx.x + 1);
    float sum_result = primitives::block_broadcast_sum(val, smem);
    float min_result = primitives::block_broadcast_min(val, smem);
    if (threadIdx.x < n) {
        out_sum[threadIdx.x] = sum_result;
        out_min[threadIdx.x] = min_result;
    }
}

// --- Generic template test ---

__global__ void k_generic_reduce(const float* input, int n,
                                  float* out_sumsq, float* out_absmax) {
    extern __shared__ float smem[];
    int i = threadIdx.x;

    // Sum of squares: caller applies x*x, reduces with SumOp
    float sq = (i < n) ? input[i] * input[i] : 0.0f;
    float sumsq = primitives::block_reduce<primitives::SumOp>(sq, smem);

    // Absolute max: caller applies fabsf, reduces with MaxOp
    float ab = (i < n) ? fabsf(input[i]) : 0.0f;
    float absmax = primitives::block_reduce<primitives::MaxOp>(ab, smem);

    if (threadIdx.x == 0) {
        *out_sumsq = sumsq;
        *out_absmax = absmax;
    }
}

// --- Distance ---

__global__ void k_distance(const float* a, const float* b, int D,
                            float* out_dsq, float* out_nsq, float* out_poincare) {
    if (threadIdx.x == 0) {
        *out_dsq      = primitives::dist_sq(a, b, D);
        *out_nsq      = primitives::norm_sq(a, D);
        *out_poincare = primitives::dist_poincare(a, b, D);
    }
}

// --- Activations ---

__global__ void k_leaky_relu_fwd(const float* z, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = primitives::leaky_relu(z[i], 0.01f);
}

__global__ void k_leaky_relu_bwd(const float* z, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = primitives::leaky_relu_grad(z[i], 1.0f, 0.01f);
}

__global__ void k_tanh_grad(const float* z, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = primitives::tanh_grad(tanhf(z[i]), 1.0f);
}

__global__ void k_gelu_fwd(const float* z, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = primitives::gelu(z[i]);
}

__global__ void k_gelu_bwd(const float* z, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = primitives::gelu_grad(z[i], 1.0f);
}

// --- Atomics ---

__global__ void k_atomic_max_min(float* out_max, float* out_min, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = (float)(i + 1);
        primitives::atomic_max_float(out_max, val);
        primitives::atomic_min_float(out_min, val);
    }
}

// --- Geometry ---

__global__ void k_conformal(const float* pos, int D, float power, float* out) {
    if (threadIdx.x == 0) {
        float nsq = primitives::norm_sq(pos, D);
        *out = primitives::conformal_factor(nsq, power);
    }
}

__global__ void k_boundary_repulsion(const float* x, int D, float lambda,
                                      float* grad) {
    if (threadIdx.x == 0) {
        for (int d = 0; d < D; d++) grad[d] = 0.0f;
        primitives::boundary_repulsion_grad(grad, x, D, lambda);
    }
}

__global__ void k_poincare_project(float* x, int D, float max_norm,
                                    float* out_norm_sq) {
    if (threadIdx.x == 0) {
        primitives::poincare_project(x, D, max_norm);
        *out_norm_sq = primitives::norm_sq(x, D);
    }
}

// --- Sparse ---

__global__ void k_sparse_accessors(const int* row_ptr, int row,
                                    int* out_begin, int* out_end, int* out_nnz) {
    if (threadIdx.x == 0) {
        *out_begin = primitives::csr_row_begin(row_ptr, row);
        *out_end   = primitives::csr_row_end(row_ptr, row);
        *out_nnz   = primitives::csr_row_nnz(row_ptr, row);
    }
}

// ================================================================
// CPU references
// ================================================================

static float cpu_gelu(float z) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kCoeff = 0.044715f;
    float inner = kSqrt2OverPi * (z + kCoeff * z * z * z);
    return 0.5f * z * (1.0f + std::tanh(inner));
}

static float cpu_gelu_grad(float z) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kCoeff = 0.044715f;
    float inner = kSqrt2OverPi * (z + kCoeff * z * z * z);
    float t = std::tanh(inner);
    float sech2 = 1.0f - t * t;
    float d_inner = kSqrt2OverPi * (1.0f + 3.0f * kCoeff * z * z);
    return 0.5f * (1.0f + t) + 0.5f * z * sech2 * d_inner;
}

// ================================================================
// Main
// ================================================================

int main() {
    std::printf("=== Test: Primitives ===\n");

    // ---- Warp reductions ----
    {
        const int N = 256;
        float* d_sum = test::to_device_zeroed<float>(1);
        float* d_max = test::to_device_zeroed<float>(1);
        // Initialize min to a large value
        std::vector<float> init_min = {1e30f};
        float* d_min = test::to_device(init_min);

        k_warp_reduce<<<1, N>>>(d_sum, d_max, d_min);
        cudaDeviceSynchronize();

        auto h_sum = test::to_host(d_sum, 1);
        auto h_max = test::to_host(d_max, 1);
        auto h_min = test::to_host(d_min, 1);

        float expected_sum = (float)(N * (N + 1) / 2);  // 32896
        FORGE_ASSERT(std::abs(h_sum[0] - expected_sum) < 1.0f, "warp_reduce_sum");
        FORGE_ASSERT(h_max[0] == (float)N, "warp_reduce_max");
        FORGE_ASSERT(h_min[0] == 1.0f, "warp_reduce_min");
        std::printf("  PASS [warp_reduce_sum/max/min]\n");

        test::free_device(d_sum);
        test::free_device(d_max);
        test::free_device(d_min);
    }

    // ---- Block reductions ----
    {
        const int N = 256;
        float* d_sum = test::to_device_zeroed<float>(1);
        float* d_max = test::to_device_zeroed<float>(1);
        float* d_min = test::to_device_zeroed<float>(1);
        int smem_bytes = (N / 32) * sizeof(float);

        k_block_reduce<<<1, N, smem_bytes>>>(d_sum, d_max, d_min);
        cudaDeviceSynchronize();

        auto h_sum = test::to_host(d_sum, 1);
        auto h_max = test::to_host(d_max, 1);
        auto h_min = test::to_host(d_min, 1);

        float expected_sum = (float)(N * (N + 1) / 2);
        FORGE_ASSERT(std::abs(h_sum[0] - expected_sum) < 1.0f, "block_reduce_sum");
        FORGE_ASSERT(h_max[0] == (float)N, "block_reduce_max");
        FORGE_ASSERT(h_min[0] == 1.0f, "block_reduce_min");
        std::printf("  PASS [block_reduce_sum/max/min]\n");

        test::free_device(d_sum);
        test::free_device(d_max);
        test::free_device(d_min);
    }

    // ---- Block broadcast ----
    {
        const int N = 256;
        float* d_out_sum = test::to_device_zeroed<float>(N);
        float* d_out_min = test::to_device_zeroed<float>(N);
        int smem_bytes = (N / 32 + 1) * sizeof(float);

        k_block_broadcast<<<1, N, smem_bytes>>>(d_out_sum, d_out_min, N);
        cudaDeviceSynchronize();

        auto h_out_sum = test::to_host(d_out_sum, N);
        auto h_out_min = test::to_host(d_out_min, N);
        float expected_sum = (float)(N * (N + 1) / 2);
        bool sum_ok = true, min_ok = true;
        for (int i = 0; i < N; i++) {
            if (std::abs(h_out_sum[i] - expected_sum) > 1.0f) {
                sum_ok = false;
                std::fprintf(stderr, "  broadcast_sum[%d]=%.1f expected=%.1f\n",
                             i, h_out_sum[i], expected_sum);
                break;
            }
            if (std::abs(h_out_min[i] - 1.0f) > 1e-6f) {
                min_ok = false;
                std::fprintf(stderr, "  broadcast_min[%d]=%.1f expected=1.0\n",
                             i, h_out_min[i]);
                break;
            }
        }
        FORGE_ASSERT(sum_ok, "block_broadcast_sum");
        FORGE_ASSERT(min_ok, "block_broadcast_min");
        std::printf("  PASS [block_broadcast_sum/min]\n");

        test::free_device(d_out_sum);
        test::free_device(d_out_min);
    }

    // ---- Generic template reduce (SumSq via SumOp, AbsMax via MaxOp) ----
    {
        const int N = 256;
        std::vector<float> h_input(N);
        for (int i = 0; i < N; i++)
            h_input[i] = (i % 2 == 0) ? (float)(i + 1) : -(float)(i + 1);
        float* d_input = test::to_device(h_input);
        float* d_sumsq = test::to_device_zeroed<float>(1);
        float* d_absmax = test::to_device_zeroed<float>(1);
        int smem_bytes = (N / 32) * sizeof(float);

        k_generic_reduce<<<1, N, smem_bytes>>>(d_input, N, d_sumsq, d_absmax);
        cudaDeviceSynchronize();

        auto h_sumsq = test::to_host(d_sumsq, 1);
        auto h_absmax = test::to_host(d_absmax, 1);

        // CPU reference
        float ref_sumsq = 0.0f, ref_absmax = 0.0f;
        for (int i = 0; i < N; i++) {
            ref_sumsq += h_input[i] * h_input[i];
            ref_absmax = std::max(ref_absmax, std::abs(h_input[i]));
        }
        FORGE_ASSERT(std::abs(h_sumsq[0] - ref_sumsq) / ref_sumsq < 1e-5f,
                     "generic SumOp(x*x)");
        FORGE_ASSERT(h_absmax[0] == ref_absmax, "generic MaxOp(|x|)");
        std::printf("  PASS [generic reduce: SumSq, AbsMax]\n");

        test::free_device(d_input);
        test::free_device(d_sumsq);
        test::free_device(d_absmax);
    }

    // ---- Distance ----
    {
        const int D = 3;
        // Points inside unit ball for Poincare
        std::vector<float> a = {0.3f, 0.4f, 0.0f};
        std::vector<float> b = {0.1f, -0.2f, 0.5f};

        float* d_a = test::to_device(a);
        float* d_b = test::to_device(b);
        float* d_dsq = test::to_device_zeroed<float>(1);
        float* d_nsq = test::to_device_zeroed<float>(1);
        float* d_poincare = test::to_device_zeroed<float>(1);

        k_distance<<<1, 1>>>(d_a, d_b, D, d_dsq, d_nsq, d_poincare);
        cudaDeviceSynchronize();

        auto h_dsq = test::to_host(d_dsq, 1);
        auto h_nsq = test::to_host(d_nsq, 1);
        auto h_poincare = test::to_host(d_poincare, 1);

        // CPU reference
        float cpu_dsq = 0.0f, cpu_nsq = 0.0f;
        float na_sq = 0.0f, nb_sq = 0.0f, diff_sq = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = a[d] - b[d];
            cpu_dsq += diff * diff;
            cpu_nsq += a[d] * a[d];
            na_sq += a[d] * a[d];
            nb_sq += b[d] * b[d];
            diff_sq += diff * diff;
        }
        float denom = std::max((1.0f - na_sq) * (1.0f - nb_sq), 1e-10f);
        float arg = 1.0f + 2.0f * diff_sq / denom;
        float cpu_poincare = std::acosh(std::max(arg, 1.0f));

        FORGE_ASSERT(std::abs(h_dsq[0] - cpu_dsq) < 1e-6f, "dist_sq");
        FORGE_ASSERT(std::abs(h_nsq[0] - cpu_nsq) < 1e-6f, "norm_sq");
        FORGE_ASSERT(std::abs(h_poincare[0] - cpu_poincare) < 1e-5f, "dist_poincare");
        std::printf("  PASS [dist_sq/norm_sq/dist_poincare]\n");

        test::free_device(d_a);
        test::free_device(d_b);
        test::free_device(d_dsq);
        test::free_device(d_nsq);
        test::free_device(d_poincare);
    }

    // ---- Activations ----
    {
        const int N = 512;
        std::vector<float> h_z(N);
        test::fill_random(h_z, -2.0f, 2.0f, 42);
        float* d_z = test::to_device(h_z);
        float* d_out = test::to_device_zeroed<float>(N);
        int blocks = (N + 255) / 256;

        // Leaky ReLU forward
        k_leaky_relu_fwd<<<blocks, 256>>>(d_z, d_out, N);
        cudaDeviceSynchronize();
        auto gpu_lrelu = test::to_host(d_out, N);
        std::vector<float> ref_lrelu(N);
        for (int i = 0; i < N; i++)
            ref_lrelu[i] = (h_z[i] > 0.0f) ? h_z[i] : 0.01f * h_z[i];
        FORGE_ASSERT(test::check_arrays(gpu_lrelu, ref_lrelu, 1e-6f, 1e-6f,
                     "leaky_relu"), "leaky_relu mismatch");

        // Leaky ReLU backward
        k_leaky_relu_bwd<<<blocks, 256>>>(d_z, d_out, N);
        cudaDeviceSynchronize();
        auto gpu_lrelu_bwd = test::to_host(d_out, N);
        std::vector<float> ref_lrelu_bwd(N);
        for (int i = 0; i < N; i++)
            ref_lrelu_bwd[i] = (h_z[i] > 0.0f) ? 1.0f : 0.01f;
        FORGE_ASSERT(test::check_arrays(gpu_lrelu_bwd, ref_lrelu_bwd, 1e-6f, 1e-6f,
                     "leaky_relu_grad"), "leaky_relu_grad mismatch");

        // Tanh grad
        k_tanh_grad<<<blocks, 256>>>(d_z, d_out, N);
        cudaDeviceSynchronize();
        auto gpu_tgrad = test::to_host(d_out, N);
        std::vector<float> ref_tgrad(N);
        for (int i = 0; i < N; i++) {
            float t = std::tanh(h_z[i]);
            ref_tgrad[i] = 1.0f - t * t;
        }
        FORGE_ASSERT(test::check_arrays(gpu_tgrad, ref_tgrad, 1e-5f, 1e-4f,
                     "tanh_grad"), "tanh_grad mismatch");

        // GELU forward
        k_gelu_fwd<<<blocks, 256>>>(d_z, d_out, N);
        cudaDeviceSynchronize();
        auto gpu_gelu = test::to_host(d_out, N);
        std::vector<float> ref_gelu(N);
        for (int i = 0; i < N; i++)
            ref_gelu[i] = cpu_gelu(h_z[i]);
        FORGE_ASSERT(test::check_arrays(gpu_gelu, ref_gelu, 1e-5f, 1e-4f,
                     "gelu"), "gelu mismatch");

        // GELU backward
        k_gelu_bwd<<<blocks, 256>>>(d_z, d_out, N);
        cudaDeviceSynchronize();
        auto gpu_gelu_bwd = test::to_host(d_out, N);
        std::vector<float> ref_gelu_bwd(N);
        for (int i = 0; i < N; i++)
            ref_gelu_bwd[i] = cpu_gelu_grad(h_z[i]);
        FORGE_ASSERT(test::check_arrays(gpu_gelu_bwd, ref_gelu_bwd, 1e-5f, 1e-4f,
                     "gelu_grad"), "gelu_grad mismatch");

        test::free_device(d_z);
        test::free_device(d_out);
    }

    // ---- Atomic float max/min ----
    {
        const int N = 1024;
        float* d_max = test::to_device_zeroed<float>(1);
        std::vector<float> init_min = {1e30f};
        float* d_min = test::to_device(init_min);

        k_atomic_max_min<<<(N + 255) / 256, 256>>>(d_max, d_min, N);
        cudaDeviceSynchronize();

        auto h_max = test::to_host(d_max, 1);
        auto h_min = test::to_host(d_min, 1);
        FORGE_ASSERT(h_max[0] == (float)N, "atomic_max_float");
        FORGE_ASSERT(h_min[0] == 1.0f, "atomic_min_float");
        std::printf("  PASS [atomic_max_float/atomic_min_float]\n");

        test::free_device(d_max);
        test::free_device(d_min);
    }

    // ---- Geometry: conformal_factor ----
    {
        const int D = 3;
        std::vector<float> pos = {0.3f, 0.4f, 0.0f};  // ||pos||^2 = 0.25
        float* d_pos = test::to_device(pos);
        float* d_out = test::to_device_zeroed<float>(1);

        k_conformal<<<1, 1>>>(d_pos, D, 2.0f, d_out);
        cudaDeviceSynchronize();

        auto h_out = test::to_host(d_out, 1);
        float nsq = 0.25f;
        float cpu_cf = std::pow(1.0f - nsq, 2.0f) * 0.25f;  // (0.75)^2 / 4
        FORGE_ASSERT(std::abs(h_out[0] - cpu_cf) < 1e-6f, "conformal_factor");
        std::printf("  PASS [conformal_factor] (gpu=%.8f cpu=%.8f)\n",
                    h_out[0], cpu_cf);

        test::free_device(d_pos);
        test::free_device(d_out);
    }

    // ---- Geometry: boundary_repulsion_grad ----
    {
        const int D = 3;
        float lambda = 0.5f;
        std::vector<float> x = {0.3f, 0.4f, 0.0f};
        float* d_x = test::to_device(x);
        float* d_grad = test::to_device_zeroed<float>(D);

        k_boundary_repulsion<<<1, 1>>>(d_x, D, lambda, d_grad);
        cudaDeviceSynchronize();

        auto h_grad = test::to_host(d_grad, D);
        float nsq = 0.25f;
        float denom = 1.0f - nsq;
        float coeff = 2.0f * lambda / denom;
        for (int d = 0; d < D; d++) {
            float expected = coeff * x[d];
            FORGE_ASSERT(std::abs(h_grad[d] - expected) < 1e-6f,
                         "boundary_repulsion_grad");
        }
        std::printf("  PASS [boundary_repulsion_grad]\n");

        test::free_device(d_x);
        test::free_device(d_grad);
    }

    // ---- Geometry: poincare_project ----
    {
        const int D = 3;
        float max_norm = 0.95f;
        // Point outside the ball: ||x|| ~ 1.22
        std::vector<float> x = {0.7f, 0.8f, 0.5f};
        float* d_x = test::to_device(x);
        float* d_nsq = test::to_device_zeroed<float>(1);

        k_poincare_project<<<1, 1>>>(d_x, D, max_norm, d_nsq);
        cudaDeviceSynchronize();

        auto h_nsq = test::to_host(d_nsq, 1);
        float projected_norm = std::sqrt(h_nsq[0]);
        FORGE_ASSERT(projected_norm < max_norm, "poincare_project norm check");
        std::printf("  PASS [poincare_project] (norm=%.6f < %.2f)\n",
                    projected_norm, max_norm);

        test::free_device(d_x);
        test::free_device(d_nsq);
    }

    // ---- Sparse CSR accessors ----
    {
        // row_ptr = [0, 3, 3, 7, 10]  (4 rows: 3, 0, 4, 3 nnz)
        std::vector<int> row_ptr = {0, 3, 3, 7, 10};
        int* d_rp = test::to_device(row_ptr);
        int* d_begin = test::to_device_zeroed<int>(1);
        int* d_end   = test::to_device_zeroed<int>(1);
        int* d_nnz   = test::to_device_zeroed<int>(1);

        // Test row 2: begin=3, end=7, nnz=4
        k_sparse_accessors<<<1, 1>>>(d_rp, 2, d_begin, d_end, d_nnz);
        cudaDeviceSynchronize();

        auto h_begin = test::to_host(d_begin, 1);
        auto h_end   = test::to_host(d_end, 1);
        auto h_nnz   = test::to_host(d_nnz, 1);
        FORGE_ASSERT(h_begin[0] == 3, "csr_row_begin");
        FORGE_ASSERT(h_end[0] == 7, "csr_row_end");
        FORGE_ASSERT(h_nnz[0] == 4, "csr_row_nnz");
        std::printf("  PASS [csr_row_begin/end/nnz]\n");

        // Also test empty row (row 1: begin=3, end=3, nnz=0)
        k_sparse_accessors<<<1, 1>>>(d_rp, 1, d_begin, d_end, d_nnz);
        cudaDeviceSynchronize();
        auto h2_nnz = test::to_host(d_nnz, 1);
        FORGE_ASSERT(h2_nnz[0] == 0, "csr_row_nnz empty row");
        std::printf("  PASS [csr empty row]\n");

        test::free_device(d_rp);
        test::free_device(d_begin);
        test::free_device(d_end);
        test::free_device(d_nnz);
    }

    std::printf("ALL PASSED\n");
    return 0;
}
