// Forge GPU Compute Library -- Test Utilities
// Copyright (c) 2026 Darren Sander. MIT License.
#pragma once

#include <cuda_runtime.h>
#include <forge/error.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <string>

namespace forge { namespace test {

// ---------- Random data generation ----------

inline void fill_random(std::vector<float>& v, float lo, float hi, uint64_t seed = 42) {
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) x = dist(rng);
}

inline void fill_random_int8(std::vector<int8_t>& v, uint64_t seed = 42) {
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::uniform_int_distribution<int> dist(-127, 127);
    for (auto& x : v) x = static_cast<int8_t>(dist(rng));
}

inline void fill_zeros(std::vector<float>& v) {
    for (auto& x : v) x = 0.0f;
}

// ---------- Device transfers ----------

template<typename T>
T* to_device(const std::vector<T>& host) {
    T* d;
    FORGE_CHECK(cudaMalloc(&d, host.size() * sizeof(T)));
    FORGE_CHECK(cudaMemcpy(d, host.data(), host.size() * sizeof(T),
                           cudaMemcpyHostToDevice));
    return d;
}

template<typename T>
T* to_device_zeroed(size_t count) {
    T* d;
    FORGE_CHECK(cudaMalloc(&d, count * sizeof(T)));
    FORGE_CHECK(cudaMemset(d, 0, count * sizeof(T)));
    return d;
}

template<typename T>
std::vector<T> to_host(const T* device, size_t count) {
    std::vector<T> h(count);
    FORGE_CHECK(cudaMemcpy(h.data(), device, count * sizeof(T),
                           cudaMemcpyDeviceToHost));
    return h;
}

template<typename T>
void free_device(T* d) {
    if (d) cudaFree(d);
}

// ---------- Comparison ----------

inline bool approx_eq(float a, float b, float atol = 1e-5f, float rtol = 1e-4f) {
    return std::abs(a - b) <= atol + rtol * std::abs(b);
}

inline bool check_arrays(const std::vector<float>& gpu,
                         const std::vector<float>& ref,
                         float atol = 1e-4f, float rtol = 1e-3f,
                         const char* label = "array") {
    if (gpu.size() != ref.size()) {
        std::fprintf(stderr, "FAIL [%s]: size mismatch %zu vs %zu\n",
                     label, gpu.size(), ref.size());
        return false;
    }
    float max_diff = 0.0f;
    int first_fail = -1;
    for (size_t i = 0; i < gpu.size(); i++) {
        float diff = std::abs(gpu[i] - ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (first_fail < 0 && !approx_eq(gpu[i], ref[i], atol, rtol))
            first_fail = static_cast<int>(i);
    }
    if (first_fail >= 0) {
        std::fprintf(stderr, "FAIL [%s]: first mismatch at [%d] gpu=%.8f ref=%.8f "
                     "(max_diff=%.6e)\n",
                     label, first_fail, gpu[first_fail], ref[first_fail], max_diff);
        return false;
    }
    std::printf("  PASS [%s] (max_diff=%.6e)\n", label, max_diff);
    return true;
}

// ---------- Test assertion ----------

#define FORGE_ASSERT(cond, msg) do {                                         \
    if (!(cond)) {                                                           \
        std::fprintf(stderr, "FAIL: %s (at %s:%d)\n", msg,                  \
                     __FILE__, __LINE__);                                    \
        return 1;                                                            \
    }                                                                        \
} while(0)

#define FORGE_TEST_BEGIN(name) \
    std::printf("=== Test: %s ===\n", name); \
    int _test_failures = 0;

#define FORGE_TEST_END() \
    if (_test_failures > 0) { \
        std::printf("FAILED (%d failures)\n\n", _test_failures); \
        return 1; \
    } \
    std::printf("ALL PASSED\n\n"); \
    return 0;

}} // namespace forge::test
