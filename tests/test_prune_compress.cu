// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/sparsity/prune_compress.cuh>
#include "test_helpers.cuh"
#include <cmath>
#include <cstdlib>

using namespace forge;

// CPU reference: 2:4 prune + compress
void prune_compress_cpu(const int8_t* weight, int8_t* compressed,
                        uint8_t* metadata, int N, int K) {
    int groups_per_row = K / 4;
    for (int row = 0; row < N; row++) {
        for (int g = 0; g < groups_per_row; g++) {
            int base = row * K + g * 4;
            int8_t vals[4] = {weight[base], weight[base+1],
                              weight[base+2], weight[base+3]};
            int abs_vals[4];
            for (int i = 0; i < 4; i++) abs_vals[i] = std::abs((int)vals[i]);

            // Find 2 smallest magnitude positions
            int min1 = 0, min2 = 1;
            if (abs_vals[min1] > abs_vals[min2]) { int t = min1; min1 = min2; min2 = t; }
            for (int i = 2; i < 4; i++) {
                if (abs_vals[i] < abs_vals[min1]) {
                    min2 = min1; min1 = i;
                } else if (abs_vals[i] < abs_vals[min2]) {
                    min2 = i;
                }
            }

            // Kept positions (sorted)
            int keep[2], ki = 0;
            for (int i = 0; i < 4; i++)
                if (i != min1 && i != min2)
                    keep[ki++] = i;

            int comp_base = row * (K / 2) + g * 2;
            compressed[comp_base]     = vals[keep[0]];
            compressed[comp_base + 1] = vals[keep[1]];

            int idx = row * groups_per_row + g;
            metadata[idx] = (uint8_t)(keep[0] | (keep[1] << 2));
        }
    }
}

int main() {
    std::printf("=== Test: Prune Compress 2:4 ===\n");
    const int N = 32, K = 128;

    std::vector<int8_t> h_weight(N * K);
    test::fill_random_int8(h_weight, 42);

    // CPU reference
    std::vector<int8_t> ref_comp(N * K / 2);
    std::vector<uint8_t> ref_meta(N * K / 4);
    prune_compress_cpu(h_weight.data(), ref_comp.data(), ref_meta.data(), N, K);

    // GPU
    int8_t* d_weight = test::to_device(h_weight);
    int8_t* d_comp = test::to_device_zeroed<int8_t>(N * K / 2);
    uint8_t* d_meta = test::to_device_zeroed<uint8_t>(N * K / 4);

    sparsity::prune_compress_24(d_weight, d_comp, d_meta, N, K);
    cudaDeviceSynchronize();

    auto gpu_comp = test::to_host(d_comp, (size_t)N * K / 2);
    auto gpu_meta = test::to_host(d_meta, (size_t)N * K / 4);

    // Compare compressed values
    int comp_mismatches = 0;
    for (size_t i = 0; i < gpu_comp.size(); i++)
        if (gpu_comp[i] != ref_comp[i]) comp_mismatches++;
    std::printf("  compressed: %d/%zu mismatches\n", comp_mismatches, gpu_comp.size());
    FORGE_ASSERT(comp_mismatches == 0, "compressed values mismatch");
    std::printf("  PASS [compressed values]\n");

    // Compare metadata
    int meta_mismatches = 0;
    for (size_t i = 0; i < gpu_meta.size(); i++)
        if (gpu_meta[i] != ref_meta[i]) meta_mismatches++;
    std::printf("  metadata: %d/%zu mismatches\n", meta_mismatches, gpu_meta.size());
    FORGE_ASSERT(meta_mismatches == 0, "metadata mismatch");
    std::printf("  PASS [metadata]\n");

    // Verify round-trip: decompress and check against pruned weight
    for (int row = 0; row < N; row++) {
        for (int g = 0; g < K / 4; g++) {
            uint8_t nib = ref_meta[row * (K / 4) + g];
            int k0 = nib & 3, k1 = (nib >> 2) & 3;
            int base = row * K + g * 4;
            int comp_base = row * (K / 2) + g * 2;
            FORGE_ASSERT(ref_comp[comp_base] == h_weight[base + k0],
                         "round-trip value 0 mismatch");
            FORGE_ASSERT(ref_comp[comp_base + 1] == h_weight[base + k1],
                         "round-trip value 1 mismatch");
        }
    }
    std::printf("  PASS [round-trip decompress]\n");

    test::free_device(d_weight);
    test::free_device(d_comp);
    test::free_device(d_meta);
    std::printf("ALL PASSED\n");
    return 0;
}
