// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/sparsity/pack_metadata.cuh>
#include "test_helpers.cuh"

using namespace forge;

int main() {
    std::printf("=== Test: Pack Metadata ===\n");
    const int N = 16, K = 128;  // K/4=32 groups per row, K/32=4 chunks per row
    const int groups_per_row = K / 4;
    const int chunks_per_row = K / 32;

    // Generate random raw metadata nibbles (0-15, but valid 2:4 indices are 0-14)
    std::vector<uint8_t> h_meta_raw(N * groups_per_row);
    std::mt19937 rng(42);
    for (auto& v : h_meta_raw)
        v = (uint8_t)(rng() & 0xF);  // 4-bit nibble

    // CPU reference packing
    std::vector<uint32_t> ref_packed(N * chunks_per_row);
    for (int row = 0; row < N; row++) {
        for (int chunk = 0; chunk < chunks_per_row; chunk++) {
            uint32_t packed = 0;
            for (int g = 0; g < 8; g++) {
                uint8_t nib = h_meta_raw[row * groups_per_row + chunk * 8 + g];
                packed |= ((uint32_t)(nib & 0xF)) << (g * 4);
            }
            ref_packed[row * chunks_per_row + chunk] = packed;
        }
    }

    // GPU
    uint8_t* d_raw = test::to_device(h_meta_raw);
    uint32_t* d_packed = test::to_device_zeroed<uint32_t>(N * chunks_per_row);

    sparsity::pack_metadata(d_raw, d_packed, N, K);
    cudaDeviceSynchronize();

    auto gpu_packed = test::to_host(d_packed, (size_t)N * chunks_per_row);

    // Compare
    int mismatches = 0;
    for (size_t i = 0; i < gpu_packed.size(); i++) {
        if (gpu_packed[i] != ref_packed[i]) {
            if (mismatches < 3)
                std::fprintf(stderr, "FAIL: packed[%zu] gpu=0x%08x ref=0x%08x\n",
                             i, gpu_packed[i], ref_packed[i]);
            mismatches++;
        }
    }
    std::printf("  pack: %d/%zu mismatches\n", mismatches, gpu_packed.size());
    FORGE_ASSERT(mismatches == 0, "pack_metadata mismatch");
    std::printf("  PASS [pack]\n");

    // Round-trip: unpack and verify
    bool rt_ok = true;
    for (int row = 0; row < N; row++) {
        for (int chunk = 0; chunk < chunks_per_row; chunk++) {
            uint32_t p = gpu_packed[row * chunks_per_row + chunk];
            for (int g = 0; g < 8; g++) {
                uint8_t unpacked = (uint8_t)((p >> (g * 4)) & 0xF);
                uint8_t original = h_meta_raw[row * groups_per_row + chunk * 8 + g];
                if (unpacked != original) {
                    std::fprintf(stderr, "FAIL: unpack [%d,%d,%d] got=%d expected=%d\n",
                                 row, chunk, g, unpacked, original);
                    rt_ok = false;
                }
            }
        }
    }
    FORGE_ASSERT(rt_ok, "pack_metadata round-trip failed");
    std::printf("  PASS [round-trip]\n");

    test::free_device(d_raw);
    test::free_device(d_packed);
    std::printf("ALL PASSED\n");
    return 0;
}
