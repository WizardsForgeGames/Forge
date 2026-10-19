// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/optim/lr_schedule.h>
#include <cstdio>
#include <cmath>

int main() {
    std::printf("=== Test: LR Schedule ===\n");
    int failures = 0;

    // Test 1: Warmup phase
    {
        forge::optim::LRScheduler sched(100, 1000, 0.01f);
        float lr0 = sched.get_lr_scale(0);    // step 0 -> 1/100 = 0.01
        float lr49 = sched.get_lr_scale(49);   // step 49 -> 50/100 = 0.50
        float lr99 = sched.get_lr_scale(99);   // step 99 -> 100/100 = 1.00

        if (std::abs(lr0 - 0.01f) > 1e-5f) {
            std::fprintf(stderr, "FAIL: warmup step 0: expected 0.01, got %.6f\n", lr0);
            failures++;
        }
        if (std::abs(lr49 - 0.50f) > 1e-5f) {
            std::fprintf(stderr, "FAIL: warmup step 49: expected 0.50, got %.6f\n", lr49);
            failures++;
        }
        if (std::abs(lr99 - 1.00f) > 1e-5f) {
            std::fprintf(stderr, "FAIL: warmup step 99: expected 1.00, got %.6f\n", lr99);
            failures++;
        }
        if (failures == 0) std::printf("  PASS [warmup phase]\n");
    }

    // Test 2: Cosine decay phase
    {
        forge::optim::LRScheduler sched(0, 1000, 0.01f);
        float lr0 = sched.get_lr_scale(0);     // Start: should be ~1.0
        float lr500 = sched.get_lr_scale(500);  // Midpoint: ~0.505
        float lr999 = sched.get_lr_scale(999);  // End: ~0.01

        if (std::abs(lr0 - 1.0f) > 0.01f) {
            std::fprintf(stderr, "FAIL: decay step 0: expected ~1.0, got %.6f\n", lr0);
            failures++;
        }
        // Midpoint of cosine: 0.01 + (1-0.01) * 0.5*(1+cos(pi*0.5)) = 0.01 + 0.99*0.5 = 0.505
        if (std::abs(lr500 - 0.505f) > 0.01f) {
            std::fprintf(stderr, "FAIL: decay step 500: expected ~0.505, got %.6f\n", lr500);
            failures++;
        }
        if (lr999 < 0.01f || lr999 > 0.03f) {
            std::fprintf(stderr, "FAIL: decay step 999: expected ~0.01, got %.6f\n", lr999);
            failures++;
        }
        if (failures == 0) std::printf("  PASS [cosine decay phase]\n");
    }

    // Test 3: Monotonic decrease
    {
        forge::optim::LRScheduler sched(100, 1000, 0.01f);
        bool monotonic = true;
        float prev = sched.get_lr_scale(100);
        for (int s = 101; s < 1000; s++) {
            float cur = sched.get_lr_scale(s);
            if (cur > prev + 1e-7f) {
                std::fprintf(stderr, "FAIL: non-monotonic at step %d: %.6f > %.6f\n", s, cur, prev);
                monotonic = false;
                break;
            }
            prev = cur;
        }
        if (monotonic) std::printf("  PASS [monotonic decrease]\n");
        else failures++;
    }

    // Test 4: Bounds
    {
        forge::optim::LRScheduler sched(100, 1000, 0.05f);
        bool in_bounds = true;
        for (int s = 0; s < 1000; s++) {
            float lr = sched.get_lr_scale(s);
            if (lr < 0.0f || lr > 1.0f + 1e-6f) {
                std::fprintf(stderr, "FAIL: out of bounds at step %d: %.6f\n", s, lr);
                in_bounds = false;
                break;
            }
        }
        // Past total_steps
        float lr_past = sched.get_lr_scale(2000);
        if (std::abs(lr_past - 0.05f) > 1e-5f) {
            std::fprintf(stderr, "FAIL: past total_steps: expected 0.05, got %.6f\n", lr_past);
            in_bounds = false;
        }
        if (in_bounds) std::printf("  PASS [bounds]\n");
        else failures++;
    }

    if (failures > 0) {
        std::printf("FAILED (%d failures)\n", failures);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
