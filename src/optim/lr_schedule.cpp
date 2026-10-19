// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/optim/lr_schedule.h>
#include <cmath>
#include <algorithm>

static constexpr float PI = 3.14159265358979323846f;

namespace forge { namespace optim {

LRScheduler::LRScheduler(int warmup, int total, float min_frac, int restarts)
    : warmup_steps(warmup), total_steps(total), min_lr_fraction(min_frac),
      num_restarts(restarts) {}

float LRScheduler::get_lr_scale(int step) const {
    if (step < warmup_steps) {
        return (float)(step + 1) / (float)warmup_steps;
    }
    int decay_steps = total_steps - warmup_steps;
    if (decay_steps <= 0) return 1.0f;

    // Past end of schedule: clamp to min
    if (step >= total_steps) return min_lr_fraction;

    // SGDR: divide decay phase into num_restarts equal cycles
    int nr = (num_restarts > 1) ? num_restarts : 1;
    int cycle_len = decay_steps / nr;
    if (cycle_len <= 0) cycle_len = decay_steps;

    int decay_step = step - warmup_steps;
    int cycle_step = decay_step % cycle_len;

    float progress = (float)cycle_step / (float)cycle_len;
    float cosine = 0.5f * (1.0f + std::cos(PI * progress));
    return min_lr_fraction + (1.0f - min_lr_fraction) * cosine;
}

}} // namespace forge::optim
