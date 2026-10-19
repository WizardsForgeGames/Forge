// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// GPU-accelerated training data generation for thermal conduction simulation.
// Matches AI Thermal Sim's generate_data.py physics and distributions exactly.

#include <forge/sim/thermal_datagen.cuh>
#include <forge/sim/thermal.cuh>
#include "cuda_utils.cuh"

#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

namespace forge { namespace sim {

// ============================================================
// Device-side stateless PRNG (PCG hash)
// ============================================================

__device__ __forceinline__ uint32_t pcg_hash(uint32_t v) {
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/// Deterministic float in [0, 1) from (seed, index).
__device__ __forceinline__ float hash_float(uint32_t seed, uint32_t idx) {
    return pcg_hash(seed ^ (idx * 0x9E3779B9u)) * (1.0f / 4294967296.0f);
}

// ============================================================
// Host-side parameter generation
// ============================================================

void fill_material_params(
    MaterialLayoutParams* params,
    int          B,
    int          G,
    float        max_alpha,
    float        min_alpha,
    bool         hard_mode,
    uint64_t     seed)
{
    std::mt19937 rng(static_cast<unsigned>(seed));

    auto uni01  = [&]() { return std::uniform_real_distribution<float>(0.0f, 1.0f)(rng); };
    auto uni_int = [&](int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(rng);
    };
    auto log_uni = [&](float lo, float hi) {
        float log_lo = std::log(lo), log_hi = std::log(hi);
        return std::exp(std::uniform_real_distribution<float>(log_lo, log_hi)(rng));
    };

    int n_strategies = hard_mode ? 6 : 4;

    for (int b = 0; b < B; b++) {
        MaterialLayoutParams& p = params[b];
        std::memset(&p, 0, sizeof(p));

        // Generate material types
        int num_types = uni_int(1, kMaxMaterialTypes - 1);  // 1-6 solid types
        p.num_types = num_types;

        // Type 0 = air
        p.k_table[0]   = 0.0f;
        p.msh_table[0] = 1.0f;

        for (int t = 1; t <= num_types; t++) {
            float k   = log_uni(0.01f, 400.0f);
            float sh  = std::uniform_real_distribution<float>(100.0f, 5000.0f)(rng);
            float rho = log_uni(0.1f, 20000.0f);
            float m   = rho * sh;

            // Floor alpha: k/msh >= min_alpha
            if (min_alpha > 0.0f) k = std::max(k, min_alpha * m);
            // Cap alpha: k/msh <= max_alpha
            if (max_alpha > 0.0f) k = std::min(k, max_alpha * m);

            p.k_table[t]   = k;
            p.msh_table[t] = m;
        }

        // Zero unused types
        for (int t = num_types + 1; t < kMaxMaterialTypes; t++) {
            p.k_table[t]   = 0.0f;
            p.msh_table[t] = 1.0f;
        }

        // Select strategy
        p.strategy = uni_int(0, n_strategies - 1);

        switch (p.strategy) {
        case 0: {  // Uniform
            p.uniform_type = uni_int(1, num_types);
            p.has_border   = (uni01() < 0.5f) ? 1 : 0;
            p.border_width = uni_int(1, 3);
            break;
        }
        case 1: {  // Random
            p.air_frac = uni01() * 0.3f;
            break;
        }
        case 2: {  // Layered
            p.axis     = uni_int(0, 2);
            p.n_layers = uni_int(2, std::min(num_types, 7) + 1);

            for (int i = 0; i < p.n_layers; i++)
                p.layer_types[i] = uni_int(1, num_types);

            // Generate sorted boundaries
            std::vector<int> bounds;
            for (int i = 0; i < p.n_layers - 1; i++)
                bounds.push_back(uni_int(1, G - 1));
            std::sort(bounds.begin(), bounds.end());

            p.layer_boundaries[0] = 0;
            for (int i = 0; i < (int)bounds.size(); i++)
                p.layer_boundaries[i + 1] = bounds[i];
            p.layer_boundaries[p.n_layers] = G;
            break;
        }
        case 3: {  // Blobs (Voronoi)
            p.n_seeds = uni_int(2, std::min(num_types, 7) + 1);
            for (int s = 0; s < p.n_seeds; s++) {
                p.seed_pos[s * 3 + 0] = uni01() * G;
                p.seed_pos[s * 3 + 1] = uni01() * G;
                p.seed_pos[s * 3 + 2] = uni01() * G;
                p.seed_types[s] = uni_int(1, num_types);
            }
            // 30% chance: first seed is air
            if (uni01() < 0.3f && p.n_seeds > 1)
                p.seed_types[0] = 0;
            break;
        }
        case 4: {  // Sandwich (hard mode)
            p.axis           = uni_int(0, 2);
            p.conductor_type = uni_int(1, num_types);
            p.insulator_type = (num_types > 1) ? ((p.conductor_type % num_types) + 1)
                                               : p.conductor_type;

            // Override insulator conductivity to very low
            p.k_table[p.insulator_type] = std::uniform_real_distribution<float>(0.01f, 0.1f)(rng);

            p.wall_pos       = uni_int(G / 4, 3 * G / 4);
            p.wall_thickness = uni_int(1, 3);
            break;
        }
        case 5: {  // Checkerboard (hard mode)
            if (num_types >= 2) {
                p.checker_type_a = uni_int(1, num_types);
                p.checker_type_b = p.checker_type_a;
                while (p.checker_type_b == p.checker_type_a && num_types > 1)
                    p.checker_type_b = uni_int(1, num_types);

                // Force extreme contrast
                p.k_table[p.checker_type_a] = std::uniform_real_distribution<float>(200.0f, 400.0f)(rng);
                p.k_table[p.checker_type_b] = std::uniform_real_distribution<float>(0.01f, 0.1f)(rng);
            } else {
                p.strategy = 0;  // Fall back to uniform
                p.uniform_type = 1;
                p.has_border = 0;
            }
            p.checker_scale = uni_int(1, 4);
            break;
        }
        }

        // Re-apply alpha cap after strategy-specific k overrides (sandwich/checkerboard)
        if (max_alpha > 0.0f) {
            for (int t = 1; t <= num_types; t++) {
                p.k_table[t] = std::min(p.k_table[t], max_alpha * p.msh_table[t]);
            }
        }
    }
}

void fill_temperature_params(
    TemperatureFieldParams* params,
    int          B,
    int          G,
    uint64_t     seed,
    float        temp_scale,
    const std::vector<float>& strategy_weights)
{
    std::mt19937 rng(static_cast<unsigned>(seed + 0xDEADBEEF));

    auto uni01   = [&]() { return std::uniform_real_distribution<float>(0.0f, 1.0f)(rng); };
    auto uni_int = [&](int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(rng);
    };

    // Temperature strategies model ACTIVE SIMULATION REGIONS in a voxel game engine.
    // These are regions the game engine flagged for thermal simulation because
    // there IS a meaningful temperature differential. Equilibrium regions are
    // skipped by the engine and never reach the ROM.
    //
    // Realistic scenario distribution (weighted):
    //   0: Gentle environmental (sun-warmed stone, residual heat)   ~10%
    //   1: Strong gradient (fire/forge proximity)                   ~15%
    //   2: Chaotic (complex multi-source scene)                     ~15%
    //   3: Hotspots (fires, torches, magical heat sources)          ~20%
    //   4: Step interface (hot object meets cold -- iron in water)  ~20%
    //   5: Hot sphere (hot object embedded in surroundings)         ~20%

    // Default weighted strategy selection (V4: hot-heavy mix)
    // Empirically validated: [1,1,1,4,4,4] weighting gives 8.5% better
    // cross-distribution generalization than the original uniform-ish table.
    // High-delta strategies (hotspot/step/sphere) train sharper gradients
    // which generalize to easy cases, but not vice versa.
    static const int strategy_table[15] = {
        0,              //  7% gentle
        1,              //  7% gradient
        2,              //  7% chaotic
        3, 3, 3, 3,     // 20% hotspots
        4, 4, 4, 4,     // 20% step
        5, 5, 5, 5,     // 20% hot sphere
    };

    // Precompute cumulative weights if custom strategy_weights provided
    float cum_weights[6] = {};
    float total_weight = 0.0f;
    bool use_custom_weights = !strategy_weights.empty();
    if (use_custom_weights) {
        for (int s = 0; s < 6 && s < (int)strategy_weights.size(); s++) {
            total_weight += strategy_weights[s];
            cum_weights[s] = total_weight;
        }
    }

    for (int b = 0; b < B; b++) {
        TemperatureFieldParams& p = params[b];
        std::memset(&p, 0, sizeof(p));

        // Strategy selection
        if (use_custom_weights) {
            float r = uni01() * total_weight;
            p.strategy = 5;  // fallback
            for (int s = 0; s < 6 && s < (int)strategy_weights.size(); s++) {
                if (r < cum_weights[s]) { p.strategy = s; break; }
            }
        } else {
            p.strategy = strategy_table[uni_int(0, 14)];
        }
        p.axis = uni_int(0, 2);

        // Ambient temperature: 270-350K (room temp / outdoor temp)
        float ambient = 270.0f + uni01() * 80.0f;
        p.base_temp = ambient;

        // ts = temp_scale: multiplies all temperature differentials
        float ts = temp_scale;

        switch (p.strategy) {
        case 0: {  // Gentle environmental -- residual heat, sun-warmed surfaces
            float diff = 100.0f + uni01() * 200.0f;  // 100-300K
            p.grad_lo = ambient;
            p.grad_hi = ambient + diff * ts;
            p.strategy = 1;  // reuse linear gradient kernel
            break;
        }

        case 1: {  // Strong gradient -- fire/forge proximity
            float diff = 500.0f + uni01() * 1500.0f;  // 500-2000K
            p.grad_lo = ambient;
            p.grad_hi = std::min(ambient + diff * ts, 5000.0f);
            break;
        }
        case 2: {  // Chaotic -- complex scene with scattered temps
            p.noise_lo    = ambient;
            p.noise_range = (300.0f + uni01() * 1200.0f) * ts;
            break;
        }
        case 3: {  // Hotspots -- fires, torches, magical sources on ambient background
            p.n_spots = uni_int(1, 5);
            for (int s = 0; s < p.n_spots; s++) {
                p.spot_x[s]     = uni01() * G;
                p.spot_y[s]     = uni01() * G;
                p.spot_z[s]     = uni01() * G;
                p.spot_sigma[s] = 2.0f + uni01() * 6.0f;
                // 85% hot spots (fire/forge), 15% cold spots (ice magic)
                if (uni01() < 0.85f)
                    p.spot_intensity[s] = (300.0f + uni01() * 1500.0f) * ts;
                else
                    p.spot_intensity[s] = -(50.0f + uni01() * 100.0f) * ts;
            }
            break;
        }
        case 4: {  // Step -- hot object meets cold (iron in water)
            float diff = 300.0f + uni01() * 1200.0f;  // 300-1500K
            p.step_cold = ambient;
            p.step_hot  = ambient + diff * ts;
            break;
        }
        case 5: {  // Hot sphere -- hot object embedded in cooler surroundings
            p.point_radius = uni_int(3, std::max(3, G / 3));
            int margin = p.point_radius + 1;
            p.point_x    = uni_int(margin, std::max(margin, G - margin - 1));
            p.point_y    = uni_int(margin, std::max(margin, G - margin - 1));
            p.point_z    = uni_int(margin, std::max(margin, G - margin - 1));
            float diff = 400.0f + uni01() * 2000.0f;  // 400-2400K
            p.point_temp = ambient + diff * ts;
            break;
        }
        }
    }
}

// ============================================================
// GPU kernel: assign material layout
// ============================================================

__global__ void assign_layout_kernel(
    int* __restrict__                        type_map,
    const MaterialLayoutParams* __restrict__ params,
    int                                      B,
    int                                      G,
    uint32_t                                 seed)
{
    int G2  = G * G;
    int G3  = G2 * G;
    int BG3 = B * G3;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BG3) return;

    int b     = idx / G3;
    int local = idx - b * G3;
    int x     = local / G2;
    int yz    = local - x * G2;
    int y     = yz / G;
    int z     = yz - y * G;

    const MaterialLayoutParams& p = params[b];
    int result = 1;  // default: first solid type

    switch (p.strategy) {
    case 0: {  // Uniform
        result = p.uniform_type;
        if (p.has_border) {
            int bw = p.border_width;
            if (x < bw || x >= G - bw ||
                y < bw || y >= G - bw ||
                z < bw || z >= G - bw) {
                result = 0;  // air border
            }
        }
        break;
    }
    case 1: {  // Random
        // Random type 0..num_types
        uint32_t h = pcg_hash(seed ^ (uint32_t)idx);
        result = (int)(h % (uint32_t)(p.num_types + 1));
        // Air fraction override
        float r = hash_float(seed + 1u, (uint32_t)idx);
        if (r < p.air_frac) result = 0;
        break;
    }
    case 2: {  // Layered
        int coord;
        if      (p.axis == 0) coord = x;
        else if (p.axis == 1) coord = y;
        else                  coord = z;

        // Find which layer this coordinate falls in
        result = p.layer_types[0];  // default to first layer
        for (int i = 0; i < p.n_layers; i++) {
            if (coord >= p.layer_boundaries[i] && coord < p.layer_boundaries[i + 1]) {
                result = p.layer_types[i];
                break;
            }
        }
        break;
    }
    case 3: {  // Blobs (Voronoi)
        float min_dist = 1e30f;
        result = 1;
        float fx = (float)x, fy = (float)y, fz = (float)z;
        for (int s = 0; s < p.n_seeds; s++) {
            float dx = fx - p.seed_pos[s * 3 + 0];
            float dy = fy - p.seed_pos[s * 3 + 1];
            float dz = fz - p.seed_pos[s * 3 + 2];
            float dist = dx * dx + dy * dy + dz * dz;
            if (dist < min_dist) {
                min_dist = dist;
                result = p.seed_types[s];
            }
        }
        break;
    }
    case 4: {  // Sandwich
        int coord;
        if      (p.axis == 0) coord = x;
        else if (p.axis == 1) coord = y;
        else                  coord = z;

        result = p.conductor_type;
        if (coord >= p.wall_pos && coord < p.wall_pos + p.wall_thickness) {
            result = p.insulator_type;
        }
        break;
    }
    case 5: {  // Checkerboard
        int sc = p.checker_scale;
        int parity = ((x / sc) + (y / sc) + (z / sc)) % 2;
        result = (parity == 0) ? p.checker_type_a : p.checker_type_b;
        break;
    }
    }

    type_map[idx] = result;
}

void assign_layout(
    int*                        type_map,
    const MaterialLayoutParams* d_params,
    int                         B,
    int                         G,
    uint32_t                    seed,
    cudaStream_t                stream)
{
    int BG3 = B * G * G * G;
    if (BG3 <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(BG3, block);
    assign_layout_kernel<<<grid, block, 0, stream>>>(
        type_map, d_params, B, G, seed);
}

// ============================================================
// GPU kernel: property lookup
// ============================================================

__global__ void lookup_properties_kernel(
    float* __restrict__                      conductivity,
    float* __restrict__                      msh,
    const int* __restrict__                  type_map,
    const MaterialLayoutParams* __restrict__ params,
    int                                      B,
    int                                      G)
{
    int G3  = G * G * G;
    int BG3 = B * G3;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BG3) return;

    int b    = idx / G3;
    int t    = type_map[idx];
    const MaterialLayoutParams& p = params[b];

    conductivity[idx] = p.k_table[t];
    msh[idx]          = p.msh_table[t];
}

void lookup_properties(
    float*                      conductivity,
    float*                      msh,
    const int*                  type_map,
    const MaterialLayoutParams* d_params,
    int                         B,
    int                         G,
    cudaStream_t                stream)
{
    int BG3 = B * G * G * G;
    if (BG3 <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(BG3, block);
    lookup_properties_kernel<<<grid, block, 0, stream>>>(
        conductivity, msh, type_map, d_params, B, G);
}

// ============================================================
// GPU kernel: temperature field generation
// ============================================================

__global__ void generate_temperature_kernel(
    float* __restrict__                         temperature,
    const TemperatureFieldParams* __restrict__   params,
    int                                         B,
    int                                         G,
    uint32_t                                    seed)
{
    int G2  = G * G;
    int G3  = G2 * G;
    int BG3 = B * G3;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BG3) return;

    int b     = idx / G3;
    int local = idx - b * G3;
    int x     = local / G2;
    int yz    = local - x * G2;
    int y     = yz / G;
    int z     = yz - y * G;

    const TemperatureFieldParams& p = params[b];
    float T = p.base_temp;

    switch (p.strategy) {
    case 0:  // Uniform
        break;

    case 1: {  // Linear gradient
        int coord;
        if      (p.axis == 0) coord = x;
        else if (p.axis == 1) coord = y;
        else                  coord = z;
        float frac = (float)coord / fmaxf((float)(G - 1), 1.0f);
        T = p.grad_lo + frac * (p.grad_hi - p.grad_lo);
        break;
    }
    case 2: {  // Random noise
        float r = hash_float(seed + 2u, (uint32_t)idx);
        T = p.noise_lo + r * p.noise_range;
        break;
    }
    case 3: {  // Gaussian hotspots
        T = p.base_temp;
        float fx = (float)x, fy = (float)y, fz = (float)z;
        for (int s = 0; s < p.n_spots; s++) {
            float dx = fx - p.spot_x[s];
            float dy = fy - p.spot_y[s];
            float dz = fz - p.spot_z[s];
            float dist_sq = dx * dx + dy * dy + dz * dz;
            float sig = p.spot_sigma[s];
            T += p.spot_intensity[s] * expf(-dist_sq / (2.0f * sig * sig));
        }
        break;
    }
    case 4: {  // Step function
        int coord;
        if      (p.axis == 0) coord = x;
        else if (p.axis == 1) coord = y;
        else                  coord = z;
        T = (coord >= G / 2) ? p.step_hot : p.step_cold;
        break;
    }
    case 5: {  // Hot sphere
        T = p.base_temp;
        int dx5 = x - p.point_x;
        int dy5 = y - p.point_y;
        int dz5 = z - p.point_z;
        int r2 = dx5*dx5 + dy5*dy5 + dz5*dz5;
        if (r2 <= p.point_radius * p.point_radius)
            T = p.point_temp;
        break;
    }
    }

    // Safety clamp: target range 200-5000K
    T = fminf(fmaxf(T, 200.0f), 5000.0f);
    temperature[idx] = T;
}

void generate_temperature(
    float*                        temperature,
    const TemperatureFieldParams* d_params,
    int                           B,
    int                           G,
    uint32_t                      seed,
    cudaStream_t                  stream)
{
    int BG3 = B * G * G * G;
    if (BG3 <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(BG3, block);
    generate_temperature_kernel<<<grid, block, 0, stream>>>(
        temperature, d_params, B, G, seed);
}

// ============================================================
// High-level: generate a complete training batch
// ============================================================

void generate_training_batch(
    float*       T_input,
    float*       conductivity,
    float*       msh,
    float*       T_target,
    float*       energy_buf,
    float*       scratch_buf,
    float*       precomp_buf,
    int*         type_map_buf,
    int          B,
    int          G,
    int          n_steps,
    float        dt,
    float        max_alpha,
    float        min_alpha,
    float        max_temp_delta,
    bool         hard_mode,
    float        temp_scale,
    const std::vector<float>& strategy_weights,
    uint64_t     seed,
    cudaStream_t stream)
{
    int BG3 = B * G * G * G;

    // --- Host: generate parameters ---
    std::vector<MaterialLayoutParams>    mat_params(B);
    std::vector<TemperatureFieldParams>  temp_params(B);

    fill_material_params(mat_params.data(), B, G, max_alpha, min_alpha, hard_mode, seed);
    fill_temperature_params(temp_params.data(), B, G, seed + 12345, temp_scale, strategy_weights);

    // --- Upload params to device ---
    MaterialLayoutParams*    d_mat_params  = nullptr;
    TemperatureFieldParams*  d_temp_params = nullptr;

    cudaMalloc(&d_mat_params,  B * sizeof(MaterialLayoutParams));
    cudaMalloc(&d_temp_params, B * sizeof(TemperatureFieldParams));

    cudaMemcpyAsync(d_mat_params, mat_params.data(),
                    B * sizeof(MaterialLayoutParams),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_temp_params, temp_params.data(),
                    B * sizeof(TemperatureFieldParams),
                    cudaMemcpyHostToDevice, stream);

    // --- GPU: material layout ---
    assign_layout(type_map_buf, d_mat_params, B, G,
                  static_cast<uint32_t>(seed), stream);

    // --- GPU: property lookup ---
    lookup_properties(conductivity, msh, type_map_buf, d_mat_params, B, G, stream);

    // --- GPU: temperature field ---
    generate_temperature(T_input, d_temp_params, B, G,
                         static_cast<uint32_t>(seed + 99), stream);

    // --- Copy T_input to T_target (will be overwritten by simulation) ---
    // Actually: convert T_input to energy, simulate, convert back to T_target.

    // T_input -> energy_buf
    temperature_to_energy(T_input, msh, energy_buf, BG3, stream);

    // Run FD simulation
    thermal_simulate(energy_buf, scratch_buf, precomp_buf,
                     conductivity, msh, B, G, n_steps,
                     dt, max_temp_delta, 1.0f, stream);

    // energy_buf -> T_target
    energy_to_temperature(energy_buf, msh, T_target, BG3, stream);

    // --- Cleanup device param buffers ---
    // Sync before free to ensure kernels using d_*_params have completed.
    cudaStreamSynchronize(stream);
    cudaFree(d_mat_params);
    cudaFree(d_temp_params);
}

}} // namespace forge::sim
