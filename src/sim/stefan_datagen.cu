// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Training data generation for Stefan problem (solidification/melting).
// Generates game-relevant solidification scenarios and runs the
// enthalpy solver to produce ground-truth training pairs.

#include <forge/sim/stefan_datagen.cuh>
#include <forge/sim/stefan.cuh>
#include "cuda_utils.cuh"

#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace forge { namespace sim {

// ============================================================
// Device-side stateless PRNG (PCG hash)
// ============================================================

__device__ __forceinline__ uint32_t stefan_pcg_hash(uint32_t v) {
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ __forceinline__ float stefan_hash_float(uint32_t seed, uint32_t idx) {
    return stefan_pcg_hash(seed ^ (idx * 0x9E3779B9u)) * (1.0f / 4294967296.0f);
}

// ============================================================
// Host-side parameter generation
// ============================================================

void fill_stefan_params(
    StefanScenarioParams* params,
    int          B,
    int          G,
    uint64_t     seed)
{
    std::mt19937 rng(static_cast<unsigned>(seed));

    auto uni01   = [&]() { return std::uniform_real_distribution<float>(0.0f, 1.0f)(rng); };
    auto uni_int = [&](int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(rng);
    };
    auto log_uni = [&](float lo, float hi) {
        float log_lo = std::log(lo), log_hi = std::log(hi);
        return std::exp(std::uniform_real_distribution<float>(log_lo, log_hi)(rng));
    };

    // Material presets (randomized around these)
    // [0] = basalt/lava, [1] = iron/metal, [2] = water/ice, [3] = stone/granite
    struct MaterialPreset {
        float k_lo, k_hi;          // conductivity range
        float msh_lo, msh_hi;      // mass*specific_heat range
        float L_lo, L_hi;          // latent heat range (per unit volume)
        float Ts_lo, Ts_hi;        // solidus range
        float Tl_lo, Tl_hi;        // liquidus range
    };
    static const MaterialPreset presets[] = {
        // Basalt/lava: k=0.5-2.5, msh=2e6-4e6, L=6e8-1.2e9, Ts=900-1100, Tl=1050-1250
        {0.5f, 2.5f, 2e6f, 4e6f, 6e8f, 1.2e9f, 900.0f, 1100.0f, 1050.0f, 1250.0f},
        // Metal/iron: k=30-80, msh=3e6-4.5e6, L=1.8e9-2.5e9, Ts=1700-1810, Tl=1750-1830
        {30.0f, 80.0f, 3e6f, 4.5e6f, 1.8e9f, 2.5e9f, 1700.0f, 1810.0f, 1750.0f, 1830.0f},
        // Water/ice: k=0.5-2.2, msh=2e6-4.2e6, L=3e8-3.5e8, Ts=270-273, Tl=273-275
        {0.5f, 2.2f, 2e6f, 4.2e6f, 3e8f, 3.5e8f, 270.0f, 273.0f, 273.0f, 275.0f},
        // Stone/granite: k=1.5-4.0, msh=2e6-3e6, L=4e8-8e8, Ts=1050-1200, Tl=1200-1400
        {1.5f, 4.0f, 2e6f, 3e6f, 4e8f, 8e8f, 1050.0f, 1200.0f, 1200.0f, 1400.0f},
    };
    constexpr int kNumPresets = 4;

    for (int b = 0; b < B; b++) {
        StefanScenarioParams& p = params[b];
        std::memset(&p, 0, sizeof(p));

        // Select scenario type
        p.scenario_type = uni_int(0, 5);

        // Generate materials (1-4)
        p.num_materials = (p.scenario_type == 4) ? uni_int(2, kStefanMaxMaterials)
                                                  : uni_int(1, 2);

        for (int m = 0; m < p.num_materials; m++) {
            // Pick a random preset and randomize within its range
            int preset_idx = uni_int(0, kNumPresets - 1);
            const auto& pr = presets[preset_idx];

            StefanMaterial& mat = p.materials[m];

            // Use normalized diffusivities for tractable simulation.
            // Realistic alpha = k/msh ranges from 1e-7 (basalt) to 2e-5 (iron),
            // which makes conduction at 1m^3 scale glacially slow. We normalize
            // to alpha in [0.01, 0.1] so that dt=0.05 gives ~1-3 voxels of
            // diffusion over 200 steps.
            //
            // Stability: dt_max = 1/(6*alpha_max) = 1/(6*0.1) = 1.67s >> 0.05.
            // Diffusion length: sqrt(2*alpha*N*dt) = sqrt(2*0.1*200*0.05) ≈ 1.4 voxels.
            //
            // Physical relationships are preserved (ratios, temperatures, latent
            // heat relative to sensible heat), just time-scaled for training.
            mat.msh = std::uniform_real_distribution<float>(500.0f, 5000.0f)(rng);
            float target_alpha = log_uni(0.01f, 0.1f);
            mat.k = target_alpha * mat.msh;

            // Latent heat scaled to msh: L/msh = Stefan number's inverse.
            // Real materials: L/(msh*dT) ~ 0.5-5.0 (Stefan number ~ 0.2-2.0)
            // Keep this ratio realistic.
            float stefan_inv = std::uniform_real_distribution<float>(0.5f, 5.0f)(rng);
            float dT = std::uniform_real_distribution<float>(
                pr.Tl_lo - pr.Ts_lo, pr.Tl_hi - pr.Ts_hi)(rng);
            dT = std::max(dT, 5.0f);
            mat.L = mat.msh * dT * stefan_inv;

            mat.T_solidus = std::uniform_real_distribution<float>(pr.Ts_lo, pr.Ts_hi)(rng);
            mat.T_liquidus = mat.T_solidus + dT;

            // Ensure T_liquidus > T_solidus (min 5K mushy zone)
            if (mat.T_liquidus <= mat.T_solidus + 5.0f)
                mat.T_liquidus = mat.T_solidus + 5.0f + uni01() * 50.0f;
        }

        // Zero unused materials
        for (int m = p.num_materials; m < kStefanMaxMaterials; m++) {
            p.materials[m] = p.materials[0];
        }

        // ---- Geometry ----
        switch (p.scenario_type) {
        case 0: {  // Lava pool cooling: uniform material, hot interior, cold boundary shell
            p.geometry_type = 3;  // sphere-in-box (hot sphere = interior, cold box = boundary)
            p.sphere_cx = G * 0.5f;
            p.sphere_cy = G * 0.5f;
            p.sphere_cz = G * 0.5f;
            p.sphere_radius = G * (0.3f + uni01() * 0.15f);
            p.sphere_material = 0;
            p.box_material = 0;  // same material, different temperature
            p.temp_strategy = 2;  // region-based: sphere hot, exterior cold
            p.T_hot = p.materials[0].T_liquidus + 100.0f + uni01() * 300.0f;
            p.T_cold = p.materials[0].T_solidus - 200.0f - uni01() * 200.0f;
            p.T_cold = std::max(p.T_cold, 200.0f);
            p.burn_in_steps = uni_int(0, 100);
            break;
        }
        case 1: {  // Lava pour: layered (hot liquid on top of cold solid)
            p.geometry_type = 1;  // layered
            p.layer_axis = 1;    // Y axis (gravity)
            p.n_layers = 2;
            int boundary = G / 3 + uni_int(0, G / 3);
            p.layer_boundaries[0] = 0;
            p.layer_boundaries[1] = boundary;
            p.layer_boundaries[2] = G;
            p.layer_materials[0] = 0;  // bottom: same material
            p.layer_materials[1] = 0;  // top: same material
            p.temp_strategy = 2;  // region-based: bottom cold, top hot
            p.T_hot = p.materials[0].T_liquidus + 50.0f + uni01() * 200.0f;
            p.T_cold = p.materials[0].T_solidus - 100.0f - uni01() * 300.0f;
            p.burn_in_steps = uni_int(10, 50);
            break;
        }
        case 2: {  // Casting mold: sphere of liquid in cold solid
            p.geometry_type = 3;  // sphere in box
            float margin = (float)G * 0.15f;
            p.sphere_cx = G * 0.5f + (uni01() - 0.5f) * G * 0.2f;
            p.sphere_cy = G * 0.5f + (uni01() - 0.5f) * G * 0.2f;
            p.sphere_cz = G * 0.5f + (uni01() - 0.5f) * G * 0.2f;
            p.sphere_radius = margin + uni01() * (G * 0.5f - margin * 1.5f);
            if (p.num_materials >= 2) {
                p.sphere_material = 0;
                p.box_material = 1;
            } else {
                p.sphere_material = 0;
                p.box_material = 0;
            }
            p.temp_strategy = 2;  // sphere hot, box cold
            p.T_hot = p.materials[p.sphere_material].T_liquidus + 50.0f + uni01() * 300.0f;
            p.T_cold = p.materials[p.box_material].T_solidus - 200.0f - uni01() * 200.0f;
            // Clamp cold to reasonable minimum
            p.T_cold = std::max(p.T_cold, 200.0f);
            p.burn_in_steps = uni_int(20, 100);
            break;
        }
        case 3: {  // Remelting: cold solid with hot sphere (heat source)
            p.geometry_type = 0;  // uniform
            p.fill_material = 0;
            p.temp_strategy = 2;  // region-based (repurpose sphere for hot region)
            // Set up sphere params even though geometry is uniform
            p.sphere_cx = G * 0.5f;
            p.sphere_cy = G * 0.5f;
            p.sphere_cz = G * 0.5f;
            p.sphere_radius = G * (0.1f + uni01() * 0.15f);
            p.sphere_material = 0;
            p.box_material = 0;
            p.T_cold = p.materials[0].T_solidus - 100.0f - uni01() * 200.0f;
            p.T_hot = p.materials[0].T_liquidus + 200.0f + uni01() * 500.0f;
            p.T_cold = std::max(p.T_cold, 200.0f);
            p.burn_in_steps = uni_int(0, 50);
            break;
        }
        case 4: {  // Two-material interface
            p.geometry_type = 1;  // layered
            p.layer_axis = uni_int(0, 2);
            p.n_layers = std::min(p.num_materials, (int)kStefanMaxRegions);
            int step = G / p.n_layers;
            for (int i = 0; i <= p.n_layers; i++)
                p.layer_boundaries[i] = std::min(i * step, G);
            p.layer_boundaries[p.n_layers] = G;
            for (int i = 0; i < p.n_layers; i++)
                p.layer_materials[i] = i % p.num_materials;
            p.temp_strategy = 1;  // gradient
            // Gradient from above liquidus to below solidus
            float max_Tl = 0.0f, min_Ts = 1e6f;
            for (int m = 0; m < p.num_materials; m++) {
                max_Tl = std::max(max_Tl, p.materials[m].T_liquidus);
                min_Ts = std::min(min_Ts, p.materials[m].T_solidus);
            }
            p.T_hot = max_Tl + 100.0f + uni01() * 200.0f;
            p.T_cold = min_Ts - 100.0f - uni01() * 200.0f;
            p.T_cold = std::max(p.T_cold, 200.0f);
            p.burn_in_steps = uni_int(20, 80);
            break;
        }
        case 5: {  // Hotspot solidification: mostly liquid with localized cold spot
            p.geometry_type = 0;  // uniform
            p.fill_material = 0;
            p.temp_strategy = 2;  // region-based but inverted
            // Sphere params for cold region
            p.sphere_cx = G * (0.2f + uni01() * 0.6f);
            p.sphere_cy = G * (0.2f + uni01() * 0.6f);
            p.sphere_cz = G * (0.2f + uni01() * 0.6f);
            p.sphere_radius = G * (0.1f + uni01() * 0.15f);
            p.sphere_material = 0;
            p.box_material = 0;
            // Invert: bulk is hot (liquid), sphere is cold (nucleation site)
            p.T_hot = p.materials[0].T_liquidus + 50.0f + uni01() * 200.0f;
            p.T_cold = p.materials[0].T_solidus - 50.0f - uni01() * 150.0f;
            p.T_cold = std::max(p.T_cold, 200.0f);
            p.burn_in_steps = uni_int(10, 60);
            break;
        }
        }
    }
}

// ============================================================
// GPU kernel: assign material properties
// ============================================================

__global__ void stefan_assign_materials_kernel(
    float* __restrict__                      conductivity,
    float* __restrict__                      msh,
    float* __restrict__                      latent_heat,
    float* __restrict__                      T_solidus,
    float* __restrict__                      T_liquidus,
    const StefanScenarioParams* __restrict__ params,
    int                                      B,
    int                                      G)
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

    const StefanScenarioParams& p = params[b];

    // Determine which material index this voxel gets
    int mat_idx = 0;

    switch (p.geometry_type) {
    case 0:  // Uniform
        mat_idx = p.fill_material;
        break;

    case 1: {  // Layered
        int coord;
        if      (p.layer_axis == 0) coord = x;
        else if (p.layer_axis == 1) coord = y;
        else                        coord = z;
        mat_idx = p.layer_materials[0];
        for (int i = 0; i < p.n_layers; i++) {
            if (coord >= p.layer_boundaries[i] && coord < p.layer_boundaries[i + 1]) {
                mat_idx = p.layer_materials[i];
                break;
            }
        }
        break;
    }
    case 2: {  // Voronoi
        float min_dist = 1e30f;
        float fx = (float)x, fy = (float)y, fz = (float)z;
        for (int s = 0; s < p.n_seeds; s++) {
            float dx = fx - p.seed_pos[s * 3 + 0];
            float dy = fy - p.seed_pos[s * 3 + 1];
            float dz = fz - p.seed_pos[s * 3 + 2];
            float dist = dx * dx + dy * dy + dz * dz;
            if (dist < min_dist) {
                min_dist = dist;
                mat_idx = p.seed_materials[s];
            }
        }
        break;
    }
    case 3: {  // Sphere in box
        float dx = (float)x - p.sphere_cx;
        float dy = (float)y - p.sphere_cy;
        float dz = (float)z - p.sphere_cz;
        float dist_sq = dx * dx + dy * dy + dz * dz;
        float r_sq = p.sphere_radius * p.sphere_radius;
        mat_idx = (dist_sq <= r_sq) ? p.sphere_material : p.box_material;
        break;
    }
    }

    // Clamp material index
    mat_idx = min(max(mat_idx, 0), p.num_materials - 1);

    const StefanMaterial& mat = p.materials[mat_idx];
    conductivity[idx] = mat.k;
    msh[idx]          = mat.msh;
    latent_heat[idx]  = mat.L;
    T_solidus[idx]    = mat.T_solidus;
    T_liquidus[idx]   = mat.T_liquidus;
}

void stefan_assign_materials(
    float*                         conductivity,
    float*                         msh,
    float*                         latent_heat,
    float*                         T_solidus,
    float*                         T_liquidus,
    const StefanScenarioParams*    d_params,
    int                            B,
    int                            G,
    cudaStream_t                   stream)
{
    int BG3 = B * G * G * G;
    if (BG3 <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(BG3, block);
    stefan_assign_materials_kernel<<<grid, block, 0, stream>>>(
        conductivity, msh, latent_heat, T_solidus, T_liquidus,
        d_params, B, G);
}

// ============================================================
// GPU kernel: generate initial temperature + liquid fraction
// ============================================================

__global__ void stefan_generate_initial_state_kernel(
    float* __restrict__                      temperature,
    float* __restrict__                      liquid_fraction,
    const float* __restrict__                T_solidus,
    const float* __restrict__                T_liquidus,
    const StefanScenarioParams* __restrict__ params,
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

    const StefanScenarioParams& p = params[b];
    float T = p.T_cold;

    switch (p.temp_strategy) {
    case 0: {  // Uniform hot (entire volume at T_hot)
        T = p.T_hot;
        break;
    }
    case 1: {  // Gradient along layer axis
        int coord;
        if      (p.layer_axis == 0) coord = x;
        else if (p.layer_axis == 1) coord = y;
        else                        coord = z;
        float frac = (float)coord / fmaxf((float)(G - 1), 1.0f);
        T = p.T_cold + frac * (p.T_hot - p.T_cold);
        break;
    }
    case 2: {  // Region-based (sphere hot, rest cold — or inverted for scenario 5)
        float dx = (float)x - p.sphere_cx;
        float dy = (float)y - p.sphere_cy;
        float dz = (float)z - p.sphere_cz;
        float dist_sq = dx * dx + dy * dy + dz * dz;
        float r_sq = p.sphere_radius * p.sphere_radius;

        if (p.scenario_type == 5) {
            // Inverted: bulk hot, sphere cold
            T = (dist_sq <= r_sq) ? p.T_cold : p.T_hot;
        } else if (p.scenario_type == 1) {
            // Lava pour: above boundary = hot, below = cold
            int coord;
            if      (p.layer_axis == 0) coord = x;
            else if (p.layer_axis == 1) coord = y;
            else                        coord = z;
            T = (coord >= p.layer_boundaries[1]) ? p.T_hot : p.T_cold;
        } else {
            // Default: sphere hot, exterior cold
            T = (dist_sq <= r_sq) ? p.T_hot : p.T_cold;
        }
        break;
    }
    }

    // Add a tiny bit of noise to break symmetry
    float noise = stefan_hash_float(seed + 3u, (uint32_t)idx) * 2.0f - 1.0f;
    T += noise * 0.5f;

    // Clamp
    T = fminf(fmaxf(T, 200.0f), 5000.0f);

    // Derive liquid fraction from temperature
    float Ts = T_solidus[idx];
    float Tl = T_liquidus[idx];
    float f;
    if (T <= Ts) {
        f = 0.0f;
    } else if (T >= Tl) {
        f = 1.0f;
    } else {
        f = (T - Ts) / fmaxf(Tl - Ts, 1e-4f);
    }

    temperature[idx]     = T;
    liquid_fraction[idx] = f;
}

void stefan_generate_initial_state(
    float*                         temperature,
    float*                         liquid_fraction,
    const float*                   T_solidus,
    const float*                   T_liquidus,
    const StefanScenarioParams*    d_params,
    int                            B,
    int                            G,
    uint32_t                       seed,
    cudaStream_t                   stream)
{
    int BG3 = B * G * G * G;
    if (BG3 <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(BG3, block);
    stefan_generate_initial_state_kernel<<<grid, block, 0, stream>>>(
        temperature, liquid_fraction, T_solidus, T_liquidus,
        d_params, B, G, seed);
}

// ============================================================
// GPU kernel: generate divergence-free velocity field
// ============================================================
//
// Velocity = curl(A) where A is a sum of Fourier modes:
//   A_x(r) = sum_m a_x[m] * sin(k . r + phase[m])
//   (same for A_y, A_z)
//
// curl(A) is automatically divergence-free: div(curl(A)) = 0.
// Velocity is then scaled by liquid fraction (enthalpy-porosity).

__global__ void stefan_generate_velocity_field_kernel(
    float* __restrict__                      vx,
    float* __restrict__                      vy,
    float* __restrict__                      vz,
    const float* __restrict__                liquid_fraction,
    const StefanScenarioParams* __restrict__ params,
    int                                      B,
    int                                      G)
{
    int G2  = G * G;
    int G3  = G2 * G;
    int BG3 = B * G3;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BG3) return;

    int b     = idx / G3;
    int local = idx - b * G3;
    int xi    = local / G2;
    int yz    = local - xi * G2;
    int yi    = yz / G;
    int zi    = yz - yi * G;

    const StefanScenarioParams& p = params[b];

    // Normalized coordinates [0, 2*pi]
    float inv_G = 6.2831853f / (float)G;  // 2*pi / G
    float rx = (float)xi * inv_G;
    float ry = (float)yi * inv_G;
    float rz = (float)zi * inv_G;

    // Accumulate spatial derivatives of vector potential for curl computation
    // dAz/dy - dAy/dz, dAx/dz - dAz/dx, dAy/dx - dAx/dy
    float dAz_dy = 0.0f, dAy_dz = 0.0f;
    float dAx_dz = 0.0f, dAz_dx = 0.0f;
    float dAy_dx = 0.0f, dAx_dy = 0.0f;

    for (int m = 0; m < p.n_velocity_modes; m++) {
        float kx = p.vel_kx[m];
        float ky = p.vel_ky[m];
        float kz = p.vel_kz[m];
        float phi = p.vel_phase[m];

        float kdotr = kx * rx + ky * ry + kz * rz + phi;
        float s = sinf(kdotr);
        float c = cosf(kdotr);

        float ax = p.vel_ax[m];
        float ay = p.vel_ay[m];
        float az = p.vel_az[m];

        // A_i = a_i * sin(k.r + phi)
        // dA_i/dj = a_i * k_j * cos(k.r + phi)
        dAz_dy += az * ky * c;
        dAy_dz += ay * kz * c;
        dAx_dz += ax * kz * c;
        dAz_dx += az * kx * c;
        dAy_dx += ay * kx * c;
        dAx_dy += ax * ky * c;
    }

    // curl(A) = (dAz/dy - dAy/dz, dAx/dz - dAz/dx, dAy/dx - dAx/dy)
    float u = (dAz_dy - dAy_dz) * p.vel_scale;
    float v = (dAx_dz - dAz_dx) * p.vel_scale;
    float w = (dAy_dx - dAx_dy) * p.vel_scale;

    // Scale by liquid fraction (enthalpy-porosity: solid doesn't flow)
    float f = liquid_fraction[idx];
    vx[idx] = u * f;
    vy[idx] = v * f;
    vz[idx] = w * f;
}

void stefan_generate_velocity_field(
    float*                         vx,
    float*                         vy,
    float*                         vz,
    const float*                   liquid_fraction,
    const StefanScenarioParams*    d_params,
    int                            B,
    int                            G,
    cudaStream_t                   stream)
{
    int BG3 = B * G * G * G;
    if (BG3 <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(BG3, block);
    stefan_generate_velocity_field_kernel<<<grid, block, 0, stream>>>(
        vx, vy, vz, liquid_fraction, d_params, B, G);
}

// ============================================================
// Host-side: fill velocity mode parameters
// ============================================================

static void fill_velocity_params(
    StefanScenarioParams& p,
    std::mt19937& rng,
    int G)
{
    auto uni01   = [&]() { return std::uniform_real_distribution<float>(0.0f, 1.0f)(rng); };
    auto uni_sym = [&](float r) { return std::uniform_real_distribution<float>(-r, r)(rng); };

    // 2-8 Fourier modes
    p.n_velocity_modes = std::uniform_int_distribution<int>(2, kStefanMaxVelocityModes)(rng);

    // Velocity scale: moderate convection (0.5 - 5.0 voxels/step at peak)
    // With dt=0.05, this gives CFL ~ 0.025 - 0.25 (well within stability)
    p.vel_scale = std::uniform_real_distribution<float>(0.5f, 5.0f)(rng);

    for (int m = 0; m < p.n_velocity_modes; m++) {
        // Wavenumbers: low-frequency (1-4) for large-scale flow
        p.vel_kx[m] = (float)std::uniform_int_distribution<int>(0, 4)(rng);
        p.vel_ky[m] = (float)std::uniform_int_distribution<int>(0, 4)(rng);
        p.vel_kz[m] = (float)std::uniform_int_distribution<int>(0, 4)(rng);
        // Ensure at least one non-zero wavenumber
        if (p.vel_kx[m] == 0.0f && p.vel_ky[m] == 0.0f && p.vel_kz[m] == 0.0f)
            p.vel_kx[m] = 1.0f;

        // Random amplitudes for vector potential
        p.vel_ax[m] = uni_sym(1.0f);
        p.vel_ay[m] = uni_sym(1.0f);
        p.vel_az[m] = uni_sym(1.0f);

        // Random phase
        p.vel_phase[m] = uni01() * 6.2831853f;
    }

    // Zero unused modes
    for (int m = p.n_velocity_modes; m < kStefanMaxVelocityModes; m++) {
        p.vel_kx[m] = p.vel_ky[m] = p.vel_kz[m] = 0.0f;
        p.vel_ax[m] = p.vel_ay[m] = p.vel_az[m] = 0.0f;
        p.vel_phase[m] = 0.0f;
    }
}

// ============================================================
// High-level: generate a complete training batch
// ============================================================

void generate_stefan_batch(
    float*       T_input,
    float*       f_input,
    float*       T_target,
    float*       f_target,
    float*       conductivity,
    float*       msh,
    float*       latent_heat,
    float*       T_solidus_out,
    float*       T_liquidus_out,
    float*       enthalpy_buf,
    float*       scratch_buf,
    float*       precomp_buf,
    int          B,
    int          G,
    int          horizon_steps,
    float        dt,
    float        max_temp_delta,
    uint64_t     seed,
    cudaStream_t stream)
{
    int BG3 = B * G * G * G;

    // --- Host: generate scenario parameters ---
    std::vector<StefanScenarioParams> params(B);
    fill_stefan_params(params.data(), B, G, seed);

    // --- Upload params to device ---
    StefanScenarioParams* d_params = nullptr;
    cudaMalloc(&d_params, B * sizeof(StefanScenarioParams));
    cudaMemcpyAsync(d_params, params.data(),
                    B * sizeof(StefanScenarioParams),
                    cudaMemcpyHostToDevice, stream);

    // --- GPU: assign material properties ---
    stefan_assign_materials(conductivity, msh, latent_heat,
                            T_solidus_out, T_liquidus_out,
                            d_params, B, G, stream);

    // --- GPU: generate initial state ---
    // We use f_input as temporary liquid fraction storage
    stefan_generate_initial_state(T_input, f_input,
                                  T_solidus_out, T_liquidus_out,
                                  d_params, B, G,
                                  static_cast<uint32_t>(seed + 99), stream);

    // --- Convert initial T + f to enthalpy ---
    temperature_to_enthalpy(T_input, f_input, msh, latent_heat,
                             enthalpy_buf, BG3, stream);

    // --- Precompute (once, used for burn-in and horizon) ---
    stefan_precompute(precomp_buf, conductivity, msh, latent_heat,
                       T_solidus_out, T_liquidus_out, B, G,
                       max_temp_delta, 1.0f, stream);

    // --- Burn-in: develop interesting intermediate state ---
    // Find max burn-in steps across all samples
    int max_burn_in = 0;
    for (int b = 0; b < B; b++)
        max_burn_in = std::max(max_burn_in, params[b].burn_in_steps);

    if (max_burn_in > 0) {
        // Run burn-in (same steps for all samples; per-sample burn-in
        // is an approximation — use the mean for simplicity)
        int mean_burn_in = 0;
        for (int b = 0; b < B; b++)
            mean_burn_in += params[b].burn_in_steps;
        mean_burn_in /= B;

        if (mean_burn_in > 0) {
            stefan_simulate(enthalpy_buf, scratch_buf, precomp_buf,
                             conductivity, msh, latent_heat,
                             T_solidus_out, T_liquidus_out,
                             B, G, mean_burn_in, dt, max_temp_delta, 1.0f,
                             stream);
            // Re-precompute is not needed (material fields haven't changed)
        }
    }

    // --- Capture input state ---
    enthalpy_to_temperature(enthalpy_buf, precomp_buf, T_input, B, G, stream);
    extract_liquid_fraction(enthalpy_buf, precomp_buf, f_input, B, G, stream);

    // --- Run horizon simulation ---
    stefan_simulate(enthalpy_buf, scratch_buf, precomp_buf,
                     conductivity, msh, latent_heat,
                     T_solidus_out, T_liquidus_out,
                     B, G, horizon_steps, dt, max_temp_delta, 1.0f, stream);

    // --- Capture target state ---
    enthalpy_to_temperature(enthalpy_buf, precomp_buf, T_target, B, G, stream);
    extract_liquid_fraction(enthalpy_buf, precomp_buf, f_target, B, G, stream);

    // --- Cleanup ---
    cudaStreamSynchronize(stream);
    cudaFree(d_params);
}

// ============================================================
// High-level: generate Stefan + convection training batch
// ============================================================

void generate_stefan_convection_batch(
    float*       T_input,
    float*       f_input,
    float*       T_target,
    float*       f_target,
    float*       conductivity,
    float*       msh,
    float*       latent_heat,
    float*       T_solidus_out,
    float*       T_liquidus_out,
    float*       vx_out,
    float*       vy_out,
    float*       vz_out,
    float*       enthalpy_buf,
    float*       scratch_buf,
    float*       precomp_buf,
    int          B,
    int          G,
    int          horizon_steps,
    float        dt,
    float        max_temp_delta,
    uint64_t     seed,
    cudaStream_t stream)
{
    int BG3 = B * G * G * G;

    // --- Host: generate scenario parameters (with velocity modes) ---
    std::vector<StefanScenarioParams> params(B);
    fill_stefan_params(params.data(), B, G, seed);

    // Add velocity parameters to each sample
    std::mt19937 vel_rng(static_cast<unsigned>(seed + 777));
    for (int b = 0; b < B; b++) {
        fill_velocity_params(params[b], vel_rng, G);
    }

    // --- Upload params to device ---
    StefanScenarioParams* d_params = nullptr;
    cudaMalloc(&d_params, B * sizeof(StefanScenarioParams));
    cudaMemcpyAsync(d_params, params.data(),
                    B * sizeof(StefanScenarioParams),
                    cudaMemcpyHostToDevice, stream);

    // --- GPU: assign material properties ---
    stefan_assign_materials(conductivity, msh, latent_heat,
                            T_solidus_out, T_liquidus_out,
                            d_params, B, G, stream);

    // --- GPU: generate initial state ---
    stefan_generate_initial_state(T_input, f_input,
                                  T_solidus_out, T_liquidus_out,
                                  d_params, B, G,
                                  static_cast<uint32_t>(seed + 99), stream);

    // --- Convert initial T + f to enthalpy ---
    temperature_to_enthalpy(T_input, f_input, msh, latent_heat,
                             enthalpy_buf, BG3, stream);

    // --- Precompute conduction values ---
    stefan_precompute(precomp_buf, conductivity, msh, latent_heat,
                       T_solidus_out, T_liquidus_out, B, G,
                       max_temp_delta, 1.0f, stream);

    // --- Generate initial velocity field ---
    // Use f_input as liquid fraction for initial velocity
    stefan_generate_velocity_field(vx_out, vy_out, vz_out,
                                   f_input, d_params, B, G, stream);

    // --- Burn-in with convection ---
    int max_burn_in = 0;
    for (int b = 0; b < B; b++)
        max_burn_in = std::max(max_burn_in, params[b].burn_in_steps);

    if (max_burn_in > 0) {
        int mean_burn_in = 0;
        for (int b = 0; b < B; b++)
            mean_burn_in += params[b].burn_in_steps;
        mean_burn_in /= B;

        if (mean_burn_in > 0) {
            // Use f_input as workspace for liquid fraction during simulation
            stefan_convection_simulate(enthalpy_buf, scratch_buf, precomp_buf,
                                       f_input,
                                       conductivity, msh, latent_heat,
                                       T_solidus_out, T_liquidus_out,
                                       vx_out, vy_out, vz_out,
                                       B, G, mean_burn_in, dt,
                                       max_temp_delta, 1.0f, stream);
        }
    }

    // --- Update velocity field after burn-in ---
    // Re-extract liquid fraction and regenerate velocity
    extract_liquid_fraction(enthalpy_buf, precomp_buf, f_input, B, G, stream);
    stefan_generate_velocity_field(vx_out, vy_out, vz_out,
                                   f_input, d_params, B, G, stream);

    // --- Capture input state ---
    enthalpy_to_temperature(enthalpy_buf, precomp_buf, T_input, B, G, stream);
    extract_liquid_fraction(enthalpy_buf, precomp_buf, f_input, B, G, stream);

    // --- Run horizon simulation with convection ---
    stefan_convection_simulate(enthalpy_buf, scratch_buf, precomp_buf,
                               f_target,  // use f_target as liquid fraction workspace
                               conductivity, msh, latent_heat,
                               T_solidus_out, T_liquidus_out,
                               vx_out, vy_out, vz_out,
                               B, G, horizon_steps, dt,
                               max_temp_delta, 1.0f, stream);

    // --- Capture target state ---
    enthalpy_to_temperature(enthalpy_buf, precomp_buf, T_target, B, G, stream);
    extract_liquid_fraction(enthalpy_buf, precomp_buf, f_target, B, G, stream);

    // --- Cleanup ---
    cudaStreamSynchronize(stream);
    cudaFree(d_params);
}

}} // namespace forge::sim
