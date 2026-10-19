// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Thermal conduction simulation on 3D voxel grids.
//
// Implements explicit Euler finite-difference for the heat equation:
//   dE/dt = sum_neighbors( k_eff * (T_neighbor - T_center) )
// where T = E / msh (mass * specific_heat), and k_eff is the harmonic
// mean of conductivities at material interfaces.
//
// Physics matches ConductionShader_v2.hlsl and AI Thermal Sim exactly:
//   - Harmonic mean conductivity at interfaces
//   - 6-neighbor stencil (+/-X, +/-Y, +/-Z)
//   - Energy-based simulation (temperature derived from energy)
//   - Stability clamping (maxTempDelta per step)
//   - Insulating boundaries (zero flux at grid edges)
//
// Memory layout: all fields are [B, G, G, G] stored contiguously in
// row-major order. Linear index = b*G*G*G + x*G*G + y*G + z.

#pragma once

#include <cuda_runtime.h>

namespace forge { namespace sim {

// ---- Buffer size helpers ----

/// Returns the number of floats needed for the precompute buffer.
/// Layout: [inv_msh, k_eff_xp, k_eff_xm, k_eff_yp, k_eff_ym, k_eff_zp,
///          k_eff_zm, max_energy_delta, max_safe_energy] = 9 * B * G^3.
inline int thermal_precomp_floats(int B, int G) {
    return 9 * B * G * G * G;
}

// ---- Conversion utilities ----

/// Convert temperature to energy: E[i] = T[i] * msh[i].
__global__ void temperature_to_energy_kernel(
    const float* __restrict__ temperature,
    const float* __restrict__ msh,
    float* __restrict__       energy,
    int                       n);

void temperature_to_energy(
    const float* temperature,
    const float* msh,
    float*       energy,
    int          n,
    cudaStream_t stream = nullptr);

/// Convert energy to temperature: T[i] = E[i] / max(msh[i], 1e-4).
__global__ void energy_to_temperature_kernel(
    const float* __restrict__ energy,
    const float* __restrict__ msh,
    float* __restrict__       temperature,
    int                       n);

void energy_to_temperature(
    const float* energy,
    const float* msh,
    float*       temperature,
    int          n,
    cudaStream_t stream = nullptr);

// ---- Precompute ----

/// Precompute step-invariant values from material fields.
/// Must be called once before thermal_step().
///
/// precomp layout (9 arrays of B*G^3 floats each):
///   [0] inv_msh          = 1 / max(msh, 1e-4)
///   [1] k_eff_xp         = harmonic_mean(k_center, k_+X) * conduction_scale
///   [2] k_eff_xm         = harmonic_mean(k_center, k_-X) * conduction_scale
///   [3] k_eff_yp         = harmonic_mean(k_center, k_+Y) * conduction_scale
///   [4] k_eff_ym         = harmonic_mean(k_center, k_-Y) * conduction_scale
///   [5] k_eff_zp         = harmonic_mean(k_center, k_+Z) * conduction_scale
///   [6] k_eff_zm         = harmonic_mean(k_center, k_-Z) * conduction_scale
///   [7] max_energy_delta = max(msh * max_temp_delta, 1.0)
///   [8] max_safe_energy  = msh * 10000.0
__global__ void thermal_precompute_kernel(
    float* __restrict__       precomp,
    const float* __restrict__ conductivity,
    const float* __restrict__ msh,
    int                       B,
    int                       G,
    float                     max_temp_delta,
    float                     conduction_scale);

void thermal_precompute(
    float*       precomp,
    const float* conductivity,
    const float* msh,
    int          B,
    int          G,
    float        max_temp_delta  = 50.0f,
    float        conduction_scale = 1.0f,
    cudaStream_t stream = nullptr);

// ---- Simulation step ----

/// One explicit Euler step of thermal conduction.
/// Reads energy_in, writes energy_out. Both must be [B * G^3] floats.
/// Uses precomputed values from thermal_precompute().
__global__ void thermal_step_kernel(
    const float* __restrict__ energy_in,
    float* __restrict__       energy_out,
    const float* __restrict__ precomp,
    int                       G,
    int                       BG3,
    float                     dt);

void thermal_step(
    const float* energy_in,
    float*       energy_out,
    const float* precomp,
    int          B,
    int          G,
    float        dt,
    cudaStream_t stream = nullptr);

// ---- Full simulation ----

/// Run n_steps of thermal conduction. Energy is modified in-place.
///
/// scratch:  [B * G^3] floats for double-buffering (caller-provided).
/// precomp:  [9 * B * G^3] floats for precomputed values (caller-provided).
///
/// Internally calls thermal_precompute() once, then loops thermal_step().
void thermal_simulate(
    float*       energy,
    float*       scratch,
    float*       precomp,
    const float* conductivity,
    const float* msh,
    int          B,
    int          G,
    int          n_steps,
    float        dt               = 0.05f,
    float        max_temp_delta   = 50.0f,
    float        conduction_scale = 1.0f,
    cudaStream_t stream = nullptr);

}} // namespace forge::sim
