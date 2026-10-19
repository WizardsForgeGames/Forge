// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Stefan problem solver — solidification/melting via the enthalpy method.
//
// Extends the thermal conduction solver with latent heat and phase change.
// The enthalpy method avoids explicit front tracking: the phase boundary
// emerges naturally as the liquid-fraction isosurface.
//
// Physics:
//   dH/dt = sum_neighbors( k_eff * (T_neighbor - T_center) )
//
// where H is total enthalpy and T(H) is piecewise-linear:
//
//   H < H_s:              T = H / msh                         (solid)
//   H_s <= H <= H_l:      T = (H*dT + L*T_s) / (msh*dT + L)  (mushy zone)
//   H > H_l:              T = (H - L) / msh                   (liquid)
//
//   H_s = msh * T_solidus
//   H_l = msh * T_liquidus + L
//   dT  = T_liquidus - T_solidus
//
// Liquid fraction f (the phase boundary indicator):
//   H < H_s:              f = 0
//   H_s <= H <= H_l:      f = (H - H_s) / (H_l - H_s)
//   H > H_l:              f = 1
//
// Memory layout: all fields are [B, G, G, G] stored contiguously.
// Linear index = b*G^3 + x*G^2 + y*G + z.

#pragma once

#include <cuda_runtime.h>

namespace forge { namespace sim {

// ---- Buffer size helpers ----

/// Returns the number of floats needed for the stefan precompute buffer.
/// Layout: [inv_msh, k_eff_xp..zm (6), max_enthalpy_delta, max_safe_enthalpy,
///          H_solidus, H_liquidus, inv_mushy_range, L, inv_msh_for_liquid]
///       = 14 * B * G^3.
inline int stefan_precomp_floats(int B, int G) {
    return 14 * B * G * G * G;
}

// ---- Conversion utilities ----

/// Convert temperature + liquid fraction to enthalpy.
/// H = msh * T + f * L
void temperature_to_enthalpy(
    const float* temperature,
    const float* liquid_fraction,
    const float* msh,
    const float* latent_heat,
    float*       enthalpy,
    int          n,
    cudaStream_t stream = nullptr);

/// Convert enthalpy to temperature using per-voxel phase change properties.
/// Uses precomputed thresholds from stefan_precompute().
void enthalpy_to_temperature(
    const float* enthalpy,
    const float* precomp,
    float*       temperature,
    int          B,
    int          G,
    cudaStream_t stream = nullptr);

/// Extract liquid fraction from enthalpy field.
/// f = 0 (solid), 0..1 (mushy), 1 (liquid).
void extract_liquid_fraction(
    const float* enthalpy,
    const float* precomp,
    float*       liquid_fraction,
    int          B,
    int          G,
    cudaStream_t stream = nullptr);

// ---- Precompute ----

/// Precompute step-invariant values from material fields.
/// Must be called once before stefan_step().
///
/// Per-voxel material inputs (all [B*G^3]):
///   conductivity:   thermal conductivity k (W/mK)
///   msh:            mass * specific_heat (J/K per voxel)
///   latent_heat:    latent heat L (J per voxel)
///   T_solidus:      solidification start temperature (K)
///   T_liquidus:     solidification end temperature (K)
///
/// precomp layout (14 arrays of B*G^3 floats each):
///   [0]  inv_msh            = 1 / max(msh, 1e-4)
///   [1]  k_eff_xp           = harmonic_mean(k_center, k_+X)
///   [2]  k_eff_xm           = harmonic_mean(k_center, k_-X)
///   [3]  k_eff_yp           = harmonic_mean(k_center, k_+Y)
///   [4]  k_eff_ym           = harmonic_mean(k_center, k_-Y)
///   [5]  k_eff_zp           = harmonic_mean(k_center, k_+Z)
///   [6]  k_eff_zm           = harmonic_mean(k_center, k_-Z)
///   [7]  max_enthalpy_delta = max(msh * max_temp_delta, 1.0)
///   [8]  max_safe_enthalpy  = msh * 10000.0 + L
///   [9]  H_solidus          = msh * T_solidus
///   [10] H_liquidus         = msh * T_liquidus + L
///   [11] inv_mushy_range    = 1 / max(H_liquidus - H_solidus, 1e-6)
///   [12] L                  = latent_heat (for liquid-region T recovery)
///   [13] mushy_T_coeff      = (T_liquidus - T_solidus) / max(H_liq - H_sol, 1e-6)
void stefan_precompute(
    float*       precomp,
    const float* conductivity,
    const float* msh,
    const float* latent_heat,
    const float* T_solidus,
    const float* T_liquidus,
    int          B,
    int          G,
    float        max_temp_delta  = 50.0f,
    float        conduction_scale = 1.0f,
    cudaStream_t stream = nullptr);

// ---- Simulation step ----

/// One explicit Euler step of the Stefan problem (enthalpy method).
/// Reads enthalpy_in, writes enthalpy_out. Both must be [B * G^3] floats.
void stefan_step(
    const float* enthalpy_in,
    float*       enthalpy_out,
    const float* precomp,
    int          B,
    int          G,
    float        dt,
    cudaStream_t stream = nullptr);

// ---- Full simulation ----

/// Run n_steps of Stefan problem simulation. Enthalpy is modified in-place.
///
/// scratch:  [B * G^3] floats for double-buffering.
/// precomp:  [14 * B * G^3] floats for precomputed values.
///
/// Calls stefan_precompute() once, then loops stefan_step().
void stefan_simulate(
    float*       enthalpy,
    float*       scratch,
    float*       precomp,
    const float* conductivity,
    const float* msh,
    const float* latent_heat,
    const float* T_solidus,
    const float* T_liquidus,
    int          B,
    int          G,
    int          n_steps,
    float        dt               = 0.05f,
    float        max_temp_delta   = 50.0f,
    float        conduction_scale = 1.0f,
    cudaStream_t stream = nullptr);

// ---- Convection extensions ----

/// One explicit Euler step with conduction + upwind advection.
/// Velocity is scaled by liquid fraction (enthalpy-porosity method).
/// vx, vy, vz are [B*G^3] velocity components.
void stefan_convection_step(
    const float* enthalpy_in,
    float*       enthalpy_out,
    const float* precomp,
    const float* vx,
    const float* vy,
    const float* vz,
    const float* liquid_fraction,
    int          B,
    int          G,
    float        dt,
    cudaStream_t stream = nullptr);

/// Run n_steps of Stefan+convection simulation. Enthalpy modified in-place.
/// liquid_fraction is updated each step from enthalpy for velocity scaling.
///
/// scratch:           [B * G^3] for double-buffering
/// precomp:           [14 * B * G^3] for precomputed conduction values
/// liquid_fraction:   [B * G^3] workspace (updated each step)
void stefan_convection_simulate(
    float*       enthalpy,
    float*       scratch,
    float*       precomp,
    float*       liquid_fraction,
    const float* conductivity,
    const float* msh,
    const float* latent_heat,
    const float* T_solidus,
    const float* T_liquidus,
    const float* vx,
    const float* vy,
    const float* vz,
    int          B,
    int          G,
    int          n_steps,
    float        dt               = 0.05f,
    float        max_temp_delta   = 50.0f,
    float        conduction_scale = 1.0f,
    cudaStream_t stream = nullptr);

}} // namespace forge::sim
