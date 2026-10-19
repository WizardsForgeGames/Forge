// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Training data generation for Stefan problem (solidification/melting).
//
// Generates random solidification scenarios on GPU and runs the enthalpy
// solver to produce ground-truth (input_state, target_state) pairs.
//
// Scenario types model game-world solidification situations:
//   0 = Lava pool cooling from walls (uniform hot liquid in cold container)
//   1 = Lava pour (hot liquid flowing over cold solid surface)
//   2 = Casting mold (hot liquid in a shaped cavity)
//   3 = Remelting (solid being heated past melting point)
//   4 = Two-material interface (different solidification temperatures)
//   5 = Hotspot solidification (localized heat source in cooling liquid)
//
// Each sample produces:
//   - T_input:          temperature field at time t
//   - f_input:          liquid fraction field at time t
//   - T_target:         temperature field at time t + horizon
//   - f_target:         liquid fraction field at time t + horizon
//   - conductivity:     per-voxel thermal conductivity
//   - msh:              per-voxel mass * specific_heat
//   - latent_heat:      per-voxel latent heat
//   - T_solidus:        per-voxel solidus temperature
//   - T_liquidus:       per-voxel liquidus temperature

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace forge { namespace sim {

// ---- Constants ----

constexpr int kStefanMaxMaterials      = 4;   // max distinct phase-change materials
constexpr int kStefanMaxRegions        = 6;   // max geometric regions per sample
constexpr int kStefanMaxVelocityModes  = 8;   // max Fourier modes for velocity field

// ---- Per-sample parameter structs ----

/// Material properties for one phase-change material.
struct StefanMaterial {
    float k;            // thermal conductivity (W/mK)
    float msh;          // mass * specific_heat (J/K per voxel)
    float L;            // latent heat (J per voxel)
    float T_solidus;    // solidification start (K)
    float T_liquidus;   // solidification end (K)
};

/// Scenario parameters for one training sample.
struct StefanScenarioParams {
    int   scenario_type;                            // 0-5
    int   num_materials;                            // 1-4

    StefanMaterial materials[kStefanMaxMaterials];

    // Geometry: which material fills which region
    int   geometry_type;    // 0=uniform, 1=layered, 2=voronoi, 3=sphere_in_box

    // Uniform (geometry_type=0)
    int   fill_material;    // material index for entire volume

    // Layered (geometry_type=1)
    int   layer_axis;
    int   n_layers;
    int   layer_boundaries[kStefanMaxRegions + 1];
    int   layer_materials[kStefanMaxRegions];

    // Voronoi (geometry_type=2)
    int   n_seeds;
    float seed_pos[kStefanMaxRegions * 3];
    int   seed_materials[kStefanMaxRegions];

    // Sphere-in-box (geometry_type=3)
    float sphere_cx, sphere_cy, sphere_cz;
    float sphere_radius;
    int   sphere_material;      // material inside sphere
    int   box_material;         // material outside sphere

    // Initial temperature field
    int   temp_strategy;        // 0=uniform_hot, 1=gradient, 2=region_based
    float T_hot;                // temperature for hot regions
    float T_cold;               // temperature for cold regions

    // Region-based temp: sphere interior gets T_hot, exterior gets T_cold
    // (or layered: alternating hot/cold layers)

    // Simulation control
    int   burn_in_steps;        // steps to run before capturing input state

    // Convection: divergence-free velocity field via curl of Fourier modes
    int   n_velocity_modes;                         // 0 = no convection
    float vel_kx[kStefanMaxVelocityModes];          // wavenumber x
    float vel_ky[kStefanMaxVelocityModes];          // wavenumber y
    float vel_kz[kStefanMaxVelocityModes];          // wavenumber z
    float vel_ax[kStefanMaxVelocityModes];          // amplitude x of vector potential
    float vel_ay[kStefanMaxVelocityModes];          // amplitude y of vector potential
    float vel_az[kStefanMaxVelocityModes];          // amplitude z of vector potential
    float vel_phase[kStefanMaxVelocityModes];       // phase offset
    float vel_scale;                                // global velocity magnitude
};

// ---- Host-side parameter generation ----

/// Fill scenario parameters for B samples.
/// Generates physically plausible solidification scenarios for common game materials.
///
/// grid_size:   G (voxels per side)
/// seed:        deterministic RNG seed
void fill_stefan_params(
    StefanScenarioParams* params,
    int          B,
    int          G,
    uint64_t     seed);

// ---- GPU kernels ----

/// Assign per-voxel material properties based on scenario geometry.
/// Writes to conductivity, msh, latent_heat, T_solidus, T_liquidus (all [B*G^3]).
void stefan_assign_materials(
    float*                         conductivity,
    float*                         msh,
    float*                         latent_heat,
    float*                         T_solidus,
    float*                         T_liquidus,
    const StefanScenarioParams*    d_params,
    int                            B,
    int                            G,
    cudaStream_t                   stream = nullptr);

/// Generate initial temperature and liquid fraction fields.
/// Temperature is set based on scenario strategy; liquid fraction is derived
/// from temperature relative to solidus/liquidus.
void stefan_generate_initial_state(
    float*                         temperature,
    float*                         liquid_fraction,
    const float*                   T_solidus,
    const float*                   T_liquidus,
    const StefanScenarioParams*    d_params,
    int                            B,
    int                            G,
    uint32_t                       seed,
    cudaStream_t                   stream = nullptr);

/// Generate divergence-free velocity field from Fourier modes (curl of vector potential).
/// Outputs vx, vy, vz [B*G^3] each. Velocity is scaled by liquid fraction
/// (enthalpy-porosity: solid doesn't flow).
void stefan_generate_velocity_field(
    float*                         vx,
    float*                         vy,
    float*                         vz,
    const float*                   liquid_fraction,
    const StefanScenarioParams*    d_params,
    int                            B,
    int                            G,
    cudaStream_t                   stream = nullptr);

// ---- High-level batch generation ----

/// Generate a complete Stefan problem training batch on GPU.
///
/// Outputs (all device pointers, [B * G^3] floats):
///   T_input:          temperature at capture time
///   f_input:          liquid fraction at capture time
///   T_target:         temperature at capture time + horizon
///   f_target:         liquid fraction at capture time + horizon
///   conductivity:     per-voxel thermal conductivity
///   msh:              per-voxel mass * specific_heat
///   latent_heat:      per-voxel latent heat
///   T_solidus_out:    per-voxel solidus temperature
///   T_liquidus_out:   per-voxel liquidus temperature
///
/// Scratch buffers (caller-provided):
///   enthalpy_buf:     [B * G^3] floats
///   scratch_buf:      [B * G^3] floats
///   precomp_buf:      [14 * B * G^3] floats
///
/// The pipeline:
///   1. Generate random scenario (materials + geometry + temps)
///   2. Optionally burn-in for N steps (develop interesting state)
///   3. Capture input state (T_input, f_input)
///   4. Run horizon_steps more steps
///   5. Capture target state (T_target, f_target)
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
    int          horizon_steps   = 200,
    float        dt              = 0.05f,
    float        max_temp_delta  = 50.0f,
    uint64_t     seed            = 42,
    cudaStream_t stream          = nullptr);

/// Generate a Stefan+convection training batch.
///
/// Same as generate_stefan_batch but with fluid convection:
///   - Generates divergence-free velocity field from random Fourier modes
///   - Advects enthalpy using upwind scheme (enthalpy-porosity: v * f_liquid)
///   - Produces spatially complex solidification patterns
///
/// Additional outputs (device pointers, [B * G^3] floats):
///   vx_out, vy_out, vz_out:  velocity field at capture time
///
/// Additional scratch:
///   vx_buf, vy_buf, vz_buf:  [B * G^3] floats each for velocity storage
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
    int          horizon_steps   = 200,
    float        dt              = 0.05f,
    float        max_temp_delta  = 50.0f,
    uint64_t     seed            = 42,
    cudaStream_t stream          = nullptr);

}} // namespace forge::sim
