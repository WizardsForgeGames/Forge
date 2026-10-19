// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// GPU-accelerated training data generation for thermal conduction simulation.
//
// Generates random material fields (conductivity, mass*specific_heat) and
// initial temperature fields on GPU, then runs the FD simulation to produce
// ground-truth (input, target) pairs. Replaces the Python BatchGenerator
// with a fully GPU-resident pipeline.
//
// Layout strategies match AI Thermal Sim's generate_data.py exactly:
//   0 = uniform (single material, optional air border)
//   1 = random (per-voxel random type)
//   2 = layered (horizontal layers along random axis)
//   3 = blobs (Voronoi regions from random seed points)
//   4 = sandwich (thin insulator between conductors) [hard mode]
//   5 = checkerboard (alternating extreme-contrast materials) [hard mode]
//
// Temperature field strategies:
//   0 = uniform, 1 = linear gradient, 2 = random noise,
//   3 = Gaussian hotspots, 4 = step function, 5 = point source

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace forge { namespace sim {

// ---- Constants ----

constexpr int kMaxMaterialTypes = 7;   // type 0 = air, types 1-6 = solid
constexpr int kMaxVoronoiSeeds  = 8;
constexpr int kMaxLayers        = 8;
constexpr int kMaxHotspots      = 6;

// ---- Per-sample parameter structs ----

/// Material layout parameters for one sample.
/// Generated on host, uploaded to device for the layout kernel.
struct MaterialLayoutParams {
    int   strategy;                          // 0-5
    int   num_types;                         // active solid types (1-6)

    float k_table[kMaxMaterialTypes];        // conductivity per type (W/mK)
    float msh_table[kMaxMaterialTypes];      // mass*specific_heat per type (J/K)

    // Strategy-specific fields
    int   axis;                              // for layered/sandwich

    // Uniform (0)
    int   uniform_type;
    int   has_border;
    int   border_width;

    // Random (1)
    float air_frac;

    // Layered (2)
    int   n_layers;
    int   layer_boundaries[kMaxLayers + 1];  // [0, ..., G]
    int   layer_types[kMaxLayers];

    // Blobs/Voronoi (3)
    int   n_seeds;
    float seed_pos[kMaxVoronoiSeeds * 3];    // (x,y,z) per seed
    int   seed_types[kMaxVoronoiSeeds];

    // Sandwich (4)
    int   wall_pos;
    int   wall_thickness;
    int   conductor_type;
    int   insulator_type;

    // Checkerboard (5)
    int   checker_type_a;
    int   checker_type_b;
    int   checker_scale;
};

/// Temperature field parameters for one sample.
struct TemperatureFieldParams {
    int   strategy;                          // 0-5
    float base_temp;
    int   axis;

    // Gradient (1)
    float grad_lo;
    float grad_hi;

    // Noise (2)
    float noise_lo;
    float noise_range;

    // Hotspots (3)
    int   n_spots;
    float spot_x[kMaxHotspots];
    float spot_y[kMaxHotspots];
    float spot_z[kMaxHotspots];
    float spot_sigma[kMaxHotspots];
    float spot_intensity[kMaxHotspots];

    // Step (4)
    float step_cold;
    float step_hot;

    // Hot sphere (5)
    int   point_x;
    int   point_y;
    int   point_z;
    float point_temp;
    int   point_radius;  // radius in voxels
};

// ---- Host-side parameter generation ----

/// Fill material layout parameters for B samples using C++ RNG.
/// Generates random material types, selects layout strategies, and
/// populates all strategy-specific fields. Deterministic for given seed.
void fill_material_params(
    MaterialLayoutParams* params,
    int          B,
    int          G,
    float        max_alpha,
    float        min_alpha,
    bool         hard_mode,
    uint64_t     seed);

/// Fill temperature field parameters for B samples using C++ RNG.
/// temp_scale: multiplier for temperature differentials (1.0 = default).
/// strategy_weights: per-strategy probabilities [6]. Empty = default distribution.
void fill_temperature_params(
    TemperatureFieldParams* params,
    int          B,
    int          G,
    uint64_t     seed,
    float        temp_scale = 1.0f,
    const std::vector<float>& strategy_weights = {});

// ---- GPU kernels ----

/// Assign material types to voxels based on per-sample layout strategy.
/// type_map[b*G^3 + x*G^2 + y*G + z] = material type index (0..num_types).
__global__ void assign_layout_kernel(
    int* __restrict__                      type_map,
    const MaterialLayoutParams* __restrict__ params,
    int                                    B,
    int                                    G,
    uint32_t                               seed);

void assign_layout(
    int*                        type_map,
    const MaterialLayoutParams* d_params,
    int                         B,
    int                         G,
    uint32_t                    seed,
    cudaStream_t                stream = nullptr);

/// Look up per-voxel conductivity and msh from type_map and params tables.
__global__ void lookup_properties_kernel(
    float* __restrict__                    conductivity,
    float* __restrict__                    msh,
    const int* __restrict__                type_map,
    const MaterialLayoutParams* __restrict__ params,
    int                                    B,
    int                                    G);

void lookup_properties(
    float*                      conductivity,
    float*                      msh,
    const int*                  type_map,
    const MaterialLayoutParams* d_params,
    int                         B,
    int                         G,
    cudaStream_t                stream = nullptr);

/// Generate initial temperature field based on per-sample strategy.
__global__ void generate_temperature_kernel(
    float* __restrict__                       temperature,
    const TemperatureFieldParams* __restrict__ params,
    int                                       B,
    int                                       G,
    uint32_t                                  seed);

void generate_temperature(
    float*                         temperature,
    const TemperatureFieldParams*  d_params,
    int                            B,
    int                            G,
    uint32_t                       seed,
    cudaStream_t                   stream = nullptr);

// ---- High-level batch generation ----

/// Generate a complete training batch on GPU.
///
/// Outputs (all device pointers, [B * G^3] floats):
///   T_input:      initial temperature field (Kelvin)
///   conductivity: per-voxel thermal conductivity (W/mK)
///   msh:          per-voxel mass*specific_heat (J/K)
///   T_target:     ground-truth temperature at horizon (Kelvin)
///
/// Scratch buffers (caller-provided device memory):
///   energy_buf:   [B * G^3] floats
///   scratch_buf:  [B * G^3] floats (for simulation double-buffer)
///   precomp_buf:  [9 * B * G^3] floats (for simulation precompute)
///   type_map_buf: [B * G^3] ints (for material layout)
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
    float        dt              = 0.05f,
    float        max_alpha       = 0.001f,
    float        min_alpha       = 0.0f,
    float        max_temp_delta  = 50.0f,
    bool         hard_mode       = false,
    float        temp_scale      = 1.0f,
    const std::vector<float>& strategy_weights = {},
    uint64_t     seed            = 42,
    cudaStream_t stream          = nullptr);

}} // namespace forge::sim
