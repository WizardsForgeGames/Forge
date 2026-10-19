// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Trilinear interpolation kernels for 3D grid sampling and scattering.
//
// Sample (gather): reads C channels from a 3D grid at arbitrary positions
//   using trilinear interpolation (8-point weighted average).
//
// Scatter (splat): writes scalar activations to a 3D grid at arbitrary
//   positions using trilinear interpolation weights (atomicAdd).
//
// Both operations produce position gradients for learned spatial layouts.
//
// Grid layout:  [C, G, G, G, B] — channels-first, batch-last (col-major per batch)
// Positions:    [N, 3] — (x, y, z) in [0, G-1] range, clamped at boundaries
// Sampled:      [N, C, B] — per-unit, per-channel, per-batch
// Values:       [N, B] — per-unit scalar activations
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

// ====== Forward ======

/// Sample C channels from a 3D grid at N positions for B batch samples.
/// For each unit i, batch b, channel c:
///   output[i * C * B + c * B + b] = trilinear_interp(grid[c, :, :, :, b], pos[i])
__global__ void trilinear_sample_kernel(
    const float* __restrict__ grid,        // [C * G * G * G * B]
    const float* __restrict__ positions,   // [N * 3]
    float* __restrict__       output,      // [N * C * B]
    int G, int N, int C, int B);

void trilinear_sample(
    const float* grid,
    const float* positions,
    float*       output,
    int G, int N, int C, int B,
    cudaStream_t stream = nullptr);

/// Scatter N scalar activations to G^3 grid using trilinear weights (atomicAdd).
/// Grid must be zeroed before calling.
/// For each unit i, batch b:
///   grid[corner, b] += w(pos[i], corner) * values[i * B + b]
__global__ void trilinear_scatter_kernel(
    const float* __restrict__ values,      // [N * B]
    const float* __restrict__ positions,   // [N * 3]
    float* __restrict__       grid,        // [G * G * G * B] (atomicAdd target)
    int G, int N, int B);

void trilinear_scatter(
    const float* values,
    const float* positions,
    float*       grid,
    int G, int N, int B,
    cudaStream_t stream = nullptr);

// ====== Backward ======

/// Position gradients from trilinear sampling.
/// d_positions accumulated via atomicAdd (caller must zero first if needed).
__global__ void trilinear_sample_pos_grad_kernel(
    const float* __restrict__ d_output,    // [N * C * B]
    const float* __restrict__ grid,        // [C * G * G * G * B]
    const float* __restrict__ positions,   // [N * 3]
    float* __restrict__       d_positions, // [N * 3] (atomicAdd)
    int G, int N, int C, int B);

void trilinear_sample_pos_grad(
    const float* d_output,
    const float* grid,
    const float* positions,
    float*       d_positions,
    int G, int N, int C, int B,
    cudaStream_t stream = nullptr);

/// Backward through scatter: computes d_values (gather from d_grid) and
/// accumulates position gradients.
/// d_positions accumulated via atomicAdd.
__global__ void trilinear_scatter_backward_kernel(
    const float* __restrict__ d_grid,      // [G * G * G * B]
    const float* __restrict__ values,      // [N * B] (needed for pos grad)
    const float* __restrict__ positions,   // [N * 3]
    float* __restrict__       d_values,    // [N * B]
    float* __restrict__       d_positions, // [N * 3] (atomicAdd)
    int G, int N, int B);

void trilinear_scatter_backward(
    const float* d_grid,
    const float* values,
    const float* positions,
    float*       d_values,
    float*       d_positions,
    int G, int N, int B,
    cudaStream_t stream = nullptr);

}} // namespace forge::nn
