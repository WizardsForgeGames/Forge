// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Thermal conduction simulation — explicit Euler FD on 3D voxel grids.
// Physics matches AI Thermal Sim (ConductionShader_v2.hlsl) exactly.

#include <forge/sim/thermal.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace sim {

// ---- Helper: clamped neighbor index for insulating BC ----

__device__ __forceinline__ int clamp_idx(int val, int lo, int hi) {
    return min(max(val, lo), hi);
}

__device__ __forceinline__ int voxel_idx(int base, int x, int y, int z,
                                         int G2, int G) {
    return base + x * G2 + y * G + z;
}

// ---- Conversion kernels ----

__global__ void temperature_to_energy_kernel(
    const float* __restrict__ temperature,
    const float* __restrict__ msh,
    float* __restrict__       energy,
    int                       n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    energy[idx] = temperature[idx] * msh[idx];
}

void temperature_to_energy(
    const float* temperature,
    const float* msh,
    float*       energy,
    int          n,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(n, block);
    temperature_to_energy_kernel<<<grid, block, 0, stream>>>(
        temperature, msh, energy, n);
}

__global__ void energy_to_temperature_kernel(
    const float* __restrict__ energy,
    const float* __restrict__ msh,
    float* __restrict__       temperature,
    int                       n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float m = msh[idx];
    temperature[idx] = energy[idx] / fmaxf(m, 1e-4f);
}

void energy_to_temperature(
    const float* energy,
    const float* msh,
    float*       temperature,
    int          n,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(n, block);
    energy_to_temperature_kernel<<<grid, block, 0, stream>>>(
        energy, msh, temperature, n);
}

// ---- Precompute kernel ----

__global__ void thermal_precompute_kernel(
    float* __restrict__       precomp,
    const float* __restrict__ conductivity,
    const float* __restrict__ msh,
    int                       B,
    int                       G,
    float                     max_temp_delta,
    float                     conduction_scale)
{
    int G2 = G * G;
    int G3 = G2 * G;
    int BG3 = B * G3;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BG3) return;

    // Decode 3D position
    int b     = idx / G3;
    int local = idx - b * G3;
    int x     = local / G2;
    int yz    = local - x * G2;
    int y     = yz / G;
    int z     = yz - y * G;
    int base  = b * G3;

    float m = msh[idx];
    float k_c = conductivity[idx];

    // inv_msh
    precomp[0 * BG3 + idx] = 1.0f / fmaxf(m, 1e-4f);

    // Harmonic mean k_eff for 6 directions
    // k_eff = (k_c > eps && k_n > eps) ? 2*k_c*k_n / (k_c+k_n+eps) * scale : 0
    const float eps = 1e-6f;

    // Direction offsets: +X, -X, +Y, -Y, +Z, -Z
    int nx[6], ny[6], nz[6];
    nx[0] = clamp_idx(x + 1, 0, G - 1); ny[0] = y;                           nz[0] = z;
    nx[1] = clamp_idx(x - 1, 0, G - 1); ny[1] = y;                           nz[1] = z;
    nx[2] = x;                           ny[2] = clamp_idx(y + 1, 0, G - 1); nz[2] = z;
    nx[3] = x;                           ny[3] = clamp_idx(y - 1, 0, G - 1); nz[3] = z;
    nx[4] = x;                           ny[4] = y;                           nz[4] = clamp_idx(z + 1, 0, G - 1);
    nx[5] = x;                           ny[5] = y;                           nz[5] = clamp_idx(z - 1, 0, G - 1);

    for (int d = 0; d < 6; d++) {
        int n_idx = voxel_idx(base, nx[d], ny[d], nz[d], G2, G);
        float k_n = conductivity[n_idx];

        float k_eff = 0.0f;
        if (k_c > eps && k_n > eps) {
            k_eff = (2.0f * k_c * k_n / (k_c + k_n + eps)) * conduction_scale;
        }
        precomp[(1 + d) * BG3 + idx] = k_eff;
    }

    // Stability clamp: max allowed |energy_delta| per voxel per step
    precomp[7 * BG3 + idx] = fmaxf(m * max_temp_delta, 1.0f);

    // Physical cap: energy can't exceed 10000K equivalent
    precomp[8 * BG3 + idx] = m * 10000.0f;
}

void thermal_precompute(
    float*       precomp,
    const float* conductivity,
    const float* msh,
    int          B,
    int          G,
    float        max_temp_delta,
    float        conduction_scale,
    cudaStream_t stream)
{
    int BG3 = B * G * G * G;
    if (BG3 <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(BG3, block);
    thermal_precompute_kernel<<<grid, block, 0, stream>>>(
        precomp, conductivity, msh, B, G, max_temp_delta, conduction_scale);
}

// ---- Step kernel ----

__global__ void thermal_step_kernel(
    const float* __restrict__ energy_in,
    float* __restrict__       energy_out,
    const float* __restrict__ precomp,
    int                       G,
    int                       BG3,
    float                     dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BG3) return;

    int G2 = G * G;
    int G3 = G2 * G;

    // Decode 3D position
    int b     = idx / G3;
    int local = idx - b * G3;
    int x     = local / G2;
    int yz    = local - x * G2;
    int y     = yz / G;
    int z     = yz - y * G;
    int base  = b * G3;

    // Read precomputed values for this voxel
    float inv_msh_c = precomp[0 * BG3 + idx];
    float max_ed    = precomp[7 * BG3 + idx];
    float max_safe  = precomp[8 * BG3 + idx];

    // Center temperature
    float E_c = energy_in[idx];
    float T_c = E_c * inv_msh_c;

    // Accumulate flux from 6 neighbors
    float flux = 0.0f;

    // +X
    {
        int ni = voxel_idx(base, clamp_idx(x + 1, 0, G - 1), y, z, G2, G);
        float T_n = energy_in[ni] * precomp[ni];  // precomp[0*BG3+ni] = inv_msh[ni]
        flux += precomp[1 * BG3 + idx] * (T_n - T_c);
    }
    // -X
    {
        int ni = voxel_idx(base, clamp_idx(x - 1, 0, G - 1), y, z, G2, G);
        float T_n = energy_in[ni] * precomp[ni];
        flux += precomp[2 * BG3 + idx] * (T_n - T_c);
    }
    // +Y
    {
        int ni = voxel_idx(base, x, clamp_idx(y + 1, 0, G - 1), z, G2, G);
        float T_n = energy_in[ni] * precomp[ni];
        flux += precomp[3 * BG3 + idx] * (T_n - T_c);
    }
    // -Y
    {
        int ni = voxel_idx(base, x, clamp_idx(y - 1, 0, G - 1), z, G2, G);
        float T_n = energy_in[ni] * precomp[ni];
        flux += precomp[4 * BG3 + idx] * (T_n - T_c);
    }
    // +Z
    {
        int ni = voxel_idx(base, x, y, clamp_idx(z + 1, 0, G - 1), G2, G);
        float T_n = energy_in[ni] * precomp[ni];
        flux += precomp[5 * BG3 + idx] * (T_n - T_c);
    }
    // -Z
    {
        int ni = voxel_idx(base, x, y, clamp_idx(z - 1, 0, G - 1), G2, G);
        float T_n = energy_in[ni] * precomp[ni];
        flux += precomp[6 * BG3 + idx] * (T_n - T_c);
    }

    // Energy update with stability clamping
    float ed = fminf(fmaxf(flux * dt, -max_ed), max_ed);
    float E_new = fminf(fmaxf(E_c + ed, 0.0f), max_safe);
    energy_out[idx] = E_new;
}

void thermal_step(
    const float* energy_in,
    float*       energy_out,
    const float* precomp,
    int          B,
    int          G,
    float        dt,
    cudaStream_t stream)
{
    int BG3 = B * G * G * G;
    if (BG3 <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(BG3, block);
    thermal_step_kernel<<<grid, block, 0, stream>>>(
        energy_in, energy_out, precomp, G, BG3, dt);
}

// ---- Full simulation ----

void thermal_simulate(
    float*       energy,
    float*       scratch,
    float*       precomp,
    const float* conductivity,
    const float* msh,
    int          B,
    int          G,
    int          n_steps,
    float        dt,
    float        max_temp_delta,
    float        conduction_scale,
    cudaStream_t stream)
{
    int BG3 = B * G * G * G;
    if (BG3 <= 0 || n_steps <= 0) return;

    // Precompute step-invariant values (once)
    thermal_precompute(precomp, conductivity, msh, B, G,
                       max_temp_delta, conduction_scale, stream);

    // Double-buffer stepping
    float* src = energy;
    float* dst = scratch;

    for (int step = 0; step < n_steps; step++) {
        thermal_step(src, dst, precomp, B, G, dt, stream);
        // Swap buffers
        float* tmp = src;
        src = dst;
        dst = tmp;
    }

    // After loop, result is in src.
    // If n_steps is odd, result is in scratch — copy back to energy.
    if (n_steps % 2 != 0) {
        cudaMemcpyAsync(energy, scratch, BG3 * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
}

}} // namespace forge::sim
