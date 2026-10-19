// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Stefan problem solver — enthalpy method on 3D voxel grids.
// Extends thermal.cu with latent heat and phase change tracking.

#include <forge/sim/stefan.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace sim {

// ---- Helper: clamped neighbor index for insulating BC ----

__device__ __forceinline__ int stefan_clamp(int val, int lo, int hi) {
    return min(max(val, lo), hi);
}

__device__ __forceinline__ int stefan_voxel_idx(int base, int x, int y, int z,
                                                int G2, int G) {
    return base + x * G2 + y * G + z;
}

// ---- H -> T conversion (device helper) ----

/// Convert enthalpy to temperature for a single voxel.
/// Uses precomputed values: inv_msh [0], H_solidus [9], H_liquidus [10],
/// L [12], mushy_T_coeff [13].
__device__ __forceinline__ float enthalpy_to_temp(
    float H, int idx, const float* __restrict__ precomp, int BG3)
{
    float inv_msh    = precomp[0  * BG3 + idx];
    float H_s        = precomp[9  * BG3 + idx];
    float H_l        = precomp[10 * BG3 + idx];
    float L          = precomp[12 * BG3 + idx];
    float mushy_coeff = precomp[13 * BG3 + idx];

    if (H < H_s) {
        // Solid region: T = H / msh
        return H * inv_msh;
    } else if (H > H_l) {
        // Liquid region: T = (H - L) / msh
        return (H - L) * inv_msh;
    } else {
        // Mushy zone: T = T_solidus + (H - H_s) * mushy_T_coeff
        // where mushy_T_coeff = (T_liq - T_sol) / (H_liq - H_sol)
        float T_solidus = H_s * inv_msh;  // H_s = msh * T_solidus
        return T_solidus + (H - H_s) * mushy_coeff;
    }
}

// ---- Conversion kernels ----

__global__ void temperature_to_enthalpy_kernel(
    const float* __restrict__ temperature,
    const float* __restrict__ liquid_fraction,
    const float* __restrict__ msh,
    const float* __restrict__ latent_heat,
    float* __restrict__       enthalpy,
    int                       n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // H = msh * T + f * L
    enthalpy[idx] = msh[idx] * temperature[idx]
                  + liquid_fraction[idx] * latent_heat[idx];
}

void temperature_to_enthalpy(
    const float* temperature,
    const float* liquid_fraction,
    const float* msh,
    const float* latent_heat,
    float*       enthalpy,
    int          n,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(n, block);
    temperature_to_enthalpy_kernel<<<grid, block, 0, stream>>>(
        temperature, liquid_fraction, msh, latent_heat, enthalpy, n);
}

__global__ void enthalpy_to_temperature_kernel(
    const float* __restrict__ enthalpy,
    const float* __restrict__ precomp,
    float* __restrict__       temperature,
    int                       BG3)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BG3) return;
    temperature[idx] = enthalpy_to_temp(enthalpy[idx], idx, precomp, BG3);
}

void enthalpy_to_temperature(
    const float* enthalpy,
    const float* precomp,
    float*       temperature,
    int          B,
    int          G,
    cudaStream_t stream)
{
    int BG3 = B * G * G * G;
    if (BG3 <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(BG3, block);
    enthalpy_to_temperature_kernel<<<grid, block, 0, stream>>>(
        enthalpy, precomp, temperature, BG3);
}

// ---- Liquid fraction extraction ----

__global__ void extract_liquid_fraction_kernel(
    const float* __restrict__ enthalpy,
    const float* __restrict__ precomp,
    float* __restrict__       liquid_fraction,
    int                       BG3)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BG3) return;

    float H   = enthalpy[idx];
    float H_s = precomp[9  * BG3 + idx];
    float H_l = precomp[10 * BG3 + idx];

    float f;
    if (H <= H_s) {
        f = 0.0f;
    } else if (H >= H_l) {
        f = 1.0f;
    } else {
        float inv_range = precomp[11 * BG3 + idx];  // 1 / (H_l - H_s)
        f = (H - H_s) * inv_range;
    }
    liquid_fraction[idx] = f;
}

void extract_liquid_fraction(
    const float* enthalpy,
    const float* precomp,
    float*       liquid_fraction,
    int          B,
    int          G,
    cudaStream_t stream)
{
    int BG3 = B * G * G * G;
    if (BG3 <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(BG3, block);
    extract_liquid_fraction_kernel<<<grid, block, 0, stream>>>(
        enthalpy, precomp, liquid_fraction, BG3);
}

// ---- Precompute kernel ----

__global__ void stefan_precompute_kernel(
    float* __restrict__       precomp,
    const float* __restrict__ conductivity,
    const float* __restrict__ msh,
    const float* __restrict__ latent_heat,
    const float* __restrict__ T_solidus,
    const float* __restrict__ T_liquidus,
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

    float m   = msh[idx];
    float k_c = conductivity[idx];
    float L   = latent_heat[idx];
    float T_s = T_solidus[idx];
    float T_l = T_liquidus[idx];

    // [0] inv_msh
    float inv_m = 1.0f / fmaxf(m, 1e-4f);
    precomp[0 * BG3 + idx] = inv_m;

    // [1-6] Harmonic mean k_eff for 6 directions (identical to thermal)
    const float eps = 1e-6f;
    int nx[6], ny[6], nz[6];
    nx[0] = stefan_clamp(x + 1, 0, G - 1); ny[0] = y;                              nz[0] = z;
    nx[1] = stefan_clamp(x - 1, 0, G - 1); ny[1] = y;                              nz[1] = z;
    nx[2] = x;                              ny[2] = stefan_clamp(y + 1, 0, G - 1);  nz[2] = z;
    nx[3] = x;                              ny[3] = stefan_clamp(y - 1, 0, G - 1);  nz[3] = z;
    nx[4] = x;                              ny[4] = y;                              nz[4] = stefan_clamp(z + 1, 0, G - 1);
    nx[5] = x;                              ny[5] = y;                              nz[5] = stefan_clamp(z - 1, 0, G - 1);

    for (int d = 0; d < 6; d++) {
        int n_idx = stefan_voxel_idx(base, nx[d], ny[d], nz[d], G2, G);
        float k_n = conductivity[n_idx];

        float k_eff = 0.0f;
        if (k_c > eps && k_n > eps) {
            k_eff = (2.0f * k_c * k_n / (k_c + k_n + eps)) * conduction_scale;
        }
        precomp[(1 + d) * BG3 + idx] = k_eff;
    }

    // [7] Stability clamp: max allowed |enthalpy_delta| per step
    precomp[7 * BG3 + idx] = fmaxf(m * max_temp_delta, 1.0f);

    // [8] Physical cap: enthalpy can't exceed 10000K equivalent + full latent heat
    precomp[8 * BG3 + idx] = m * 10000.0f + L;

    // [9] H_solidus = msh * T_solidus
    float H_s = m * T_s;
    precomp[9 * BG3 + idx] = H_s;

    // [10] H_liquidus = msh * T_liquidus + L
    float H_l = m * T_l + L;
    precomp[10 * BG3 + idx] = H_l;

    // [11] inv_mushy_range = 1 / (H_liquidus - H_solidus)
    float mushy_range = H_l - H_s;
    precomp[11 * BG3 + idx] = 1.0f / fmaxf(mushy_range, 1e-6f);

    // [12] L (latent heat, needed for liquid-region T = (H-L)/msh)
    precomp[12 * BG3 + idx] = L;

    // [13] mushy_T_coeff = dT / mushy_range = (T_liq - T_sol) / (H_liq - H_sol)
    float dT = T_l - T_s;
    precomp[13 * BG3 + idx] = dT / fmaxf(mushy_range, 1e-6f);
}

void stefan_precompute(
    float*       precomp,
    const float* conductivity,
    const float* msh,
    const float* latent_heat,
    const float* T_solidus,
    const float* T_liquidus,
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
    stefan_precompute_kernel<<<grid, block, 0, stream>>>(
        precomp, conductivity, msh, latent_heat,
        T_solidus, T_liquidus, B, G,
        max_temp_delta, conduction_scale);
}

// ---- Step kernel ----

__global__ void stefan_step_kernel(
    const float* __restrict__ enthalpy_in,
    float* __restrict__       enthalpy_out,
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

    // Read precomputed values
    float max_hd   = precomp[7 * BG3 + idx];
    float max_safe = precomp[8 * BG3 + idx];

    // Center temperature from enthalpy
    float H_c = enthalpy_in[idx];
    float T_c = enthalpy_to_temp(H_c, idx, precomp, BG3);

    // Accumulate flux from 6 neighbors
    float flux = 0.0f;

    // +X
    {
        int ni = stefan_voxel_idx(base, stefan_clamp(x + 1, 0, G - 1), y, z, G2, G);
        float T_n = enthalpy_to_temp(enthalpy_in[ni], ni, precomp, BG3);
        flux += precomp[1 * BG3 + idx] * (T_n - T_c);
    }
    // -X
    {
        int ni = stefan_voxel_idx(base, stefan_clamp(x - 1, 0, G - 1), y, z, G2, G);
        float T_n = enthalpy_to_temp(enthalpy_in[ni], ni, precomp, BG3);
        flux += precomp[2 * BG3 + idx] * (T_n - T_c);
    }
    // +Y
    {
        int ni = stefan_voxel_idx(base, x, stefan_clamp(y + 1, 0, G - 1), z, G2, G);
        float T_n = enthalpy_to_temp(enthalpy_in[ni], ni, precomp, BG3);
        flux += precomp[3 * BG3 + idx] * (T_n - T_c);
    }
    // -Y
    {
        int ni = stefan_voxel_idx(base, x, stefan_clamp(y - 1, 0, G - 1), z, G2, G);
        float T_n = enthalpy_to_temp(enthalpy_in[ni], ni, precomp, BG3);
        flux += precomp[4 * BG3 + idx] * (T_n - T_c);
    }
    // +Z
    {
        int ni = stefan_voxel_idx(base, x, y, stefan_clamp(z + 1, 0, G - 1), G2, G);
        float T_n = enthalpy_to_temp(enthalpy_in[ni], ni, precomp, BG3);
        flux += precomp[5 * BG3 + idx] * (T_n - T_c);
    }
    // -Z
    {
        int ni = stefan_voxel_idx(base, x, y, stefan_clamp(z - 1, 0, G - 1), G2, G);
        float T_n = enthalpy_to_temp(enthalpy_in[ni], ni, precomp, BG3);
        flux += precomp[6 * BG3 + idx] * (T_n - T_c);
    }

    // Enthalpy update with stability clamping
    float hd = fminf(fmaxf(flux * dt, -max_hd), max_hd);
    float H_new = fminf(fmaxf(H_c + hd, 0.0f), max_safe);
    enthalpy_out[idx] = H_new;
}

void stefan_step(
    const float* enthalpy_in,
    float*       enthalpy_out,
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
    stefan_step_kernel<<<grid, block, 0, stream>>>(
        enthalpy_in, enthalpy_out, precomp, G, BG3, dt);
}

// ---- Full simulation ----

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
    float        dt,
    float        max_temp_delta,
    float        conduction_scale,
    cudaStream_t stream)
{
    int BG3 = B * G * G * G;
    if (BG3 <= 0 || n_steps <= 0) return;

    // Precompute step-invariant values (once)
    stefan_precompute(precomp, conductivity, msh, latent_heat,
                      T_solidus, T_liquidus, B, G,
                      max_temp_delta, conduction_scale, stream);

    // Double-buffer stepping
    float* src = enthalpy;
    float* dst = scratch;

    for (int step = 0; step < n_steps; step++) {
        stefan_step(src, dst, precomp, B, G, dt, stream);
        float* tmp = src;
        src = dst;
        dst = tmp;
    }

    // If n_steps is odd, result is in scratch — copy back to enthalpy.
    if (n_steps % 2 != 0) {
        cudaMemcpyAsync(enthalpy, scratch, BG3 * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
}

// ---- Convection step kernel ----

__global__ void stefan_convection_step_kernel(
    const float* __restrict__ enthalpy_in,
    float* __restrict__       enthalpy_out,
    const float* __restrict__ precomp,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ vz,
    const float* __restrict__ liquid_fraction,
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

    // Read precomputed values
    float max_hd   = precomp[7 * BG3 + idx];
    float max_safe = precomp[8 * BG3 + idx];

    // Center values
    float H_c = enthalpy_in[idx];
    float T_c = enthalpy_to_temp(H_c, idx, precomp, BG3);

    // ---- Conduction flux (same as stefan_step_kernel) ----
    float flux = 0.0f;

    // +X
    {
        int ni = stefan_voxel_idx(base, stefan_clamp(x + 1, 0, G - 1), y, z, G2, G);
        float T_n = enthalpy_to_temp(enthalpy_in[ni], ni, precomp, BG3);
        flux += precomp[1 * BG3 + idx] * (T_n - T_c);
    }
    // -X
    {
        int ni = stefan_voxel_idx(base, stefan_clamp(x - 1, 0, G - 1), y, z, G2, G);
        float T_n = enthalpy_to_temp(enthalpy_in[ni], ni, precomp, BG3);
        flux += precomp[2 * BG3 + idx] * (T_n - T_c);
    }
    // +Y
    {
        int ni = stefan_voxel_idx(base, x, stefan_clamp(y + 1, 0, G - 1), z, G2, G);
        float T_n = enthalpy_to_temp(enthalpy_in[ni], ni, precomp, BG3);
        flux += precomp[3 * BG3 + idx] * (T_n - T_c);
    }
    // -Y
    {
        int ni = stefan_voxel_idx(base, x, stefan_clamp(y - 1, 0, G - 1), z, G2, G);
        float T_n = enthalpy_to_temp(enthalpy_in[ni], ni, precomp, BG3);
        flux += precomp[4 * BG3 + idx] * (T_n - T_c);
    }
    // +Z
    {
        int ni = stefan_voxel_idx(base, x, y, stefan_clamp(z + 1, 0, G - 1), G2, G);
        float T_n = enthalpy_to_temp(enthalpy_in[ni], ni, precomp, BG3);
        flux += precomp[5 * BG3 + idx] * (T_n - T_c);
    }
    // -Z
    {
        int ni = stefan_voxel_idx(base, x, y, stefan_clamp(z - 1, 0, G - 1), G2, G);
        float T_n = enthalpy_to_temp(enthalpy_in[ni], ni, precomp, BG3);
        flux += precomp[6 * BG3 + idx] * (T_n - T_c);
    }

    // ---- Upwind advection of enthalpy ----
    // Enthalpy-porosity: effective velocity = v * f_liquid
    float f_c = liquid_fraction[idx];

    float u = vx[idx] * f_c;
    float v = vy[idx] * f_c;
    float w = vz[idx] * f_c;

    float advection = 0.0f;

    // X-direction upwind
    {
        int i_fwd = stefan_voxel_idx(base, stefan_clamp(x + 1, 0, G - 1), y, z, G2, G);
        int i_bwd = stefan_voxel_idx(base, stefan_clamp(x - 1, 0, G - 1), y, z, G2, G);
        float H_fwd = enthalpy_in[i_fwd];
        float H_bwd = enthalpy_in[i_bwd];
        // Upwind: if u > 0, use backward difference; if u < 0, use forward difference
        advection += (u > 0.0f) ? u * (H_c - H_bwd) : u * (H_fwd - H_c);
    }

    // Y-direction upwind
    {
        int i_fwd = stefan_voxel_idx(base, x, stefan_clamp(y + 1, 0, G - 1), z, G2, G);
        int i_bwd = stefan_voxel_idx(base, x, stefan_clamp(y - 1, 0, G - 1), z, G2, G);
        float H_fwd = enthalpy_in[i_fwd];
        float H_bwd = enthalpy_in[i_bwd];
        advection += (v > 0.0f) ? v * (H_c - H_bwd) : v * (H_fwd - H_c);
    }

    // Z-direction upwind
    {
        int i_fwd = stefan_voxel_idx(base, x, y, stefan_clamp(z + 1, 0, G - 1), G2, G);
        int i_bwd = stefan_voxel_idx(base, x, y, stefan_clamp(z - 1, 0, G - 1), G2, G);
        float H_fwd = enthalpy_in[i_fwd];
        float H_bwd = enthalpy_in[i_bwd];
        advection += (w > 0.0f) ? w * (H_c - H_bwd) : w * (H_fwd - H_c);
    }

    // Total enthalpy change: conduction - advection
    float total_flux = flux - advection;

    // Stability clamping
    float hd = fminf(fmaxf(total_flux * dt, -max_hd), max_hd);
    float H_new = fminf(fmaxf(H_c + hd, 0.0f), max_safe);
    enthalpy_out[idx] = H_new;
}

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
    cudaStream_t stream)
{
    int BG3 = B * G * G * G;
    if (BG3 <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid  = internal::divup(BG3, block);
    stefan_convection_step_kernel<<<grid, block, 0, stream>>>(
        enthalpy_in, enthalpy_out, precomp, vx, vy, vz,
        liquid_fraction, G, BG3, dt);
}

// ---- Full convection simulation ----

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
    float        dt,
    float        max_temp_delta,
    float        conduction_scale,
    cudaStream_t stream)
{
    int BG3 = B * G * G * G;
    if (BG3 <= 0 || n_steps <= 0) return;

    // Precompute conduction values (once)
    stefan_precompute(precomp, conductivity, msh, latent_heat,
                      T_solidus, T_liquidus, B, G,
                      max_temp_delta, conduction_scale, stream);

    // Double-buffer stepping
    float* src = enthalpy;
    float* dst = scratch;

    for (int step = 0; step < n_steps; step++) {
        // Update liquid fraction from current enthalpy (for porosity scaling)
        extract_liquid_fraction(src, precomp, liquid_fraction, B, G, stream);

        // Step with conduction + advection
        stefan_convection_step(src, dst, precomp, vx, vy, vz,
                               liquid_fraction, B, G, dt, stream);

        float* tmp = src;
        src = dst;
        dst = tmp;
    }

    // If n_steps is odd, result is in scratch — copy back to enthalpy.
    if (n_steps % 2 != 0) {
        cudaMemcpyAsync(enthalpy, scratch, BG3 * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
}

}} // namespace forge::sim
