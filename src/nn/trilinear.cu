// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Trilinear interpolation kernels for 3D grid sampling and scattering.

#include <forge/nn/trilinear.cuh>
#include "cuda_utils.cuh"

namespace forge { namespace nn {

// ============================================================
//  Helpers: trilinear weight computation
// ============================================================

// Compute floor indices and fractional offsets for trilinear interpolation.
// Positions are in [0, G-1] range, clamped at boundaries.
struct __device__ TrilinearCoords {
    int   ix0, iy0, iz0;   // floor indices
    int   ix1, iy1, iz1;   // ceil indices
    float dx,  dy,  dz;    // fractional offsets [0, 1)
};

__device__ __forceinline__ TrilinearCoords compute_coords(
    const float* __restrict__ positions, int i, int G)
{
    TrilinearCoords c;
    float x = positions[i * 3 + 0];
    float y = positions[i * 3 + 1];
    float z = positions[i * 3 + 2];

    // Clamp to [0, G-1]
    float gm1 = (float)(G - 1);
    x = fminf(fmaxf(x, 0.0f), gm1);
    y = fminf(fmaxf(y, 0.0f), gm1);
    z = fminf(fmaxf(z, 0.0f), gm1);

    c.ix0 = (int)floorf(x);
    c.iy0 = (int)floorf(y);
    c.iz0 = (int)floorf(z);
    c.ix1 = min(c.ix0 + 1, G - 1);
    c.iy1 = min(c.iy0 + 1, G - 1);
    c.iz1 = min(c.iz0 + 1, G - 1);

    c.dx = x - (float)c.ix0;
    c.dy = y - (float)c.iy0;
    c.dz = z - (float)c.iz0;

    return c;
}

// 3D index into [G, G, G] grid (row-major: x*G*G + y*G + z)
__device__ __forceinline__ int idx3d(int x, int y, int z, int G) {
    return x * G * G + y * G + z;
}

// ============================================================
//  Forward: trilinear sample (gather from grid)
// ============================================================

// Grid layout: [C, G, G, G, B] — for channel c, voxel (x,y,z), batch b:
//   grid[c * G*G*G * B + idx3d(x,y,z,G) * B + b]
// Output layout: [N, C, B] — for unit i, channel c, batch b:
//   output[i * C * B + c * B + b]

__global__ void trilinear_sample_kernel(
    const float* __restrict__ grid,
    const float* __restrict__ positions,
    float* __restrict__       output,
    int G, int N, int C, int B)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * B;
    if (tid >= total) return;

    int i = tid / B;  // unit index
    int b = tid % B;  // batch index

    TrilinearCoords co = compute_coords(positions, i, G);

    // 8 trilinear weights
    float w000 = (1 - co.dx) * (1 - co.dy) * (1 - co.dz);
    float w001 = (1 - co.dx) * (1 - co.dy) * co.dz;
    float w010 = (1 - co.dx) * co.dy       * (1 - co.dz);
    float w011 = (1 - co.dx) * co.dy       * co.dz;
    float w100 = co.dx       * (1 - co.dy) * (1 - co.dz);
    float w101 = co.dx       * (1 - co.dy) * co.dz;
    float w110 = co.dx       * co.dy       * (1 - co.dz);
    float w111 = co.dx       * co.dy       * co.dz;

    // 8 voxel indices in the [G, G, G] sub-grid
    int v000 = idx3d(co.ix0, co.iy0, co.iz0, G);
    int v001 = idx3d(co.ix0, co.iy0, co.iz1, G);
    int v010 = idx3d(co.ix0, co.iy1, co.iz0, G);
    int v011 = idx3d(co.ix0, co.iy1, co.iz1, G);
    int v100 = idx3d(co.ix1, co.iy0, co.iz0, G);
    int v101 = idx3d(co.ix1, co.iy0, co.iz1, G);
    int v110 = idx3d(co.ix1, co.iy1, co.iz0, G);
    int v111 = idx3d(co.ix1, co.iy1, co.iz1, G);

    int G3B = G * G * G * B;

    for (int c = 0; c < C; c++) {
        const float* gc = grid + c * G3B;  // channel slice

        float val = w000 * gc[v000 * B + b]
                  + w001 * gc[v001 * B + b]
                  + w010 * gc[v010 * B + b]
                  + w011 * gc[v011 * B + b]
                  + w100 * gc[v100 * B + b]
                  + w101 * gc[v101 * B + b]
                  + w110 * gc[v110 * B + b]
                  + w111 * gc[v111 * B + b];

        output[i * C * B + c * B + b] = val;
    }
}

void trilinear_sample(
    const float* grid, const float* positions, float* output,
    int G, int N, int C, int B, cudaStream_t stream)
{
    int total = N * B;
    if (total <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid_dim = internal::divup(total, block);
    trilinear_sample_kernel<<<grid_dim, block, 0, stream>>>(
        grid, positions, output, G, N, C, B);
}

// ============================================================
//  Forward: trilinear scatter (splat to grid)
// ============================================================

// Values layout: [N, B] — values[i * B + b]
// Grid layout: [G, G, G, B] — grid[idx3d(x,y,z,G) * B + b]

__global__ void trilinear_scatter_kernel(
    const float* __restrict__ values,
    const float* __restrict__ positions,
    float* __restrict__       grid,
    int G, int N, int B)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * B;
    if (tid >= total) return;

    int i = tid / B;
    int b = tid % B;

    TrilinearCoords co = compute_coords(positions, i, G);
    float val = values[i * B + b];

    // 8 trilinear weights
    float w000 = (1 - co.dx) * (1 - co.dy) * (1 - co.dz);
    float w001 = (1 - co.dx) * (1 - co.dy) * co.dz;
    float w010 = (1 - co.dx) * co.dy       * (1 - co.dz);
    float w011 = (1 - co.dx) * co.dy       * co.dz;
    float w100 = co.dx       * (1 - co.dy) * (1 - co.dz);
    float w101 = co.dx       * (1 - co.dy) * co.dz;
    float w110 = co.dx       * co.dy       * (1 - co.dz);
    float w111 = co.dx       * co.dy       * co.dz;

    // Scatter to 8 corners via atomicAdd
    atomicAdd(&grid[idx3d(co.ix0, co.iy0, co.iz0, G) * B + b], w000 * val);
    atomicAdd(&grid[idx3d(co.ix0, co.iy0, co.iz1, G) * B + b], w001 * val);
    atomicAdd(&grid[idx3d(co.ix0, co.iy1, co.iz0, G) * B + b], w010 * val);
    atomicAdd(&grid[idx3d(co.ix0, co.iy1, co.iz1, G) * B + b], w011 * val);
    atomicAdd(&grid[idx3d(co.ix1, co.iy0, co.iz0, G) * B + b], w100 * val);
    atomicAdd(&grid[idx3d(co.ix1, co.iy0, co.iz1, G) * B + b], w101 * val);
    atomicAdd(&grid[idx3d(co.ix1, co.iy1, co.iz0, G) * B + b], w110 * val);
    atomicAdd(&grid[idx3d(co.ix1, co.iy1, co.iz1, G) * B + b], w111 * val);
}

void trilinear_scatter(
    const float* values, const float* positions, float* grid,
    int G, int N, int B, cudaStream_t stream)
{
    int total = N * B;
    if (total <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid_dim = internal::divup(total, block);
    trilinear_scatter_kernel<<<grid_dim, block, 0, stream>>>(
        values, positions, grid, G, N, B);
}

// ============================================================
//  Backward: position gradients from sampling
// ============================================================

// For position p_i = (x, y, z) and grid value at corner (ix, iy, iz):
//   dw/dx varies per corner. For example:
//     w000 = (1-dx)(1-dy)(1-dz)  =>  dw000/dx = -(1-dy)(1-dz)
//     w100 = dx*(1-dy)(1-dz)     =>  dw100/dx = +(1-dy)(1-dz)
//   The x-gradient for unit i = sum over corners and channels:
//     d_pos[i, 0] += sum_c sum_b d_output[i,c,b] * sum_corners(dw/dx * grid[corner,c,b])

__global__ void trilinear_sample_pos_grad_kernel(
    const float* __restrict__ d_output,
    const float* __restrict__ grid,
    const float* __restrict__ positions,
    float* __restrict__       d_positions,
    int G, int N, int C, int B)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * B;
    if (tid >= total) return;

    int i = tid / B;
    int b = tid % B;

    TrilinearCoords co = compute_coords(positions, i, G);

    int v000 = idx3d(co.ix0, co.iy0, co.iz0, G);
    int v001 = idx3d(co.ix0, co.iy0, co.iz1, G);
    int v010 = idx3d(co.ix0, co.iy1, co.iz0, G);
    int v011 = idx3d(co.ix0, co.iy1, co.iz1, G);
    int v100 = idx3d(co.ix1, co.iy0, co.iz0, G);
    int v101 = idx3d(co.ix1, co.iy0, co.iz1, G);
    int v110 = idx3d(co.ix1, co.iy1, co.iz0, G);
    int v111 = idx3d(co.ix1, co.iy1, co.iz1, G);

    int G3B = G * G * G * B;

    float grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;

    for (int c = 0; c < C; c++) {
        float do_icb = d_output[i * C * B + c * B + b];
        const float* gc = grid + c * G3B;

        float g000 = gc[v000 * B + b];
        float g001 = gc[v001 * B + b];
        float g010 = gc[v010 * B + b];
        float g011 = gc[v011 * B + b];
        float g100 = gc[v100 * B + b];
        float g101 = gc[v101 * B + b];
        float g110 = gc[v110 * B + b];
        float g111 = gc[v111 * B + b];

        // dw/dx: corner with ix0 gets -(1-dy or dy)(1-dz or dz),
        //        corner with ix1 gets +(1-dy or dy)(1-dz or dz)
        float dw_dx_000 = -(1 - co.dy) * (1 - co.dz);
        float dw_dx_001 = -(1 - co.dy) * co.dz;
        float dw_dx_010 = -co.dy       * (1 - co.dz);
        float dw_dx_011 = -co.dy       * co.dz;
        float dw_dx_100 =  (1 - co.dy) * (1 - co.dz);
        float dw_dx_101 =  (1 - co.dy) * co.dz;
        float dw_dx_110 =  co.dy       * (1 - co.dz);
        float dw_dx_111 =  co.dy       * co.dz;

        grad_x += do_icb * (dw_dx_000 * g000 + dw_dx_001 * g001
                           + dw_dx_010 * g010 + dw_dx_011 * g011
                           + dw_dx_100 * g100 + dw_dx_101 * g101
                           + dw_dx_110 * g110 + dw_dx_111 * g111);

        // dw/dy
        float dw_dy_000 = -(1 - co.dx) * (1 - co.dz);
        float dw_dy_001 = -(1 - co.dx) * co.dz;
        float dw_dy_010 =  (1 - co.dx) * (1 - co.dz);
        float dw_dy_011 =  (1 - co.dx) * co.dz;
        float dw_dy_100 = -co.dx       * (1 - co.dz);
        float dw_dy_101 = -co.dx       * co.dz;
        float dw_dy_110 =  co.dx       * (1 - co.dz);
        float dw_dy_111 =  co.dx       * co.dz;

        grad_y += do_icb * (dw_dy_000 * g000 + dw_dy_001 * g001
                           + dw_dy_010 * g010 + dw_dy_011 * g011
                           + dw_dy_100 * g100 + dw_dy_101 * g101
                           + dw_dy_110 * g110 + dw_dy_111 * g111);

        // dw/dz
        float dw_dz_000 = -(1 - co.dx) * (1 - co.dy);
        float dw_dz_001 =  (1 - co.dx) * (1 - co.dy);
        float dw_dz_010 = -(1 - co.dx) * co.dy;
        float dw_dz_011 =  (1 - co.dx) * co.dy;
        float dw_dz_100 = -co.dx       * (1 - co.dy);
        float dw_dz_101 =  co.dx       * (1 - co.dy);
        float dw_dz_110 = -co.dx       * co.dy;
        float dw_dz_111 =  co.dx       * co.dy;

        grad_z += do_icb * (dw_dz_000 * g000 + dw_dz_001 * g001
                           + dw_dz_010 * g010 + dw_dz_011 * g011
                           + dw_dz_100 * g100 + dw_dz_101 * g101
                           + dw_dz_110 * g110 + dw_dz_111 * g111);
    }

    atomicAdd(&d_positions[i * 3 + 0], grad_x);
    atomicAdd(&d_positions[i * 3 + 1], grad_y);
    atomicAdd(&d_positions[i * 3 + 2], grad_z);
}

void trilinear_sample_pos_grad(
    const float* d_output, const float* grid, const float* positions,
    float* d_positions, int G, int N, int C, int B, cudaStream_t stream)
{
    int total = N * B;
    if (total <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid_dim = internal::divup(total, block);
    trilinear_sample_pos_grad_kernel<<<grid_dim, block, 0, stream>>>(
        d_output, grid, positions, d_positions, G, N, C, B);
}

// ============================================================
//  Backward: scatter backward (d_values + position gradients)
// ============================================================

// Scatter backward for values is just a trilinear gather (dual operation):
//   d_values[i,b] = sum_corners w(pos[i], corner) * d_grid[corner, b]
//
// Position gradients use the same dw/dx pattern as sample_pos_grad,
// but multiplied by values[i,b] * d_grid[corner,b] instead of grid values.

__global__ void trilinear_scatter_backward_kernel(
    const float* __restrict__ d_grid,
    const float* __restrict__ values,
    const float* __restrict__ positions,
    float* __restrict__       d_values,
    float* __restrict__       d_positions,
    int G, int N, int B)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * B;
    if (tid >= total) return;

    int i = tid / B;
    int b = tid % B;

    TrilinearCoords co = compute_coords(positions, i, G);
    float val = values[i * B + b];

    // 8 trilinear weights
    float w000 = (1 - co.dx) * (1 - co.dy) * (1 - co.dz);
    float w001 = (1 - co.dx) * (1 - co.dy) * co.dz;
    float w010 = (1 - co.dx) * co.dy       * (1 - co.dz);
    float w011 = (1 - co.dx) * co.dy       * co.dz;
    float w100 = co.dx       * (1 - co.dy) * (1 - co.dz);
    float w101 = co.dx       * (1 - co.dy) * co.dz;
    float w110 = co.dx       * co.dy       * (1 - co.dz);
    float w111 = co.dx       * co.dy       * co.dz;

    // Read d_grid at 8 corners
    float dg000 = d_grid[idx3d(co.ix0, co.iy0, co.iz0, G) * B + b];
    float dg001 = d_grid[idx3d(co.ix0, co.iy0, co.iz1, G) * B + b];
    float dg010 = d_grid[idx3d(co.ix0, co.iy1, co.iz0, G) * B + b];
    float dg011 = d_grid[idx3d(co.ix0, co.iy1, co.iz1, G) * B + b];
    float dg100 = d_grid[idx3d(co.ix1, co.iy0, co.iz0, G) * B + b];
    float dg101 = d_grid[idx3d(co.ix1, co.iy0, co.iz1, G) * B + b];
    float dg110 = d_grid[idx3d(co.ix1, co.iy1, co.iz0, G) * B + b];
    float dg111 = d_grid[idx3d(co.ix1, co.iy1, co.iz1, G) * B + b];

    // d_values: gather (dual of scatter)
    d_values[i * B + b] = w000 * dg000 + w001 * dg001
                        + w010 * dg010 + w011 * dg011
                        + w100 * dg100 + w101 * dg101
                        + w110 * dg110 + w111 * dg111;

    // Position gradients: dL/dpos through scatter weights
    // dw/dx for each corner (same pattern as sample_pos_grad)
    float grad_x = val * (
        -(1 - co.dy) * (1 - co.dz) * dg000
        -(1 - co.dy) * co.dz       * dg001
        -co.dy       * (1 - co.dz) * dg010
        -co.dy       * co.dz       * dg011
        +(1 - co.dy) * (1 - co.dz) * dg100
        +(1 - co.dy) * co.dz       * dg101
        +co.dy       * (1 - co.dz) * dg110
        +co.dy       * co.dz       * dg111);

    float grad_y = val * (
        -(1 - co.dx) * (1 - co.dz) * dg000
        -(1 - co.dx) * co.dz       * dg001
        +(1 - co.dx) * (1 - co.dz) * dg010
        +(1 - co.dx) * co.dz       * dg011
        -co.dx       * (1 - co.dz) * dg100
        -co.dx       * co.dz       * dg101
        +co.dx       * (1 - co.dz) * dg110
        +co.dx       * co.dz       * dg111);

    float grad_z = val * (
        -(1 - co.dx) * (1 - co.dy) * dg000
        +(1 - co.dx) * (1 - co.dy) * dg001
        -(1 - co.dx) * co.dy       * dg010
        +(1 - co.dx) * co.dy       * dg011
        -co.dx       * (1 - co.dy) * dg100
        +co.dx       * (1 - co.dy) * dg101
        -co.dx       * co.dy       * dg110
        +co.dx       * co.dy       * dg111);

    atomicAdd(&d_positions[i * 3 + 0], grad_x);
    atomicAdd(&d_positions[i * 3 + 1], grad_y);
    atomicAdd(&d_positions[i * 3 + 2], grad_z);
}

void trilinear_scatter_backward(
    const float* d_grid, const float* values, const float* positions,
    float* d_values, float* d_positions, int G, int N, int B, cudaStream_t stream)
{
    int total = N * B;
    if (total <= 0) return;
    int block = internal::kDefaultBlockSize;
    int grid_dim = internal::divup(total, block);
    trilinear_scatter_backward_kernel<<<grid_dim, block, 0, stream>>>(
        d_grid, values, positions, d_values, d_positions, G, N, B);
}

}} // namespace forge::nn
