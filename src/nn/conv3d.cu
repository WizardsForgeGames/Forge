// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Conv3d implementation via vol2col/col2vol + GEMM.
// Extends the Conv2d pattern (im2col) to 3D volumes.
//
// Forward:  vol2col(input) → col, then weight @ col → output
// Backward: col2vol for d_input, vol2col + gemm for d_weight
#include <forge/nn/conv3d.cuh>
#include <forge/gemm/fp32_tiled.cuh>
#include <forge/error.h>

namespace forge { namespace nn {

// =====================================================================
// Output dimension helper
// =====================================================================

static inline int conv_out_size(int in_size, int pad, int dil, int kernel, int stride) {
    return (in_size + 2 * pad - dil * (kernel - 1) - 1) / stride + 1;
}

// =====================================================================
// vol2col kernel: unroll 3D input patches into column matrix
// =====================================================================
//
// For a single sample+group, transforms input[C_in, D, H, W] into
// col[C_in*kD*kH*kW, D_out*H_out*W_out] where each column is a flattened 3D patch.

static __global__ void vol2col_kernel(
    const float* __restrict__ input,   // [C_in, D, H, W]
    float*       __restrict__ col,     // [C_in*kD*kH*kW, D_out*H_out*W_out]
    int C_in, int D, int H, int W,
    int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD, int dilH, int dilW,
    int D_out, int H_out, int W_out)
{
    int spatial_out = D_out * H_out * W_out;
    int col_rows = C_in * kD * kH * kW;
    int total = col_rows * spatial_out;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x)
    {
        // Decompose linear index into (c_kd_kh_kw, d_out, h_out, w_out)
        int w_out = idx % W_out;
        int tmp   = idx / W_out;
        int h_out = tmp % H_out;
        tmp       = tmp / H_out;
        int d_out = tmp % D_out;
        int c_kdkhkw = tmp / D_out;

        // Decompose c_kdkhkw into (c, kd, kh, kw)
        int kw = c_kdkhkw % kW;
        int tmp2 = c_kdkhkw / kW;
        int kh = tmp2 % kH;
        tmp2 = tmp2 / kH;
        int kd = tmp2 % kD;
        int c  = tmp2 / kD;

        // Input position
        int d_in = d_out * strideD + kd * dilD - padD;
        int h_in = h_out * strideH + kh * dilH - padH;
        int w_in = w_out * strideW + kw * dilW - padW;

        // col is [col_rows, spatial_out] row-major
        int col_idx = c_kdkhkw * spatial_out + d_out * (H_out * W_out) + h_out * W_out + w_out;

        if (d_in >= 0 && d_in < D &&
            h_in >= 0 && h_in < H &&
            w_in >= 0 && w_in < W) {
            col[col_idx] = input[((c * D + d_in) * H + h_in) * W + w_in];
        } else {
            col[col_idx] = 0.0f;  // zero-padding
        }
    }
}

// =====================================================================
// col2vol kernel: scatter-add column matrix back to 3D input
// =====================================================================

static __global__ void col2vol_kernel(
    const float* __restrict__ col,     // [C_in*kD*kH*kW, D_out*H_out*W_out]
    float*       __restrict__ input,   // [C_in, D, H, W]  (accumulates via atomicAdd)
    int C_in, int D, int H, int W,
    int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD, int dilH, int dilW,
    int D_out, int H_out, int W_out)
{
    int spatial_out = D_out * H_out * W_out;
    int col_rows = C_in * kD * kH * kW;
    int total = col_rows * spatial_out;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x)
    {
        int w_out = idx % W_out;
        int tmp   = idx / W_out;
        int h_out = tmp % H_out;
        tmp       = tmp / H_out;
        int d_out = tmp % D_out;
        int c_kdkhkw = tmp / D_out;

        int kw = c_kdkhkw % kW;
        int tmp2 = c_kdkhkw / kW;
        int kh = tmp2 % kH;
        tmp2 = tmp2 / kH;
        int kd = tmp2 % kD;
        int c  = tmp2 / kD;

        int d_in = d_out * strideD + kd * dilD - padD;
        int h_in = h_out * strideH + kh * dilH - padH;
        int w_in = w_out * strideW + kw * dilW - padW;

        if (d_in >= 0 && d_in < D &&
            h_in >= 0 && h_in < H &&
            w_in >= 0 && w_in < W) {
            float val = col[c_kdkhkw * spatial_out + d_out * (H_out * W_out) + h_out * W_out + w_out];
            atomicAdd(&input[((c * D + d_in) * H + h_in) * W + w_in], val);
        }
    }
}

// =====================================================================
// Bias kernels
// =====================================================================

static __global__ void conv3d_add_bias_kernel(
    float*       __restrict__ output,
    const float* __restrict__ bias,
    int total,      // N * C_out * spatial
    int C_out,
    int spatial)    // D_out * H_out * W_out
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = (idx / spatial) % C_out;
    output[idx] += bias[c];
}

static __global__ void conv3d_bias_backward_kernel(
    const float* __restrict__ d_output,  // [N, C_out, D_out, H_out, W_out]
    float*       __restrict__ d_bias,    // [C_out]
    int N, int C_out, int spatial)       // spatial = D_out * H_out * W_out
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C_out) return;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        const float* slice = d_output + (n * C_out + c) * spatial;
        for (int s = 0; s < spatial; ++s) {
            sum += slice[s];
        }
    }
    d_bias[c] = sum;
}

// =====================================================================
// conv3d_forward
// =====================================================================

void conv3d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    float*       workspace,
    int N, int C_in, int D, int H, int W,
    int C_out, int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD, int dilH, int dilW,
    int groups,
    cudaStream_t stream)
{
    int D_out = conv_out_size(D, padD, dilD, kD, strideD);
    int H_out = conv_out_size(H, padH, dilH, kH, strideH);
    int W_out = conv_out_size(W, padW, dilW, kW, strideW);

    int C_in_per_group  = C_in  / groups;
    int C_out_per_group = C_out / groups;
    int spatial_out = D_out * H_out * W_out;

    int col_rows = C_in_per_group * kD * kH * kW;
    int col_cols = spatial_out;
    int col_size = col_rows * col_cols;

    // Workspace management
    float* col_buf = workspace;
    bool own_workspace = (workspace == nullptr);
    if (own_workspace) {
        FORGE_CHECK(cudaMalloc(&col_buf, col_size * sizeof(float)));
    }

    constexpr int BLOCK = 256;
    // Cap grid size for very large vol2col operations
    int vol2col_grid = min((col_size + BLOCK - 1) / BLOCK, 65535);

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            const float* input_slice  = input + ((size_t)n * C_in + g * C_in_per_group) * D * H * W;
            float*       output_slice = output + ((size_t)n * C_out + g * C_out_per_group) * spatial_out;
            const float* weight_slice = weight + (size_t)g * C_out_per_group * C_in_per_group * kD * kH * kW;

            // 1. vol2col: input_slice[C_in_per_group, D, H, W] → col_buf[col_rows, col_cols]
            vol2col_kernel<<<vol2col_grid, BLOCK, 0, stream>>>(
                input_slice, col_buf,
                C_in_per_group, D, H, W,
                kD, kH, kW,
                padD, padH, padW,
                strideD, strideH, strideW,
                dilD, dilH, dilW,
                D_out, H_out, W_out);

            // 2. GEMM: output[C_out_per_group, col_cols] = weight[C_out_per_group, col_rows] @ col[col_rows, col_cols]
            gemm::fp32_gemm(
                gemm::Op::N, gemm::Op::N,
                C_out_per_group, col_cols, col_rows,
                1.0f,
                weight_slice, col_rows,
                col_buf, col_cols,
                0.0f,
                output_slice, col_cols,
                nullptr,
                stream);
        }
    }

    // Add bias
    if (bias) {
        int total = N * C_out * spatial_out;
        int grid = (total + BLOCK - 1) / BLOCK;
        conv3d_add_bias_kernel<<<grid, BLOCK, 0, stream>>>(output, bias, total, C_out, spatial_out);
    }

    if (own_workspace) {
        FORGE_CHECK(cudaFree(col_buf));
    }
}

// =====================================================================
// conv3d_backward_data
// =====================================================================

void conv3d_backward_data(
    const float* d_output,
    const float* weight,
    float*       d_input,
    float*       workspace,
    int N, int C_in, int D, int H, int W,
    int C_out, int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD, int dilH, int dilW,
    int groups,
    cudaStream_t stream)
{
    int D_out = conv_out_size(D, padD, dilD, kD, strideD);
    int H_out = conv_out_size(H, padH, dilH, kH, strideH);
    int W_out = conv_out_size(W, padW, dilW, kW, strideW);

    int C_in_per_group  = C_in  / groups;
    int C_out_per_group = C_out / groups;
    int spatial_out = D_out * H_out * W_out;

    int col_rows = C_in_per_group * kD * kH * kW;
    int col_cols = spatial_out;
    int col_size = col_rows * col_cols;

    float* col_buf = workspace;
    bool own_workspace = (workspace == nullptr);
    if (own_workspace) {
        FORGE_CHECK(cudaMalloc(&col_buf, col_size * sizeof(float)));
    }

    // Zero d_input (col2vol does atomic adds)
    FORGE_CHECK(cudaMemsetAsync(d_input, 0,
        (size_t)N * C_in * D * H * W * sizeof(float), stream));

    constexpr int BLOCK = 256;
    int col2vol_grid = min((col_size + BLOCK - 1) / BLOCK, 65535);

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            const float* d_output_slice = d_output + ((size_t)n * C_out + g * C_out_per_group) * spatial_out;
            float*       d_input_slice  = d_input + ((size_t)n * C_in + g * C_in_per_group) * D * H * W;
            const float* weight_slice   = weight + (size_t)g * C_out_per_group * C_in_per_group * kD * kH * kW;

            // 1. GEMM: col_buf = weight^T @ d_output
            // col[col_rows, col_cols] = weight^T[col_rows, C_out_per_group] @ d_output[C_out_per_group, col_cols]
            gemm::fp32_gemm(
                gemm::Op::T, gemm::Op::N,
                col_rows, col_cols, C_out_per_group,
                1.0f,
                weight_slice, col_rows,
                d_output_slice, col_cols,
                0.0f,
                col_buf, col_cols,
                nullptr,
                stream);

            // 2. col2vol: col_buf → d_input_slice (atomic add)
            col2vol_kernel<<<col2vol_grid, BLOCK, 0, stream>>>(
                col_buf, d_input_slice,
                C_in_per_group, D, H, W,
                kD, kH, kW,
                padD, padH, padW,
                strideD, strideH, strideW,
                dilD, dilH, dilW,
                D_out, H_out, W_out);
        }
    }

    if (own_workspace) {
        FORGE_CHECK(cudaFree(col_buf));
    }
}

// =====================================================================
// conv3d_backward_weight
// =====================================================================

void conv3d_backward_weight(
    const float* d_output,
    const float* input,
    float*       d_weight,
    float*       d_bias,
    float*       workspace,
    int N, int C_in, int D, int H, int W,
    int C_out, int kD, int kH, int kW,
    int padD, int padH, int padW,
    int strideD, int strideH, int strideW,
    int dilD, int dilH, int dilW,
    int groups,
    float beta,
    cudaStream_t stream)
{
    int D_out = conv_out_size(D, padD, dilD, kD, strideD);
    int H_out = conv_out_size(H, padH, dilH, kH, strideH);
    int W_out = conv_out_size(W, padW, dilW, kW, strideW);

    int C_in_per_group  = C_in  / groups;
    int C_out_per_group = C_out / groups;
    int spatial_out = D_out * H_out * W_out;

    int col_rows = C_in_per_group * kD * kH * kW;
    int col_cols = spatial_out;
    int col_size = col_rows * col_cols;

    float* col_buf = workspace;
    bool own_workspace = (workspace == nullptr);
    if (own_workspace) {
        FORGE_CHECK(cudaMalloc(&col_buf, col_size * sizeof(float)));
    }

    constexpr int BLOCK = 256;
    int vol2col_grid = min((col_size + BLOCK - 1) / BLOCK, 65535);

    for (int n = 0; n < N; ++n) {
        float sample_beta = (n == 0) ? beta : 1.0f;

        for (int g = 0; g < groups; ++g) {
            const float* input_slice    = input + ((size_t)n * C_in + g * C_in_per_group) * D * H * W;
            const float* d_output_slice = d_output + ((size_t)n * C_out + g * C_out_per_group) * spatial_out;
            float*       d_weight_slice = d_weight + (size_t)g * C_out_per_group * C_in_per_group * kD * kH * kW;

            // 1. vol2col: input → col_buf
            vol2col_kernel<<<vol2col_grid, BLOCK, 0, stream>>>(
                input_slice, col_buf,
                C_in_per_group, D, H, W,
                kD, kH, kW,
                padD, padH, padW,
                strideD, strideH, strideW,
                dilD, dilH, dilW,
                D_out, H_out, W_out);

            // 2. GEMM: d_weight += d_output @ col^T
            // d_weight[C_out_per_group, col_rows] = d_output[C_out_per_group, col_cols] @ col^T[col_cols, col_rows]
            gemm::fp32_gemm(
                gemm::Op::N, gemm::Op::T,
                C_out_per_group, col_rows, col_cols,
                1.0f,
                d_output_slice, col_cols,
                col_buf, col_cols,
                sample_beta,
                d_weight_slice, col_rows,
                nullptr,
                stream);
        }
    }

    // Bias gradient
    if (d_bias) {
        int grid = (C_out + BLOCK - 1) / BLOCK;
        conv3d_bias_backward_kernel<<<grid, BLOCK, 0, stream>>>(
            d_output, d_bias, N, C_out, spatial_out);
    }

    if (own_workspace) {
        FORGE_CHECK(cudaFree(col_buf));
    }
}

}} // namespace forge::nn
