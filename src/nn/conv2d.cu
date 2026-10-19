// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Conv2d implementation via batched im2col/col2im + GEMM.
//
// Samples are processed in chunks of nb: im2col unrolls the whole chunk into one column matrix
// col[C_in/g * kH * kW, nb * H_out * W_out] (column = n * HW + s), so each group is ONE GEMM per chunk instead of one
// per sample (the per-sample loop cost 197 ms a step against cuDNN's 2 ms on a 4-layer encoder at N = 360).
// The GEMM result is [C_out, nb * HW]; permute kernels move it to and from NCHW (bias fused into the forward one).
// The weight gradient reduces over nb * HW, a long K with a small output, so it runs split-K.
// Workspaces are stream-ordered (cudaMallocAsync): no device sync per call.
#include <forge/nn/conv2d.cuh>
#include <forge/gemm/fp32_tiled.cuh>
#include <forge/autograd.h>
#include <forge/error.h>

namespace forge { namespace nn {

// =====================================================================
// Output dimension helper
// =====================================================================

static inline int conv_out_size(int in_size, int pad, int dil, int kernel, int stride) {
    return (in_size + 2 * pad - dil * (kernel - 1) - 1) / stride + 1;
}

// Workspace budget for one chunk (floats): col + the [C_out, nb * HW] staging matrix
static constexpr long long kChunkFloats = 32ll << 20;   // 128 MB

static int chunk_samples(int N, int col_rows, int C_out, int HW) {
    long long per = (long long)HW * (col_rows + C_out);
    long long nb = kChunkFloats / (per > 0 ? per : 1);
    if (nb < 1) nb = 1;
    if (nb > N) nb = N;
    // keep every kernel's flat index inside int
    while (nb > 1 && (long long)nb * HW * (col_rows > C_out ? col_rows : C_out) >= (1ll << 31)) nb /= 2;
    return (int)nb;
}

// =====================================================================
// im2col / col2im over a chunk of samples
// =====================================================================
//
// col[row, n * HW + s], row = (c, kh, kw) over channels [c0, c0 + Cg) of samples n0 .. n0 + nb.
// One thread per column element; consecutive threads walk s, so col writes coalesce.

static __global__ void im2col_batched_kernel(
    const float* __restrict__ input,   // [N, C_in, H, W], already offset to sample n0
    float*       __restrict__ col,     // [Cg*kH*kW, nb*HW]
    int nb, int C_in, int c0, int Cg, int H, int W,
    int kH, int kW, int padH, int padW, int strideH, int strideW, int dilH, int dilW,
    int H_out, int W_out)
{
    int HW = H_out * W_out;
    int total = Cg * kH * kW * nb * HW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int s   = idx % HW;
    int n   = (idx / HW) % nb;
    int row = idx / (HW * nb);
    int w_out = s % W_out, h_out = s / W_out;
    int kw = row % kW, kh = (row / kW) % kH, c = row / (kW * kH);

    int h_in = h_out * strideH + kh * dilH - padH;
    int w_in = w_out * strideW + kw * dilW - padW;
    float v = 0.0f;
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
        v = input[((size_t)n * C_in + c0 + c) * H * W + h_in * W + w_in];
    col[idx] = v;
}

// Scatter-add of the column matrix back to d_input (several output positions read one input position).
static __global__ void col2im_batched_kernel(
    const float* __restrict__ col,     // [Cg*kH*kW, nb*HW]
    float*       __restrict__ d_input, // [N, C_in, H, W], offset to sample n0 (accumulates)
    int nb, int C_in, int c0, int Cg, int H, int W,
    int kH, int kW, int padH, int padW, int strideH, int strideW, int dilH, int dilW,
    int H_out, int W_out)
{
    int HW = H_out * W_out;
    int total = Cg * kH * kW * nb * HW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int s   = idx % HW;
    int n   = (idx / HW) % nb;
    int row = idx / (HW * nb);
    int w_out = s % W_out, h_out = s / W_out;
    int kw = row % kW, kh = (row / kW) % kH, c = row / (kW * kH);

    int h_in = h_out * strideH + kh * dilH - padH;
    int w_in = w_out * strideW + kw * dilW - padW;
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
        atomicAdd(&d_input[((size_t)n * C_in + c0 + c) * H * W + h_in * W + w_in], col[idx]);
}

// =====================================================================
// Layout permutes: [C, nb*HW] <-> [nb, C, HW]
// =====================================================================

// out[n, c, s] = tmp[c, n*HW + s] (+ bias[c])
static __global__ void cnhw_to_nchw_kernel(
    const float* __restrict__ tmp, const float* __restrict__ bias, float* __restrict__ out,
    int nb, int C, int HW)
{
    int total = nb * C * HW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int s = idx % HW, c = (idx / HW) % C, n = idx / (HW * C);
    float v = tmp[(size_t)c * nb * HW + n * HW + s];
    out[idx] = bias ? v + bias[c] : v;
}

// tmp[c, n*HW + s] = in[n, c, s]
static __global__ void nchw_to_cnhw_kernel(
    const float* __restrict__ in, float* __restrict__ tmp, int nb, int C, int HW)
{
    int total = nb * C * HW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int s = idx % HW, n = (idx / HW) % nb, c = idx / (HW * nb);
    tmp[idx] = in[((size_t)n * C + c) * HW + s];
}

// =====================================================================
// Bias gradient: d_bias[c] = sum over (N, H_out, W_out) of d_output, one block a channel
// =====================================================================

static __global__ void conv2d_bias_backward_kernel(
    const float* __restrict__ d_output,  // [N, C_out, H_out, W_out]
    float*       __restrict__ d_bias,    // [C_out]
    int N, int C_out, int spatial)
{
    __shared__ float part[256];
    int c = blockIdx.x;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N * spatial; i += blockDim.x) {
        int n = i / spatial, s = i % spatial;
        sum += d_output[((size_t)n * C_out + c) * spatial + s];
    }
    part[threadIdx.x] = sum;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) part[threadIdx.x] += part[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) d_bias[c] = part[0];
}

static inline int blocks_for(long long n, int block) { return (int)((n + block - 1) / block); }

// =====================================================================
// conv2d_forward
// =====================================================================

void conv2d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW,
    int groups,
    cudaStream_t stream)
{
    int H_out = conv_out_size(H, padH, dilH, kH, strideH);
    int W_out = conv_out_size(W, padW, dilW, kW, strideW);
    int HW = H_out * W_out;
    int Cig = C_in / groups, Cog = C_out / groups;
    int col_rows = Cig * kH * kW;
    int nb_max = chunk_samples(N, col_rows, C_out, HW);

    float *col = nullptr, *tmp = nullptr;
    FORGE_CHECK(cudaMallocAsync(&col, (size_t)col_rows * nb_max * HW * sizeof(float), stream));
    FORGE_CHECK(cudaMallocAsync(&tmp, (size_t)C_out * nb_max * HW * sizeof(float), stream));
    const int block = 256;

    for (int n0 = 0; n0 < N; n0 += nb_max) {
        int nb = (N - n0 < nb_max) ? N - n0 : nb_max;
        int cols = nb * HW;
        for (int g = 0; g < groups; ++g) {
            im2col_batched_kernel<<<blocks_for((long long)col_rows * cols, block), block, 0, stream>>>(
                input + (size_t)n0 * C_in * H * W, col, nb, C_in, g * Cig, Cig, H, W,
                kH, kW, padH, padW, strideH, strideW, dilH, dilW, H_out, W_out);
            // tmp[g rows] [Cog, cols] = W_g [Cog, col_rows] @ col [col_rows, cols]
            gemm::fp32_gemm(gemm::Op::N, gemm::Op::N, Cog, cols, col_rows, 1.0f,
                            weight + (size_t)g * Cog * col_rows, col_rows,
                            col, cols, 0.0f,
                            tmp + (size_t)g * Cog * cols, cols, nullptr, stream);
        }
        cnhw_to_nchw_kernel<<<blocks_for((long long)C_out * cols, block), block, 0, stream>>>(
            tmp, bias, output + (size_t)n0 * C_out * HW, nb, C_out, HW);
    }

    FORGE_CHECK(cudaFreeAsync(col, stream));
    FORGE_CHECK(cudaFreeAsync(tmp, stream));
}

// =====================================================================
// conv2d_backward_data
// =====================================================================

void conv2d_backward_data(
    const float* d_output,
    const float* weight,
    float*       d_input,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW,
    int groups,
    cudaStream_t stream)
{
    int H_out = conv_out_size(H, padH, dilH, kH, strideH);
    int W_out = conv_out_size(W, padW, dilW, kW, strideW);
    int HW = H_out * W_out;
    int Cig = C_in / groups, Cog = C_out / groups;
    int col_rows = Cig * kH * kW;
    int nb_max = chunk_samples(N, col_rows, C_out, HW);

    float *col = nullptr, *dtmp = nullptr;
    FORGE_CHECK(cudaMallocAsync(&col, (size_t)col_rows * nb_max * HW * sizeof(float), stream));
    FORGE_CHECK(cudaMallocAsync(&dtmp, (size_t)C_out * nb_max * HW * sizeof(float), stream));
    // col2im accumulates
    FORGE_CHECK(cudaMemsetAsync(d_input, 0, (size_t)N * C_in * H * W * sizeof(float), stream));
    const int block = 256;

    for (int n0 = 0; n0 < N; n0 += nb_max) {
        int nb = (N - n0 < nb_max) ? N - n0 : nb_max;
        int cols = nb * HW;
        nchw_to_cnhw_kernel<<<blocks_for((long long)C_out * cols, block), block, 0, stream>>>(
            d_output + (size_t)n0 * C_out * HW, dtmp, nb, C_out, HW);
        for (int g = 0; g < groups; ++g) {
            // col [col_rows, cols] = W_g^T [col_rows, Cog] @ dtmp[g rows] [Cog, cols]
            gemm::fp32_gemm(gemm::Op::T, gemm::Op::N, col_rows, cols, Cog, 1.0f,
                            weight + (size_t)g * Cog * col_rows, col_rows,
                            dtmp + (size_t)g * Cog * cols, cols, 0.0f,
                            col, cols, nullptr, stream);
            col2im_batched_kernel<<<blocks_for((long long)col_rows * cols, block), block, 0, stream>>>(
                col, d_input + (size_t)n0 * C_in * H * W, nb, C_in, g * Cig, Cig, H, W,
                kH, kW, padH, padW, strideH, strideW, dilH, dilW, H_out, W_out);
        }
    }

    FORGE_CHECK(cudaFreeAsync(col, stream));
    FORGE_CHECK(cudaFreeAsync(dtmp, stream));
}

// =====================================================================
// conv2d_backward_weight
// =====================================================================

void conv2d_backward_weight(
    const float* d_output,
    const float* input,
    float*       d_weight,
    float*       d_bias,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW,
    int groups,
    float beta,
    cudaStream_t stream)
{
    int H_out = conv_out_size(H, padH, dilH, kH, strideH);
    int W_out = conv_out_size(W, padW, dilW, kW, strideW);
    int HW = H_out * W_out;
    int Cig = C_in / groups, Cog = C_out / groups;
    int col_rows = Cig * kH * kW;
    int nb_max = chunk_samples(N, col_rows, C_out, HW);

    // split-K over the chunk's nb * HW: aim for ~4 K-steps of 1024+ a slice, at most 16 slices
    int k_max = nb_max * HW;
    int splits = k_max / 4096;
    if (splits < 1) splits = 1;
    if (splits > 16) splits = 16;

    float *col = nullptr, *dtmp = nullptr, *ws = nullptr;
    FORGE_CHECK(cudaMallocAsync(&col, (size_t)col_rows * nb_max * HW * sizeof(float), stream));
    FORGE_CHECK(cudaMallocAsync(&dtmp, (size_t)C_out * nb_max * HW * sizeof(float), stream));
    if (splits > 1)
        FORGE_CHECK(cudaMallocAsync(&ws, (size_t)gemm::fp32_gemm_splitk_workspace(Cog, col_rows, splits) * sizeof(float),
                                    stream));
    const int block = 256;

    for (int n0 = 0; n0 < N; n0 += nb_max) {
        int nb = (N - n0 < nb_max) ? N - n0 : nb_max;
        int cols = nb * HW;
        float chunk_beta = (n0 == 0) ? beta : 1.0f;   // later chunks accumulate
        nchw_to_cnhw_kernel<<<blocks_for((long long)C_out * cols, block), block, 0, stream>>>(
            d_output + (size_t)n0 * C_out * HW, dtmp, nb, C_out, HW);
        for (int g = 0; g < groups; ++g) {
            im2col_batched_kernel<<<blocks_for((long long)col_rows * cols, block), block, 0, stream>>>(
                input + (size_t)n0 * C_in * H * W, col, nb, C_in, g * Cig, Cig, H, W,
                kH, kW, padH, padW, strideH, strideW, dilH, dilW, H_out, W_out);
            // dW_g [Cog, col_rows] (+)= dtmp[g rows] [Cog, cols] @ col^T [cols, col_rows]
            const float* A = dtmp + (size_t)g * Cog * cols;
            float* C = d_weight + (size_t)g * Cog * col_rows;
            if (splits > 1)
                gemm::fp32_gemm_splitk(gemm::Op::N, gemm::Op::T, Cog, col_rows, cols, 1.0f,
                                       A, cols, col, cols, chunk_beta, C, col_rows, nullptr, ws, splits, stream);
            else
                gemm::fp32_gemm(gemm::Op::N, gemm::Op::T, Cog, col_rows, cols, 1.0f,
                                A, cols, col, cols, chunk_beta, C, col_rows, nullptr, stream);
        }
    }

    if (d_bias)
        conv2d_bias_backward_kernel<<<C_out, 256, 0, stream>>>(d_output, d_bias, N, C_out, HW);

    FORGE_CHECK(cudaFreeAsync(col, stream));
    FORGE_CHECK(cudaFreeAsync(dtmp, stream));
    if (ws) FORGE_CHECK(cudaFreeAsync(ws, stream));
}

}} // namespace forge::nn


// =====================================================================
// Tensor-level conv2d with autograd
// =====================================================================

namespace forge {

// Forward declaration for record function (defined in autograd_ops.cu)
void record_conv2d(Tensor& out, const Tensor& input, const Tensor& weight,
                   const Tensor& bias,
                   int C_in, int H, int W,
                   int C_out, int kH, int kW,
                   int padH, int padW,
                   int strideH, int strideW,
                   int dilH, int dilW, int groups,
                   bool has_bias);

static inline int conv_out_dim(int in_size, int pad, int dil, int kernel, int stride) {
    return (in_size + 2 * pad - dil * (kernel - 1) - 1) / stride + 1;
}

Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int padH, int padW,
              int strideH, int strideW,
              int dilH, int dilW,
              int groups) {
    int N     = static_cast<int>(input.size(0));
    int C_in  = static_cast<int>(input.size(1));
    int H     = static_cast<int>(input.size(2));
    int W     = static_cast<int>(input.size(3));
    int C_out = static_cast<int>(weight.size(0));
    int kH    = static_cast<int>(weight.size(2));
    int kW    = static_cast<int>(weight.size(3));

    int H_out = conv_out_dim(H, padH, dilH, kH, strideH);
    int W_out = conv_out_dim(W, padW, dilW, kW, strideW);

    Tensor output = Tensor::zeros({N, C_out, H_out, W_out});

    bool has_bias = bias.defined();
    nn::conv2d_forward(
        input.data<float>(), weight.data<float>(),
        has_bias ? bias.data<float>() : nullptr,
        output.data<float>(),
        N, C_in, H, W, C_out, kH, kW,
        padH, padW, strideH, strideW, dilH, dilW, groups);

    cudaDeviceSynchronize();

    if (should_record(input, weight) || (has_bias && should_record(bias))) {
        record_conv2d(output, input, weight, bias,
                      C_in, H, W, C_out, kH, kW,
                      padH, padW, strideH, strideW, dilH, dilW, groups, has_bias);
    }

    return output;
}

Tensor conv2d(const Tensor& input, const Tensor& weight,
              int padH, int padW,
              int strideH, int strideW,
              int dilH, int dilW,
              int groups) {
    return conv2d(input, weight, Tensor(), padH, padW, strideH, strideW, dilH, dilW, groups);
}

} // namespace forge
