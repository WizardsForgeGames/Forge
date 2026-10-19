// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/nn/linear.cuh>
#include <forge/gemm/fp32_tiled.cuh>

namespace forge { namespace nn {

// Forward: Y[B,N] = X[B,K] @ W[N,K]^T + bias[N]
//
// In GEMM terms: C[B,N] = A[B,K] * B_mat[K,N]
// where B_mat = W^T, so we pass W as transB=T.
//
// gemm(Op::N, Op::T, M=B, N=N_out, K=K_in,
//      alpha=1, A=input, lda=K, B=weight, ldb=K, beta=0, C=output, ldc=N)
//
// Bias: fp32_gemm supports optional bias[M] added per row.
// But we need bias[N] added per column (broadcast across batch).
// fp32_gemm adds bias[m] to C[m, n], but we want bias[n] on C[b, n].
// So we cannot use fp32_gemm's built-in bias here — we'll apply it separately.

// Simple bias-add kernel: output[b*N + n] += bias[n]
static __global__ void add_bias_kernel(
    float* __restrict__       output,
    const float* __restrict__ bias,
    int                       B,
    int                       N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N) return;
    int n = idx % N;
    output[idx] += bias[n];
}

void linear_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    int          B,
    int          N,
    int          K,
    cudaStream_t stream)
{
    // Y[B,N] = X[B,K] @ W[N,K]^T
    // C = alpha * op(A) * op(B) + beta * C
    // op(A) = X [B, K], op(B) = W^T [K, N]
    gemm::fp32_gemm(
        gemm::Op::N, gemm::Op::T,
        B, N, K,
        1.0f,
        input, K,       // A[B, K], lda=K
        weight, K,      // B[N, K], ldb=K (transposed → [K, N])
        0.0f,
        output, N,      // C[B, N], ldc=N
        nullptr,        // no built-in bias (wrong dimension)
        stream);

    if (bias) {
        int total = B * N;
        int block = 256;
        int grid  = (total + block - 1) / block;
        add_bias_kernel<<<grid, block, 0, stream>>>(output, bias, B, N);
    }
}

// Backward data: dX[B,K] = dY[B,N] @ W[N,K]
// C = A * B where A = dY[B,N], B = W[N,K] → need B not transposed
void linear_backward_data(
    const float* d_output,
    const float* weight,
    float*       d_input,
    int          B,
    int          N,
    int          K,
    cudaStream_t stream)
{
    // dX[B, K] = dY[B, N] @ W[N, K]
    gemm::fp32_gemm(
        gemm::Op::N, gemm::Op::N,
        B, K, N,
        1.0f,
        d_output, N,    // A[B, N], lda=N
        weight, K,      // B[N, K], ldb=K
        0.0f,
        d_input, K,     // C[B, K], ldc=K
        nullptr,
        stream);
}

// Backward weight: dW[N,K] = dY[B,N]^T @ X[B,K]
// C = A^T * B where A = dY[B,N] → A^T = [N,B], B = X[B,K]
void linear_backward_weight(
    const float* d_output,
    const float* input,
    float*       d_weight,
    int          B,
    int          N,
    int          K,
    float        beta,
    cudaStream_t stream)
{
    // dW[N, K] = dY[B, N]^T @ X[B, K]
    gemm::fp32_gemm(
        gemm::Op::T, gemm::Op::N,
        N, K, B,
        1.0f,
        d_output, N,    // A[B, N], lda=N (transposed → [N, B])
        input, K,       // B[B, K], ldb=K
        beta,
        d_weight, K,    // C[N, K], ldc=K
        nullptr,
        stream);
}

}} // namespace forge::nn
