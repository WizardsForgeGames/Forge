// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Fused 3D Winograd F(2,3)^3 Convolution.
//
// Uses Winograd minimal filtering to reduce arithmetic from 27 FMAs per
// output element (direct convolution) to 64/8 = 8 FMAs (3.375x reduction).
// The entire pipeline (input transform, element-wise multiply, output
// transform) is fused into a single kernel to avoid materializing 64 MB
// of intermediate buffers.
//
// Algorithm (per output tile of 2x2x2):
//   V = B^T * input_patch(4x4x4) * B        (input transform)
//   U = G  * filter(3x3x3)      * G^T       (filter transform, precomputed)
//   M[alpha] = sum_cin U[alpha][cin][cout] * V[alpha][cin][tile]
//   output_patch(2x2x2) = A^T * M * A       (output transform)
//
// Constraints: k=3x3x3, stride=1, dilation=1, pad=1 (same convolution).
// These match all ThermalUNet residual/encoder layers.
// For k=1x1x1 and k=2x2x2 stride=2, use the implicit GEMM or vol2col path.
#pragma once

#include <cuda_runtime.h>

namespace forge { namespace nn {

/// Pre-transform weight tensor from spatial domain to Winograd domain.
/// Call once after weight update; result cached until next update.
///   weight:  [C_out, C_in, 3, 3, 3]  (spatial domain)
///   U:       [64, C_in, C_out]        (Winograd domain, coalesced for forward)
///   U_t:     [64, C_out, C_in]        (transposed, coalesced for backward) or nullptr
void conv3d_winograd_filter_transform(
    const float* weight,    // [C_out, C_in, 3, 3, 3]
    float*       U,         // [64, C_in, C_out]
    float*       U_t,       // [64, C_out, C_in] or nullptr (backward-transposed layout)
    int C_out, int C_in,
    cudaStream_t stream = nullptr);

/// Fused Winograd F(2,3)^3 forward convolution.
/// Requirements: k=3, stride=1, dilation=1, pad=1.
/// U must be pre-transformed via conv3d_winograd_filter_transform.
void conv3d_winograd_forward(
    const float* input,     // [N, C_in, D, H, W]
    const float* U,         // [64, C_in, C_out]  (pre-transformed weight)
    const float* bias,      // [C_out] or nullptr
    float*       output,    // [N, C_out, D, H, W]  (same spatial, pad=1 stride=1)
    int N, int C_in, int D, int H, int W,
    int C_out,
    cudaStream_t stream = nullptr);

/// Fused Winograd F(2,3)^3 backward data (d_input from d_output).
/// Uses U_t (transposed layout from filter_transform) for coalesced memory access.
/// The kernel applies alpha flip permutation internally.
void conv3d_winograd_backward_data(
    const float* d_output,  // [N, C_out, D, H, W]
    const float* U_t,       // [64, C_out, C_in]  (transposed layout from filter_transform)
    float*       d_input,   // [N, C_in, D, H, W]
    int N, int C_in, int D, int H, int W,
    int C_out,
    cudaStream_t stream = nullptr);

/// Workspace size in floats for the Winograd-domain weight buffer.
inline int conv3d_winograd_U_size(int C_out, int C_in) {
    return 64 * C_out * C_in;
}

}} // namespace forge::nn
