// Forge GPU Compute Library
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Unified INT8 GEMM dispatch. Routes to wmma (sm_80+) or dp4a (sm_61+).

#include <forge/gemm/int8_gemm.cuh>
#include <forge/gemm/int8_dense.cuh>
#include <forge/gemm/int8_dp4a.cuh>

namespace forge { namespace gemm {

int get_compute_capability(int device) {
    static int cached = 0;
    if (cached == 0) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        cached = major * 10 + minor;
    }
    return cached;
}

bool has_wmma_int8(int device) {
    return get_compute_capability(device) >= 80;
}

void int8_gemm(
    const int8_t* A,
    const int8_t* B,
    const float*  act_scale,
    const __half* w_scale,
    const __half* bias,
    __half*       output,
    int           M,
    int           K,
    int           N,
    cudaStream_t  stream)
{
    if (has_wmma_int8()) {
        int8_gemm_dequant(A, B, act_scale, w_scale, bias, output,
                          M, K, N, stream);
    } else {
        int8_dp4a_gemm_dequant(A, B, act_scale, w_scale, bias, output,
                               M, K, N, stream);
    }
}

}} // namespace forge::gemm
