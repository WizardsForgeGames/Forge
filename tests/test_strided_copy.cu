// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Tests strided_copy_kernel via Tensor::contiguous() on non-contiguous views.
// Pattern: create contiguous tensor with known data -> create non-contiguous view
//          -> call contiguous() -> D2H readback -> compare against CPU reference.
#include <forge/tensor.h>
#include "test_helpers.cuh"

using namespace forge;

int main() {
    std::printf("=== Test: Strided Copy (contiguous on non-contiguous views) ===\n");

    // ------ 2D transpose ------
    {
        // Create {4,3} with known data: row-major [0,1,2,...,11]
        const int R = 4, C = 3;
        std::vector<float> h_data(R * C);
        for (int i = 0; i < R * C; i++) h_data[i] = static_cast<float>(i);

        float* d_ptr = test::to_device(h_data);
        Tensor t = Tensor::from_ptr(d_ptr, {R, C});

        // Transpose to {3,4}
        Tensor tt = t.transpose(0, 1);
        FORGE_ASSERT(!tt.is_contiguous(), "transpose: should be non-contiguous");
        FORGE_ASSERT(tt.size(0) == C, "transpose: size(0) should be 3");
        FORGE_ASSERT(tt.size(1) == R, "transpose: size(1) should be 4");

        // Make contiguous (triggers strided_copy_kernel)
        Tensor tc = tt.contiguous();
        cudaDeviceSynchronize();
        FORGE_ASSERT(tc.is_contiguous(), "transpose contiguous: should be contiguous");

        // Readback and verify against CPU reference
        auto gpu = test::to_host(tc.data<float>(), R * C);

        // CPU reference: transpose of row-major {4,3} -> row-major {3,4}
        // Original[r][c] = r*3+c, Transposed[c][r] = r*3+c
        std::vector<float> ref(R * C);
        for (int c = 0; c < C; c++)
            for (int r = 0; r < R; r++)
                ref[c * R + r] = static_cast<float>(r * C + c);

        FORGE_ASSERT(test::check_arrays(gpu, ref, 0.0f, 0.0f, "2d_transpose"),
                     "2d_transpose: data mismatch");

        tc = Tensor(); tt = Tensor(); t = Tensor();
        cudaFree(d_ptr);
    }

    // ------ Slice ------
    {
        // Create {10} with data [0..9]
        const int N = 10;
        std::vector<float> h_data(N);
        for (int i = 0; i < N; i++) h_data[i] = static_cast<float>(i);

        float* d_ptr = test::to_device(h_data);
        Tensor t = Tensor::from_ptr(d_ptr, {N});

        // Slice [2, 5): elements 2,3,4
        Tensor ts = t.slice(0, 2, 3);
        FORGE_ASSERT(ts.numel() == 3, "slice: numel should be 3");

        // Slice of a 1D contiguous tensor is still contiguous (stride preserved)
        // but let's make contiguous anyway to verify data
        Tensor tc = ts.contiguous();
        cudaDeviceSynchronize();

        auto gpu = test::to_host(tc.data<float>(), 3);
        std::vector<float> ref = {2.0f, 3.0f, 4.0f};
        FORGE_ASSERT(test::check_arrays(gpu, ref, 0.0f, 0.0f, "slice"),
                     "slice: data mismatch");

        tc = Tensor(); ts = Tensor(); t = Tensor();
        cudaFree(d_ptr);
    }

    // ------ Expand (stride-0 broadcast) ------
    {
        // Create {1,4} with data [10,20,30,40]
        std::vector<float> h_data = {10.0f, 20.0f, 30.0f, 40.0f};
        float* d_ptr = test::to_device(h_data);
        Tensor t = Tensor::from_ptr(d_ptr, {1, 4});

        // Expand to {3,4}
        Tensor te = t.expand({3, 4});
        FORGE_ASSERT(te.numel() == 12, "expand: numel should be 12");
        FORGE_ASSERT(!te.is_contiguous(), "expand: stride-0 should be non-contiguous");

        Tensor tc = te.contiguous();
        cudaDeviceSynchronize();

        auto gpu = test::to_host(tc.data<float>(), 12);
        // Each row should be [10,20,30,40]
        std::vector<float> ref = {
            10,20,30,40,
            10,20,30,40,
            10,20,30,40
        };
        FORGE_ASSERT(test::check_arrays(gpu, ref, 0.0f, 0.0f, "expand"),
                     "expand: data mismatch");

        tc = Tensor(); te = Tensor(); t = Tensor();
        cudaFree(d_ptr);
    }

    // ------ Unsqueeze + squeeze roundtrip ------
    {
        const int N = 6;
        std::vector<float> h_data(N);
        for (int i = 0; i < N; i++) h_data[i] = static_cast<float>(i * 10);

        float* d_ptr = test::to_device(h_data);
        Tensor t = Tensor::from_ptr(d_ptr, {2, 3});

        // unsqueeze(0) -> {1,2,3} -> squeeze(0) -> {2,3}
        Tensor tu = t.unsqueeze(0);
        FORGE_ASSERT(tu.ndim() == 3, "unsqueeze: ndim should be 3");
        FORGE_ASSERT(tu.size(0) == 1, "unsqueeze: size(0) should be 1");

        Tensor ts = tu.squeeze(0);
        FORGE_ASSERT(ts.ndim() == 2, "squeeze: ndim should be 2");

        Tensor tc = ts.contiguous();
        cudaDeviceSynchronize();

        auto gpu = test::to_host(tc.data<float>(), N);
        std::vector<float> ref(N);
        for (int i = 0; i < N; i++) ref[i] = static_cast<float>(i * 10);
        FORGE_ASSERT(test::check_arrays(gpu, ref, 0.0f, 0.0f, "unsqueeze_squeeze"),
                     "unsqueeze_squeeze: data mismatch");

        tc = Tensor(); ts = Tensor(); tu = Tensor(); t = Tensor();
        cudaFree(d_ptr);
    }

    // ------ Large tensor: 256x256 transpose ------
    {
        const int N = 256;
        std::vector<float> h_data(N * N);
        for (int i = 0; i < N * N; i++) h_data[i] = static_cast<float>(i);

        float* d_ptr = test::to_device(h_data);
        Tensor t = Tensor::from_ptr(d_ptr, {N, N});

        Tensor tt = t.transpose(0, 1);
        Tensor tc = tt.contiguous();
        cudaDeviceSynchronize();

        auto gpu = test::to_host(tc.data<float>(), N * N);

        // CPU reference: transposed
        std::vector<float> ref(N * N);
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++)
                ref[c * N + r] = static_cast<float>(r * N + c);

        FORGE_ASSERT(test::check_arrays(gpu, ref, 0.0f, 0.0f, "large_transpose"),
                     "large_transpose: data mismatch");

        tc = Tensor(); tt = Tensor(); t = Tensor();
        cudaFree(d_ptr);
    }

    // ------ Single element ------
    {
        std::vector<float> h_data = {77.0f};
        float* d_ptr = test::to_device(h_data);
        Tensor t = Tensor::from_ptr(d_ptr, {1});

        Tensor tc = t.contiguous();
        cudaDeviceSynchronize();

        auto gpu = test::to_host(tc.data<float>(), 1);
        FORGE_ASSERT(gpu[0] == 77.0f, "single_elem: value should be 77.0");

        tc = Tensor(); t = Tensor();
        cudaFree(d_ptr);
        std::printf("  PASS [single_element]\n");
    }

    // ------ Size-1 dimensions ------
    {
        // Create {1,5,1} tensor
        std::vector<float> h_data = {1,2,3,4,5};
        float* d_ptr = test::to_device(h_data);
        Tensor t = Tensor::from_ptr(d_ptr, {1, 5, 1});

        // transpose(0,2) -> {1,5,1} with swapped strides
        Tensor tt = t.transpose(0, 2);
        Tensor tc = tt.contiguous();
        cudaDeviceSynchronize();

        auto gpu = test::to_host(tc.data<float>(), 5);
        std::vector<float> ref = {1,2,3,4,5};
        FORGE_ASSERT(test::check_arrays(gpu, ref, 0.0f, 0.0f, "size1_dims"),
                     "size1_dims: data mismatch");

        tc = Tensor(); tt = Tensor(); t = Tensor();
        cudaFree(d_ptr);
    }

    std::printf("ALL PASSED\n");
    return 0;
}
