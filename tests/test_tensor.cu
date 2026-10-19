// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
#include <forge/tensor.h>
#include "test_helpers.cuh"

using namespace forge;

int main() {
    std::printf("=== Test: Tensor ===\n");

    // ------ empty: shape metadata ------
    {
        Tensor t = Tensor::empty({3, 4, 5});
        FORGE_ASSERT(t.defined(), "empty: should be defined");
        FORGE_ASSERT(t.numel() == 60, "empty: numel should be 3*4*5=60");
        FORGE_ASSERT(t.ndim() == 3, "empty: ndim should be 3");
        FORGE_ASSERT(t.dtype() == DType::Float32, "empty: default dtype should be Float32");
        FORGE_ASSERT(t.size(0) == 3, "empty: size(0) should be 3");
        FORGE_ASSERT(t.size(1) == 4, "empty: size(1) should be 4");
        FORGE_ASSERT(t.size(2) == 5, "empty: size(2) should be 5");
        // Row-major strides: [20, 5, 1]
        FORGE_ASSERT(t.stride(0) == 20, "empty: stride(0) should be 20");
        FORGE_ASSERT(t.stride(1) == 5, "empty: stride(1) should be 5");
        FORGE_ASSERT(t.stride(2) == 1, "empty: stride(2) should be 1");
        FORGE_ASSERT(t.is_contiguous(), "empty: should be contiguous");
        FORGE_ASSERT(t.nbytes() == 60 * sizeof(float), "empty: nbytes should be 240");
        FORGE_ASSERT(t.data_ptr() != nullptr, "empty: data_ptr should be non-null");

        std::printf("  PASS [empty_shape]\n");
    }

    // ------ zeros: all elements zero ------
    {
        Tensor t = Tensor::zeros({100});
        cudaDeviceSynchronize();
        auto h = test::to_host(t.data<float>(), 100);
        bool all_zero = true;
        for (int i = 0; i < 100; i++) {
            if (h[i] != 0.0f) { all_zero = false; break; }
        }
        FORGE_ASSERT(all_zero, "zeros: all elements should be 0.0");
        std::printf("  PASS [zeros_data]\n");
    }

    // ------ ones: all elements one ------
    {
        Tensor t = Tensor::ones({50});
        cudaDeviceSynchronize();
        auto h = test::to_host(t.data<float>(), 50);
        bool all_one = true;
        for (int i = 0; i < 50; i++) {
            if (h[i] != 1.0f) { all_one = false; break; }
        }
        FORGE_ASSERT(all_one, "ones: all elements should be 1.0");
        std::printf("  PASS [ones_data]\n");
    }

    // ------ fill_ ------
    {
        Tensor t = Tensor::empty({20});
        t.fill_(42.0f);
        cudaDeviceSynchronize();
        auto h = test::to_host(t.data<float>(), 20);
        bool all_42 = true;
        for (int i = 0; i < 20; i++) {
            if (h[i] != 42.0f) { all_42 = false; break; }
        }
        FORGE_ASSERT(all_42, "fill_: all elements should be 42.0");
        std::printf("  PASS [fill_]\n");
    }

    // ------ from_ptr: non-owning typed wrap ------
    {
        float* d_ptr = test::to_device_zeroed<float>(12);
        Tensor t = Tensor::from_ptr(d_ptr, {3, 4});
        FORGE_ASSERT(t.data<float>() == d_ptr, "from_ptr: should share device pointer");
        FORGE_ASSERT(t.numel() == 12, "from_ptr: numel should be 12");
        FORGE_ASSERT(t.dtype() == DType::Float32, "from_ptr: dtype should be Float32");

        // from_ptr is non-owning; we must free manually
        // Destroy tensor first, then free
        t = Tensor(); // release handle
        cudaFree(d_ptr);
        std::printf("  PASS [from_ptr]\n");
    }

    // ------ from_blob: non-owning untyped wrap ------
    {
        int32_t* d_ptr = test::to_device_zeroed<int32_t>(6);
        Tensor t = Tensor::from_blob(d_ptr, {2, 3}, DType::Int32);
        FORGE_ASSERT(t.data_ptr() == d_ptr, "from_blob: should share device pointer");
        FORGE_ASSERT(t.numel() == 6, "from_blob: numel should be 6");
        FORGE_ASSERT(t.dtype() == DType::Int32, "from_blob: dtype should be Int32");

        t = Tensor();
        cudaFree(d_ptr);
        std::printf("  PASS [from_blob]\n");
    }

    // ------ data<float>() returns valid device pointer ------
    {
        Tensor t = Tensor::ones({10});
        float* ptr = t.data<float>();
        FORGE_ASSERT(ptr != nullptr, "data<float>: should be non-null");
        FORGE_ASSERT(ptr == t.data_ptr(), "data<float>: should match data_ptr()");

        cudaDeviceSynchronize();
        auto h = test::to_host(ptr, 10);
        FORGE_ASSERT(h[0] == 1.0f, "data<float>: readback through typed ptr should work");
        std::printf("  PASS [data_typed]\n");
    }

    // ------ Move semantics ------
    {
        Tensor t1 = Tensor::empty({5});
        void* original_ptr = t1.data_ptr();
        int original_use_count = t1.use_count();

        Tensor t2 = std::move(t1);
        FORGE_ASSERT(!t1.defined(), "move: source should be undefined after move");
        FORGE_ASSERT(t2.defined(), "move: target should be defined after move");
        FORGE_ASSERT(t2.data_ptr() == original_ptr, "move: target should have same pointer");
        FORGE_ASSERT(t2.use_count() == original_use_count,
                     "move: no refcount change");

        std::printf("  PASS [move_semantics]\n");
    }

    // ------ Copy semantics ------
    {
        Tensor t1 = Tensor::empty({5});
        FORGE_ASSERT(t1.use_count() == 1, "copy: initial use_count should be 1");

        Tensor t2 = t1; // copy
        FORGE_ASSERT(t1.use_count() == 2, "copy: use_count should be 2 after copy");
        FORGE_ASSERT(t2.use_count() == 2, "copy: both should show 2");
        FORGE_ASSERT(t1.data_ptr() == t2.data_ptr(),
                     "copy: should share same data pointer");

        std::printf("  PASS [copy_semantics]\n");
    }

    // ------ defined() ------
    {
        Tensor t_default;
        FORGE_ASSERT(!t_default.defined(), "defined: default tensor should be !defined()");

        Tensor t_real = Tensor::empty({1});
        FORGE_ASSERT(t_real.defined(), "defined: factory tensor should be defined()");

        std::printf("  PASS [defined]\n");
    }

    // ------ requires_grad ------
    {
        Tensor t = Tensor::empty({3, 4});
        FORGE_ASSERT(!t.requires_grad(), "requires_grad: default should be false");

        t.set_requires_grad_(true);
        FORGE_ASSERT(t.requires_grad(), "requires_grad: should be true after set");

        t.set_requires_grad_(false);
        FORGE_ASSERT(!t.requires_grad(), "requires_grad: should be false after unset");

        std::printf("  PASS [requires_grad]\n");
    }

    // ------ Version tracking ------
    {
        Tensor t = Tensor::empty({4});
        FORGE_ASSERT(t.version() == 0, "version: initial should be 0");

        t.bump_version();
        FORGE_ASSERT(t.version() == 1, "version: after bump should be 1");

        t.bump_version();
        FORGE_ASSERT(t.version() == 2, "version: after 2 bumps should be 2");

        std::printf("  PASS [version_tracking]\n");
    }

    // ------ 0-element tensor ------
    {
        Tensor t = Tensor::empty({0});
        FORGE_ASSERT(t.numel() == 0, "zero_elem: numel should be 0");
        FORGE_ASSERT(t.defined(), "zero_elem: should still be defined");
        FORGE_ASSERT(t.ndim() == 1, "zero_elem: ndim should be 1");

        std::printf("  PASS [zero_element]\n");
    }

    // ------ Int32 dtype ------
    {
        Tensor t = Tensor::empty({10}, DType::Int32);
        FORGE_ASSERT(t.dtype() == DType::Int32, "int32: dtype should be Int32");
        FORGE_ASSERT(t.nbytes() == 10 * sizeof(int32_t), "int32: nbytes should be 40");

        std::printf("  PASS [int32_dtype]\n");
    }

    // ------ Metadata accessors: numel, ndim, size, stride, nbytes ------
    {
        Tensor t = Tensor::empty({2, 3, 4});
        FORGE_ASSERT(t.numel() == 24, "meta: numel=24");
        FORGE_ASSERT(t.ndim() == 3, "meta: ndim=3");
        FORGE_ASSERT(t.size(0) == 2, "meta: size(0)=2");
        FORGE_ASSERT(t.size(1) == 3, "meta: size(1)=3");
        FORGE_ASSERT(t.size(2) == 4, "meta: size(2)=4");
        // Negative indexing
        FORGE_ASSERT(t.size(-1) == 4, "meta: size(-1)=4");
        FORGE_ASSERT(t.size(-2) == 3, "meta: size(-2)=3");
        FORGE_ASSERT(t.stride(-1) == 1, "meta: stride(-1)=1");
        FORGE_ASSERT(t.nbytes() == 24 * 4, "meta: nbytes=96");
        FORGE_ASSERT(t.sizes() != nullptr, "meta: sizes() non-null");
        FORGE_ASSERT(t.strides() != nullptr, "meta: strides() non-null");

        std::printf("  PASS [metadata_accessors]\n");
    }

    std::printf("ALL PASSED\n");
    return 0;
}
