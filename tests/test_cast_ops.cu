// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Tests copy, cast, and type conversion operations.
#include <forge/tensor.h>
#include "test_helpers.cuh"
#include <cmath>

using namespace forge;

int main() {
    std::printf("=== Test: Copy, Cast & Type Conversion ===\n");

    // ------ clone ------
    {
        std::vector<float> ha = {1, 2, 3, 4, 5, 6};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {2, 3});

        Tensor b = a.clone();
        cudaDeviceSynchronize();

        // Verify data matches
        auto gb = test::to_host(b.data<float>(), 6);
        FORGE_ASSERT(test::check_arrays(gb, ha, 0.0f, 0.0f, "clone"),
                     "clone: data mismatch");

        // Verify independent storage: modify original, clone unchanged
        a.fill_(99.0f);
        cudaDeviceSynchronize();
        auto gb2 = test::to_host(b.data<float>(), 6);
        FORGE_ASSERT(test::check_arrays(gb2, ha, 0.0f, 0.0f, "clone_independent"),
                     "clone_independent: data should be unchanged");

        b = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ Float32 → Int32 → Float32 ------
    {
        std::vector<float> ha = {1.9f, -2.1f, 0.0f, 100.5f};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {4});

        Tensor b = a.to(DType::Int32);
        cudaDeviceSynchronize();
        FORGE_ASSERT(b.dtype() == DType::Int32, "f32_to_i32: dtype should be Int32");
        auto gb = test::to_host(b.data<int32_t>(), 4);
        // static_cast<int32_t> truncates toward zero
        FORGE_ASSERT(gb[0] == 1 && gb[1] == -2 && gb[2] == 0 && gb[3] == 100,
                     "f32_to_i32: data mismatch");

        Tensor c = b.to(DType::Float32);
        cudaDeviceSynchronize();
        FORGE_ASSERT(c.dtype() == DType::Float32, "i32_to_f32: dtype should be Float32");
        auto gc = test::to_host(c.data<float>(), 4);
        std::vector<float> ref = {1.0f, -2.0f, 0.0f, 100.0f};
        FORGE_ASSERT(test::check_arrays(gc, ref, 0.0f, 0.0f, "f32_i32_roundtrip"),
                     "f32_i32_roundtrip: mismatch");

        c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ Float32 → Int64 → Float32 ------
    {
        std::vector<float> ha = {-5.7f, 0.0f, 42.3f, 1000.0f};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {4});

        Tensor b = a.to(DType::Int64);
        cudaDeviceSynchronize();
        FORGE_ASSERT(b.dtype() == DType::Int64, "f32_to_i64: dtype");
        auto gb = test::to_host(b.data<int64_t>(), 4);
        FORGE_ASSERT(gb[0] == -5 && gb[1] == 0 && gb[2] == 42 && gb[3] == 1000,
                     "f32_to_i64: data mismatch");

        Tensor c = b.to_float();
        cudaDeviceSynchronize();
        auto gc = test::to_host(c.data<float>(), 4);
        std::vector<float> ref = {-5.0f, 0.0f, 42.0f, 1000.0f};
        FORGE_ASSERT(test::check_arrays(gc, ref, 0.0f, 0.0f, "f32_i64_roundtrip"),
                     "f32_i64_roundtrip: mismatch");

        c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ Float32 → Bool → Float32 ------
    {
        std::vector<float> ha = {0.0f, 1.0f, -3.5f, 0.0f, 100.0f};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {5});

        Tensor b = a.to_bool();
        cudaDeviceSynchronize();
        FORGE_ASSERT(b.dtype() == DType::Bool, "f32_to_bool: dtype");
        auto gb = test::to_host(static_cast<const uint8_t*>(b.data_ptr()), 5);
        FORGE_ASSERT(gb[0]==0 && gb[1]==1 && gb[2]==1 && gb[3]==0 && gb[4]==1,
                     "f32_to_bool: data mismatch");

        Tensor c = b.to_float();
        cudaDeviceSynchronize();
        auto gc = test::to_host(c.data<float>(), 5);
        std::vector<float> ref = {0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
        FORGE_ASSERT(test::check_arrays(gc, ref, 0.0f, 0.0f, "f32_bool_roundtrip"),
                     "f32_bool_roundtrip: mismatch");

        c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ Float32 → Float16 → Float32 (precision loss) ------
    {
        std::vector<float> ha = {1.0f, 0.5f, -3.25f, 1024.0f};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {4});

        Tensor b = a.to_half();
        cudaDeviceSynchronize();
        FORGE_ASSERT(b.dtype() == DType::Float16, "f32_to_f16: dtype");
        FORGE_ASSERT(b.numel() == 4, "f32_to_f16: numel");

        Tensor c = b.to_float();
        cudaDeviceSynchronize();
        auto gc = test::to_host(c.data<float>(), 4);
        // These values are exact in fp16
        FORGE_ASSERT(test::check_arrays(gc, ha, 0.0f, 0.0f, "f32_f16_roundtrip"),
                     "f32_f16_roundtrip: mismatch");

        c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ Float32 → BFloat16 → Float32 ------
    {
        std::vector<float> ha = {1.0f, -2.0f, 0.0f, 256.0f};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {4});

        Tensor b = a.to_bfloat16();
        cudaDeviceSynchronize();
        FORGE_ASSERT(b.dtype() == DType::BFloat16, "f32_to_bf16: dtype");

        Tensor c = b.to_float();
        cudaDeviceSynchronize();
        auto gc = test::to_host(c.data<float>(), 4);
        // These values are exact in bf16 (powers of 2 and small ints)
        FORGE_ASSERT(test::check_arrays(gc, ha, 0.0f, 0.0f, "f32_bf16_roundtrip"),
                     "f32_bf16_roundtrip: mismatch");

        c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ Float32 → Int8 ------
    {
        std::vector<float> ha = {0.0f, 1.9f, -5.5f, 127.0f, -128.0f};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {5});

        Tensor b = a.to(DType::Int8);
        cudaDeviceSynchronize();
        FORGE_ASSERT(b.dtype() == DType::Int8, "f32_to_i8: dtype");
        auto gb = test::to_host(b.data<int8_t>(), 5);
        FORGE_ASSERT(gb[0]==0 && gb[1]==1 && gb[2]==-5 && gb[3]==127 && gb[4]==-128,
                     "f32_to_i8: data mismatch");

        std::printf("  PASS [f32_to_i8]\n");
        b = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ Int32 → Int64 (direct, no float intermediate) ------
    {
        std::vector<int32_t> ha = {0, -1, 100, 2147483647}; // INT32_MAX
        int32_t* da = test::to_device(ha);
        Tensor a = Tensor::from_blob(da, {4}, DType::Int32);

        Tensor b = a.to_long();
        cudaDeviceSynchronize();
        FORGE_ASSERT(b.dtype() == DType::Int64, "i32_to_i64: dtype");
        auto gb = test::to_host(b.data<int64_t>(), 4);
        FORGE_ASSERT(gb[0]==0 && gb[1]==-1 && gb[2]==100 && gb[3]==2147483647,
                     "i32_to_i64: data mismatch");

        std::printf("  PASS [i32_to_i64]\n");
        b = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ Two-step: Int32 → Bool (via nonzero test) ------
    {
        std::vector<int32_t> ha = {0, 1, -5, 0, 42};
        int32_t* da = test::to_device(ha);
        Tensor a = Tensor::from_blob(da, {5}, DType::Int32);

        Tensor b = a.to_bool();
        cudaDeviceSynchronize();
        FORGE_ASSERT(b.dtype() == DType::Bool, "i32_to_bool: dtype");
        auto gb = test::to_host(static_cast<const uint8_t*>(b.data_ptr()), 5);
        FORGE_ASSERT(gb[0]==0 && gb[1]==1 && gb[2]==1 && gb[3]==0 && gb[4]==1,
                     "i32_to_bool: data mismatch");

        std::printf("  PASS [i32_to_bool]\n");
        b = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ to() same dtype returns clone ------
    {
        std::vector<float> ha = {1, 2, 3};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {3});

        Tensor b = a.to(DType::Float32);
        cudaDeviceSynchronize();
        FORGE_ASSERT(b.dtype() == DType::Float32, "to_same: dtype");
        auto gb = test::to_host(b.data<float>(), 3);
        FORGE_ASSERT(test::check_arrays(gb, ha, 0.0f, 0.0f, "to_same_dtype"),
                     "to_same_dtype: mismatch");

        // Verify independent storage
        a.fill_(0.0f);
        cudaDeviceSynchronize();
        auto gb2 = test::to_host(b.data<float>(), 3);
        FORGE_ASSERT(test::check_arrays(gb2, ha, 0.0f, 0.0f, "to_same_independent"),
                     "to_same_independent: should be unchanged");

        b = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ type_as ------
    {
        std::vector<float> ha = {1.5f, 2.5f, 3.5f};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {3});

        // Create an Int32 tensor as the type reference
        int32_t* di = test::to_device_zeroed<int32_t>(1);
        Tensor ref = Tensor::from_blob(di, {1}, DType::Int32);

        Tensor b = a.type_as(ref);
        cudaDeviceSynchronize();
        FORGE_ASSERT(b.dtype() == DType::Int32, "type_as: dtype should match ref");
        auto gb = test::to_host(b.data<int32_t>(), 3);
        FORGE_ASSERT(gb[0]==1 && gb[1]==2 && gb[2]==3,
                     "type_as: data mismatch");

        std::printf("  PASS [type_as]\n");
        b = Tensor(); ref = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(di);
    }

    // ------ detach ------
    {
        std::vector<float> ha = {10, 20, 30};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {3});

        Tensor b = a.detach();
        cudaDeviceSynchronize();
        auto gb = test::to_host(b.data<float>(), 3);
        FORGE_ASSERT(test::check_arrays(gb, ha, 0.0f, 0.0f, "detach"),
                     "detach: data mismatch");

        std::printf("  PASS [detach]\n");
        b = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ Non-contiguous tensor cast ------
    {
        // Create {3,4}, transpose to {4,3}, then cast to Int32
        std::vector<float> ha = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        float* da = test::to_device(ha);
        Tensor base = Tensor::from_ptr(da, {3, 4});
        Tensor input = base.transpose(0, 1); // {4,3}, non-contiguous

        Tensor b = input.to_int();
        cudaDeviceSynchronize();
        FORGE_ASSERT(b.dtype() == DType::Int32, "noncontig_cast: dtype");
        FORGE_ASSERT(b.size(0) == 4 && b.size(1) == 3,
                     "noncontig_cast: shape should be {4,3}");
        auto gb = test::to_host(b.data<int32_t>(), 12);
        // Transposed: row0=[1,5,9], row1=[2,6,10], row2=[3,7,11], row3=[4,8,12]
        std::vector<int32_t> ref = {1,5,9, 2,6,10, 3,7,11, 4,8,12};
        bool ok = true;
        for (int i = 0; i < 12; i++) {
            if (gb[i] != ref[i]) { ok = false; break; }
        }
        FORGE_ASSERT(ok, "noncontig_cast: data mismatch");
        std::printf("  PASS [noncontig_cast]\n");

        b = Tensor(); input = Tensor(); base = Tensor();
        cudaFree(da);
    }

    // ------ Two-step: Int8 → BFloat16 (through Float32) ------
    {
        std::vector<int8_t> ha = {0, 1, -1, 127, -128};
        int8_t* da = test::to_device(ha);
        Tensor a = Tensor::from_blob(da, {5}, DType::Int8);

        Tensor b = a.to_bfloat16();
        cudaDeviceSynchronize();
        FORGE_ASSERT(b.dtype() == DType::BFloat16, "i8_to_bf16: dtype");

        // Convert back to float to check values
        Tensor c = b.to_float();
        cudaDeviceSynchronize();
        auto gc = test::to_host(c.data<float>(), 5);
        std::vector<float> ref = {0.0f, 1.0f, -1.0f, 127.0f, -128.0f};
        FORGE_ASSERT(test::check_arrays(gc, ref, 0.0f, 0.0f, "i8_bf16_roundtrip"),
                     "i8_bf16_roundtrip: mismatch");

        c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ Shape preservation ------
    {
        std::vector<float> ha = {1,2,3,4,5,6};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {2, 3});

        Tensor b = a.to_int();
        cudaDeviceSynchronize();
        FORGE_ASSERT(b.ndim() == 2 && b.size(0) == 2 && b.size(1) == 3,
                     "shape_preserve: shape should be {2,3}");
        FORGE_ASSERT(b.is_contiguous(), "shape_preserve: should be contiguous");

        std::printf("  PASS [shape_preserve]\n");
        b = Tensor(); a = Tensor();
        cudaFree(da);
    }

    std::printf("ALL PASSED\n");
    return 0;
}
