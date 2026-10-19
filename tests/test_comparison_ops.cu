// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Tests comparison, logical, and conditional operations.
#include <forge/tensor.h>
#include <forge/tensor_ops.h>
#include <forge/comparison_ops.h>
#include "test_helpers.cuh"
#include <cmath>
#include <limits>

using namespace forge;

int main() {
    std::printf("=== Test: Comparison & Logical Ops ===\n");

    // ------ eq / ne ------
    {
        std::vector<float> ha = {1, 2, 3, 4, 5};
        std::vector<float> hb = {1, 0, 3, 0, 5};
        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {5});
        Tensor b = Tensor::from_ptr(db, {5});

        Tensor c = forge::eq(a, b);
        cudaDeviceSynchronize();
        FORGE_ASSERT(c.dtype() == DType::Bool, "eq: dtype should be Bool");
        auto gc = test::to_host(static_cast<const uint8_t*>(c.data_ptr()), 5);
        FORGE_ASSERT(gc[0] == 1 && gc[1] == 0 && gc[2] == 1 && gc[3] == 0 && gc[4] == 1,
                     "eq: data mismatch");

        c = forge::ne(a, b);
        cudaDeviceSynchronize();
        gc = test::to_host(static_cast<const uint8_t*>(c.data_ptr()), 5);
        FORGE_ASSERT(gc[0] == 0 && gc[1] == 1 && gc[2] == 0 && gc[3] == 1 && gc[4] == 0,
                     "ne: data mismatch");

        std::printf("  PASS [eq_ne]\n");
        c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ lt / le / gt / ge ------
    {
        std::vector<float> ha = {1, 2, 3, 4, 5};
        std::vector<float> hb = {5, 4, 3, 2, 1};
        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {5});
        Tensor b = Tensor::from_ptr(db, {5});

        // lt: [1<5, 2<4, 3<3, 4<2, 5<1] = [1,1,0,0,0]
        Tensor c = forge::lt(a, b);
        cudaDeviceSynchronize();
        auto gc = test::to_host(static_cast<const uint8_t*>(c.data_ptr()), 5);
        FORGE_ASSERT(gc[0] == 1 && gc[1] == 1 && gc[2] == 0 && gc[3] == 0 && gc[4] == 0,
                     "lt: data mismatch");

        // le: [1<=5, 2<=4, 3<=3, 4<=2, 5<=1] = [1,1,1,0,0]
        c = forge::le(a, b);
        cudaDeviceSynchronize();
        gc = test::to_host(static_cast<const uint8_t*>(c.data_ptr()), 5);
        FORGE_ASSERT(gc[0] == 1 && gc[1] == 1 && gc[2] == 1 && gc[3] == 0 && gc[4] == 0,
                     "le: data mismatch");

        // gt: [1>5, 2>4, 3>3, 4>2, 5>1] = [0,0,0,1,1]
        c = forge::gt(a, b);
        cudaDeviceSynchronize();
        gc = test::to_host(static_cast<const uint8_t*>(c.data_ptr()), 5);
        FORGE_ASSERT(gc[0] == 0 && gc[1] == 0 && gc[2] == 0 && gc[3] == 1 && gc[4] == 1,
                     "gt: data mismatch");

        // ge: [1>=5, 2>=4, 3>=3, 4>=2, 5>=1] = [0,0,1,1,1]
        c = forge::ge(a, b);
        cudaDeviceSynchronize();
        gc = test::to_host(static_cast<const uint8_t*>(c.data_ptr()), 5);
        FORGE_ASSERT(gc[0] == 0 && gc[1] == 0 && gc[2] == 1 && gc[3] == 1 && gc[4] == 1,
                     "ge: data mismatch");

        std::printf("  PASS [lt_le_gt_ge]\n");
        c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ Scalar comparison ------
    {
        std::vector<float> ha = {1, 2, 3, 4, 5};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {5});

        Tensor c = forge::gt(a, 3.0f);
        cudaDeviceSynchronize();
        auto gc = test::to_host(static_cast<const uint8_t*>(c.data_ptr()), 5);
        FORGE_ASSERT(gc[0] == 0 && gc[1] == 0 && gc[2] == 0 && gc[3] == 1 && gc[4] == 1,
                     "gt_scalar: data mismatch");

        c = forge::eq(a, 3.0f);
        cudaDeviceSynchronize();
        gc = test::to_host(static_cast<const uint8_t*>(c.data_ptr()), 5);
        FORGE_ASSERT(gc[0] == 0 && gc[1] == 0 && gc[2] == 1 && gc[3] == 0 && gc[4] == 0,
                     "eq_scalar: data mismatch");

        std::printf("  PASS [scalar_comparison]\n");
        c = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ Broadcasting comparison: {3,1} > {1,3} ------
    {
        std::vector<float> ha = {1, 2, 3};
        std::vector<float> hb = {2, 2, 2};
        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {3, 1});
        Tensor b = Tensor::from_ptr(db, {1, 3});

        Tensor c = forge::gt(a, b);
        cudaDeviceSynchronize();
        FORGE_ASSERT(c.size(0) == 3 && c.size(1) == 3,
                     "broadcast_cmp: shape should be {3,3}");
        auto gc = test::to_host(static_cast<const uint8_t*>(c.data_ptr()), 9);
        // a[i] > b[j]: [1>2,1>2,1>2, 2>2,2>2,2>2, 3>2,3>2,3>2]
        //             = [0,0,0, 0,0,0, 1,1,1]
        FORGE_ASSERT(gc[0]==0 && gc[1]==0 && gc[2]==0 &&
                     gc[3]==0 && gc[4]==0 && gc[5]==0 &&
                     gc[6]==1 && gc[7]==1 && gc[8]==1,
                     "broadcast_cmp: data mismatch");

        std::printf("  PASS [broadcast_comparison]\n");
        c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ Logical ops ------
    {
        std::vector<uint8_t> ha = {1, 1, 0, 0};
        std::vector<uint8_t> hb = {1, 0, 1, 0};
        uint8_t* da = test::to_device(ha);
        uint8_t* db = test::to_device(hb);
        Tensor a = Tensor::from_blob(da, {4}, DType::Bool);
        Tensor b = Tensor::from_blob(db, {4}, DType::Bool);

        // AND: [1,0,0,0]
        Tensor c = forge::logical_and(a, b);
        cudaDeviceSynchronize();
        auto gc = test::to_host(static_cast<const uint8_t*>(c.data_ptr()), 4);
        FORGE_ASSERT(gc[0]==1 && gc[1]==0 && gc[2]==0 && gc[3]==0,
                     "logical_and: mismatch");

        // OR: [1,1,1,0]
        c = forge::logical_or(a, b);
        cudaDeviceSynchronize();
        gc = test::to_host(static_cast<const uint8_t*>(c.data_ptr()), 4);
        FORGE_ASSERT(gc[0]==1 && gc[1]==1 && gc[2]==1 && gc[3]==0,
                     "logical_or: mismatch");

        // XOR: [0,1,1,0]
        c = forge::logical_xor(a, b);
        cudaDeviceSynchronize();
        gc = test::to_host(static_cast<const uint8_t*>(c.data_ptr()), 4);
        FORGE_ASSERT(gc[0]==0 && gc[1]==1 && gc[2]==1 && gc[3]==0,
                     "logical_xor: mismatch");

        // NOT a: [0,0,1,1]
        c = forge::logical_not(a);
        cudaDeviceSynchronize();
        gc = test::to_host(static_cast<const uint8_t*>(c.data_ptr()), 4);
        FORGE_ASSERT(gc[0]==0 && gc[1]==0 && gc[2]==1 && gc[3]==1,
                     "logical_not: mismatch");

        std::printf("  PASS [logical_ops]\n");
        c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ isnan / isinf / isfinite ------
    {
        float nan_val = std::numeric_limits<float>::quiet_NaN();
        float inf_val = std::numeric_limits<float>::infinity();
        std::vector<float> ha = {1.0f, nan_val, inf_val, -inf_val, 0.0f};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {5});

        // isnan: [0, 1, 0, 0, 0]
        Tensor c = forge::isnan(a);
        cudaDeviceSynchronize();
        auto gc = test::to_host(static_cast<const uint8_t*>(c.data_ptr()), 5);
        FORGE_ASSERT(gc[0]==0 && gc[1]==1 && gc[2]==0 && gc[3]==0 && gc[4]==0,
                     "isnan: mismatch");

        // isinf: [0, 0, 1, 1, 0]
        c = forge::isinf(a);
        cudaDeviceSynchronize();
        gc = test::to_host(static_cast<const uint8_t*>(c.data_ptr()), 5);
        FORGE_ASSERT(gc[0]==0 && gc[1]==0 && gc[2]==1 && gc[3]==1 && gc[4]==0,
                     "isinf: mismatch");

        // isfinite: [1, 0, 0, 0, 1]
        c = forge::isfinite(a);
        cudaDeviceSynchronize();
        gc = test::to_host(static_cast<const uint8_t*>(c.data_ptr()), 5);
        FORGE_ASSERT(gc[0]==1 && gc[1]==0 && gc[2]==0 && gc[3]==0 && gc[4]==1,
                     "isfinite: mismatch");

        std::printf("  PASS [special_value_checks]\n");
        c = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ where ------
    {
        // condition: [true, false, true, false, true]
        std::vector<uint8_t> hc = {1, 0, 1, 0, 1};
        std::vector<float> hx = {10, 20, 30, 40, 50};
        std::vector<float> hy = {-1, -2, -3, -4, -5};

        uint8_t* dc = test::to_device(hc);
        float* dx = test::to_device(hx);
        float* dy = test::to_device(hy);
        Tensor cond = Tensor::from_blob(dc, {5}, DType::Bool);
        Tensor x = Tensor::from_ptr(dx, {5});
        Tensor y = Tensor::from_ptr(dy, {5});

        Tensor result = forge::where(cond, x, y);
        cudaDeviceSynchronize();
        FORGE_ASSERT(result.dtype() == DType::Float32, "where: output should be Float32");
        auto gr = test::to_host(result.data<float>(), 5);
        std::vector<float> ref = {10, -2, 30, -4, 50};
        FORGE_ASSERT(test::check_arrays(gr, ref, 0.0f, 0.0f, "where"),
                     "where: data mismatch");

        result = Tensor(); y = Tensor(); x = Tensor(); cond = Tensor();
        cudaFree(dc); cudaFree(dx); cudaFree(dy);
    }

    // ------ where with broadcasting ------
    {
        // cond: {3,1} = [[true],[false],[true]]  — broadcasts along cols
        // x: {3,2} = [[10,20],[30,40],[50,60]]
        // y: scalar-like {1} = [-1]
        std::vector<uint8_t> hc = {1, 0, 1};
        std::vector<float> hx = {10,20, 30,40, 50,60};
        std::vector<float> hy = {-1.0f};

        uint8_t* dc = test::to_device(hc);
        float* dx = test::to_device(hx);
        float* dy = test::to_device(hy);
        Tensor cond = Tensor::from_blob(dc, {3, 1}, DType::Bool);
        Tensor x = Tensor::from_ptr(dx, {3, 2});
        Tensor y = Tensor::from_ptr(dy, {1});

        Tensor result = forge::where(cond, x, y);
        cudaDeviceSynchronize();
        FORGE_ASSERT(result.size(0) == 3 && result.size(1) == 2,
                     "where_broadcast: shape should be {3,2}");
        auto gr = test::to_host(result.data<float>(), 6);
        // cond broadcasts: row0=true, row1=false, row2=true
        // y broadcasts -1 everywhere
        std::vector<float> ref = {10, 20, -1, -1, 50, 60};
        FORGE_ASSERT(test::check_arrays(gr, ref, 0.0f, 0.0f, "where_broadcast"),
                     "where_broadcast: data mismatch");

        result = Tensor(); y = Tensor(); x = Tensor(); cond = Tensor();
        cudaFree(dc); cudaFree(dx); cudaFree(dy);
    }

    // ------ any / all ------
    {
        // All true
        std::vector<uint8_t> h1 = {1, 1, 1, 1};
        uint8_t* d1 = test::to_device(h1);
        Tensor t1 = Tensor::from_blob(d1, {4}, DType::Bool);
        FORGE_ASSERT(forge::any(t1) == true, "any: all-true should be true");
        FORGE_ASSERT(forge::all(t1) == true, "all: all-true should be true");

        // Some true
        std::vector<uint8_t> h2 = {0, 0, 1, 0};
        uint8_t* d2 = test::to_device(h2);
        Tensor t2 = Tensor::from_blob(d2, {4}, DType::Bool);
        FORGE_ASSERT(forge::any(t2) == true, "any: some-true should be true");
        FORGE_ASSERT(forge::all(t2) == false, "all: some-true should be false");

        // All false
        std::vector<uint8_t> h3 = {0, 0, 0, 0};
        uint8_t* d3 = test::to_device(h3);
        Tensor t3 = Tensor::from_blob(d3, {4}, DType::Bool);
        FORGE_ASSERT(forge::any(t3) == false, "any: all-false should be false");
        FORGE_ASSERT(forge::all(t3) == false, "all: all-false should be false");

        std::printf("  PASS [any_all]\n");
        t3 = Tensor(); t2 = Tensor(); t1 = Tensor();
        cudaFree(d1); cudaFree(d2); cudaFree(d3);
    }

    // ------ allclose ------
    {
        std::vector<float> ha = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> hb = {1.0f, 2.0f, 3.0f, 4.0f};
        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {4});
        Tensor b = Tensor::from_ptr(db, {4});

        FORGE_ASSERT(forge::allclose(a, b) == true, "allclose: exact match should be true");

        // Slightly different
        std::vector<float> hc = {1.0f, 2.0f, 3.0f, 100.0f};
        float* dc = test::to_device(hc);
        Tensor c = Tensor::from_ptr(dc, {4});
        FORGE_ASSERT(forge::allclose(a, c) == false, "allclose: different should be false");

        // Within tolerance
        std::vector<float> hd = {1.0f + 1e-6f, 2.0f + 1e-6f, 3.0f, 4.0f - 1e-6f};
        float* dd = test::to_device(hd);
        Tensor d = Tensor::from_ptr(dd, {4});
        FORGE_ASSERT(forge::allclose(a, d, 1e-5f, 1e-5f) == true,
                     "allclose: within tolerance should be true");

        std::printf("  PASS [allclose]\n");
        d = Tensor(); c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dd);
    }

    // ------ Zero-element tensor ------
    {
        Tensor a = Tensor::empty({0});
        Tensor b = Tensor::empty({0});
        Tensor c = forge::eq(a, b);
        FORGE_ASSERT(c.numel() == 0, "zero_elem_cmp: numel should be 0");
        FORGE_ASSERT(c.dtype() == DType::Bool, "zero_elem_cmp: dtype should be Bool");

        std::printf("  PASS [zero_element]\n");
    }

    std::printf("ALL PASSED\n");
    return 0;
}
