// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Tests concatenation, stacking, and splitting operations.
#include <forge/tensor.h>
#include <forge/concat_ops.h>
#include "test_helpers.cuh"

using namespace forge;

int main() {
    std::printf("=== Test: Concatenation & Splitting Ops ===\n");

    // ------ cat dim=0 ------
    {
        // a: {2,3} = [[1,2,3],[4,5,6]]
        // b: {3,3} = [[7,8,9],[10,11,12],[13,14,15]]
        // cat(dim=0) → {5,3}
        std::vector<float> ha = {1,2,3, 4,5,6};
        std::vector<float> hb = {7,8,9, 10,11,12, 13,14,15};
        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {2, 3});
        Tensor b = Tensor::from_ptr(db, {3, 3});

        Tensor out = forge::cat({a, b}, 0);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.size(0) == 5 && out.size(1) == 3,
                     "cat_dim0: shape should be {5,3}");
        auto go = test::to_host(out.data<float>(), 15);
        std::vector<float> ref = {1,2,3, 4,5,6, 7,8,9, 10,11,12, 13,14,15};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "cat_dim0"),
                     "cat_dim0: data mismatch");

        out = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ cat dim=1 ------
    {
        // a: {2,2} = [[1,2],[5,6]]
        // b: {2,3} = [[3,4,0],[7,8,0]]
        // cat(dim=1) → {2,5}
        std::vector<float> ha = {1,2, 5,6};
        std::vector<float> hb = {3,4,0, 7,8,0};
        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {2, 2});
        Tensor b = Tensor::from_ptr(db, {2, 3});

        Tensor out = forge::cat({a, b}, 1);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.size(0) == 2 && out.size(1) == 5,
                     "cat_dim1: shape should be {2,5}");
        auto go = test::to_host(out.data<float>(), 10);
        // row0: [1,2,3,4,0]  row1: [5,6,7,8,0]
        std::vector<float> ref = {1,2,3,4,0, 5,6,7,8,0};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "cat_dim1"),
                     "cat_dim1: data mismatch");

        out = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ cat 3 tensors, dim=1 ------
    {
        std::vector<float> ha = {1, 4};
        std::vector<float> hb = {2,3, 5,6};
        std::vector<float> hc = {7, 8};
        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        float* dc = test::to_device(hc);
        Tensor a = Tensor::from_ptr(da, {2, 1});
        Tensor b = Tensor::from_ptr(db, {2, 2});
        Tensor c = Tensor::from_ptr(dc, {2, 1});

        Tensor out = forge::cat({a, b, c}, 1);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.size(0) == 2 && out.size(1) == 4,
                     "cat_3tensors: shape should be {2,4}");
        auto go = test::to_host(out.data<float>(), 8);
        std::vector<float> ref = {1,2,3,7, 4,5,6,8};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "cat_3tensors"),
                     "cat_3tensors: data mismatch");

        out = Tensor(); c = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db); cudaFree(dc);
    }

    // ------ cat 1D ------
    {
        std::vector<float> ha = {1,2,3};
        std::vector<float> hb = {4,5};
        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {3});
        Tensor b = Tensor::from_ptr(db, {2});

        Tensor out = forge::cat({a, b}, 0);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.size(0) == 5, "cat_1d: shape should be {5}");
        auto go = test::to_host(out.data<float>(), 5);
        std::vector<float> ref = {1,2,3,4,5};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "cat_1d"),
                     "cat_1d: data mismatch");

        out = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ cat single tensor ------
    {
        std::vector<float> ha = {1,2,3,4};
        float* da = test::to_device(ha);
        Tensor a = Tensor::from_ptr(da, {2, 2});

        Tensor out = forge::cat({a}, 0);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.size(0) == 2 && out.size(1) == 2,
                     "cat_single: shape should be {2,2}");
        auto go = test::to_host(out.data<float>(), 4);
        std::vector<float> ref = {1,2,3,4};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "cat_single"),
                     "cat_single: data mismatch");

        out = Tensor(); a = Tensor();
        cudaFree(da);
    }

    // ------ stack dim=0 ------
    {
        // a: {3} = [1,2,3], b: {3} = [4,5,6]
        // stack(dim=0) → {2,3}
        std::vector<float> ha = {1,2,3};
        std::vector<float> hb = {4,5,6};
        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {3});
        Tensor b = Tensor::from_ptr(db, {3});

        Tensor out = forge::stack({a, b}, 0);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.size(0) == 2 && out.size(1) == 3,
                     "stack_dim0: shape should be {2,3}");
        auto go = test::to_host(out.data<float>(), 6);
        std::vector<float> ref = {1,2,3, 4,5,6};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "stack_dim0"),
                     "stack_dim0: data mismatch");

        out = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ stack dim=1 ------
    {
        // a: {2,3}, b: {2,3}
        // stack(dim=1) → {2,2,3}
        std::vector<float> ha = {1,2,3, 4,5,6};
        std::vector<float> hb = {7,8,9, 10,11,12};
        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {2, 3});
        Tensor b = Tensor::from_ptr(db, {2, 3});

        Tensor out = forge::stack({a, b}, 1);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.size(0) == 2 && out.size(1) == 2 && out.size(2) == 3,
                     "stack_dim1: shape should be {2,2,3}");
        auto go = test::to_host(out.data<float>(), 12);
        // out[0][0] = a[0] = [1,2,3], out[0][1] = b[0] = [7,8,9]
        // out[1][0] = a[1] = [4,5,6], out[1][1] = b[1] = [10,11,12]
        std::vector<float> ref = {1,2,3, 7,8,9, 4,5,6, 10,11,12};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "stack_dim1"),
                     "stack_dim1: data mismatch");

        out = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ hstack ------
    {
        // 2D: hstack = cat(dim=1)
        std::vector<float> ha = {1,2, 5,6};
        std::vector<float> hb = {3,4, 7,8};
        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {2, 2});
        Tensor b = Tensor::from_ptr(db, {2, 2});

        Tensor out = forge::hstack({a, b});
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.size(0) == 2 && out.size(1) == 4,
                     "hstack_2d: shape should be {2,4}");
        auto go = test::to_host(out.data<float>(), 8);
        std::vector<float> ref = {1,2,3,4, 5,6,7,8};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "hstack_2d"),
                     "hstack_2d: data mismatch");

        out = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ hstack 1D (should cat along dim=0) ------
    {
        std::vector<float> ha = {1,2};
        std::vector<float> hb = {3,4,5};
        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {2});
        Tensor b = Tensor::from_ptr(db, {3});

        Tensor out = forge::hstack({a, b});
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.ndim() == 1 && out.size(0) == 5,
                     "hstack_1d: shape should be {5}");
        auto go = test::to_host(out.data<float>(), 5);
        std::vector<float> ref = {1,2,3,4,5};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "hstack_1d"),
                     "hstack_1d: data mismatch");

        out = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ vstack ------
    {
        std::vector<float> ha = {1,2,3};
        std::vector<float> hb = {4,5,6};
        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {1, 3});
        Tensor b = Tensor::from_ptr(db, {1, 3});

        Tensor out = forge::vstack({a, b});
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.size(0) == 2 && out.size(1) == 3,
                     "vstack: shape should be {2,3}");
        auto go = test::to_host(out.data<float>(), 6);
        std::vector<float> ref = {1,2,3,4,5,6};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "vstack"),
                     "vstack: data mismatch");

        out = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ split ------
    {
        // {10} split into chunks of 3 → [{3},{3},{3},{1}]
        std::vector<float> hd = {1,2,3,4,5,6,7,8,9,10};
        float* dd = test::to_device(hd);
        Tensor t = Tensor::from_ptr(dd, {10});

        auto parts = forge::split(t, 3, 0);
        FORGE_ASSERT(parts.size() == 4, "split: should have 4 parts");
        FORGE_ASSERT(parts[0].size(0) == 3, "split: part0 size should be 3");
        FORGE_ASSERT(parts[1].size(0) == 3, "split: part1 size should be 3");
        FORGE_ASSERT(parts[2].size(0) == 3, "split: part2 size should be 3");
        FORGE_ASSERT(parts[3].size(0) == 1, "split: part3 size should be 1");

        cudaDeviceSynchronize();
        auto g0 = test::to_host(parts[0].data<float>(), 3);
        auto g3 = test::to_host(parts[3].data<float>(), 1);
        std::vector<float> ref0 = {1,2,3};
        std::vector<float> ref3 = {10};
        FORGE_ASSERT(test::check_arrays(g0, ref0, 0.0f, 0.0f, "split_part0"),
                     "split_part0: mismatch");
        FORGE_ASSERT(test::check_arrays(g3, ref3, 0.0f, 0.0f, "split_part3"),
                     "split_part3: mismatch");

        parts.clear();
        t = Tensor();
        cudaFree(dd);
    }

    // ------ chunk ------
    {
        // {12} chunk into 4 → [{3},{3},{3},{3}]
        std::vector<float> hd = {1,2,3,4,5,6,7,8,9,10,11,12};
        float* dd = test::to_device(hd);
        Tensor t = Tensor::from_ptr(dd, {12});

        auto parts = forge::chunk(t, 4, 0);
        FORGE_ASSERT(parts.size() == 4, "chunk: should have 4 parts");
        for (int i = 0; i < 4; i++)
            FORGE_ASSERT(parts[i].size(0) == 3, "chunk: each part should be size 3");

        cudaDeviceSynchronize();
        auto g2 = test::to_host(parts[2].data<float>(), 3);
        std::vector<float> ref2 = {7,8,9};
        FORGE_ASSERT(test::check_arrays(g2, ref2, 0.0f, 0.0f, "chunk_part2"),
                     "chunk_part2: mismatch");

        parts.clear();
        t = Tensor();
        cudaFree(dd);
    }

    // ------ split dim=1 (2D) ------
    {
        // {2,6} split along dim=1 with split_size=2 → 3 parts of {2,2}
        std::vector<float> hd = {1,2,3,4,5,6, 7,8,9,10,11,12};
        float* dd = test::to_device(hd);
        Tensor t = Tensor::from_ptr(dd, {2, 6});

        auto parts = forge::split(t, 2, 1);
        FORGE_ASSERT(parts.size() == 3, "split_dim1: should have 3 parts");
        for (int i = 0; i < 3; i++) {
            FORGE_ASSERT(parts[i].size(0) == 2 && parts[i].size(1) == 2,
                         "split_dim1: each part should be {2,2}");
        }

        // Views from split on non-zero dim are non-contiguous; make contiguous to read
        Tensor c0 = parts[0].contiguous();
        Tensor c1 = parts[1].contiguous();
        Tensor c2 = parts[2].contiguous();
        cudaDeviceSynchronize();
        auto g0 = test::to_host(c0.data<float>(), 4);
        auto g1 = test::to_host(c1.data<float>(), 4);
        auto g2 = test::to_host(c2.data<float>(), 4);
        std::vector<float> ref0 = {1,2, 7,8};
        std::vector<float> ref1 = {3,4, 9,10};
        std::vector<float> ref2 = {5,6, 11,12};
        FORGE_ASSERT(test::check_arrays(g0, ref0, 0.0f, 0.0f, "split_dim1_p0"),
                     "split_dim1_p0: mismatch");
        FORGE_ASSERT(test::check_arrays(g1, ref1, 0.0f, 0.0f, "split_dim1_p1"),
                     "split_dim1_p1: mismatch");
        FORGE_ASSERT(test::check_arrays(g2, ref2, 0.0f, 0.0f, "split_dim1_p2"),
                     "split_dim1_p2: mismatch");

        parts.clear();
        t = Tensor();
        cudaFree(dd);
    }

    // ------ cat → split roundtrip ------
    {
        std::vector<float> ha = {1,2,3};
        std::vector<float> hb = {4,5};
        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {3});
        Tensor b = Tensor::from_ptr(db, {2});

        Tensor joined = forge::cat({a, b}, 0);
        auto parts = forge::split(joined, 3, 0);
        FORGE_ASSERT(parts.size() == 2, "roundtrip: should have 2 parts");
        FORGE_ASSERT(parts[0].size(0) == 3, "roundtrip: part0 size 3");
        FORGE_ASSERT(parts[1].size(0) == 2, "roundtrip: part1 size 2");

        cudaDeviceSynchronize();
        auto g0 = test::to_host(parts[0].data<float>(), 3);
        auto g1 = test::to_host(parts[1].data<float>(), 2);
        FORGE_ASSERT(test::check_arrays(g0, ha, 0.0f, 0.0f, "roundtrip_p0"),
                     "roundtrip_p0: mismatch");
        FORGE_ASSERT(test::check_arrays(g1, hb, 0.0f, 0.0f, "roundtrip_p1"),
                     "roundtrip_p1: mismatch");

        parts.clear();
        joined = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    // ------ cat with negative dim ------
    {
        std::vector<float> ha = {1,2, 3,4};
        std::vector<float> hb = {5, 6};
        float* da = test::to_device(ha);
        float* db = test::to_device(hb);
        Tensor a = Tensor::from_ptr(da, {2, 2});
        Tensor b = Tensor::from_ptr(db, {2, 1});

        Tensor out = forge::cat({a, b}, -1); // dim=-1 → dim=1
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.size(0) == 2 && out.size(1) == 3,
                     "cat_negdim: shape should be {2,3}");
        auto go = test::to_host(out.data<float>(), 6);
        std::vector<float> ref = {1,2,5, 3,4,6};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "cat_negdim"),
                     "cat_negdim: data mismatch");

        out = Tensor(); b = Tensor(); a = Tensor();
        cudaFree(da); cudaFree(db);
    }

    std::printf("ALL PASSED\n");
    return 0;
}
