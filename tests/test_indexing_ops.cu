// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Tests indexing and selection operations.
#include <forge/tensor.h>
#include <forge/tensor_ops.h>
#include <forge/indexing_ops.h>
#include "test_helpers.cuh"
#include <cmath>

using namespace forge;

int main() {
    std::printf("=== Test: Indexing & Selection Ops ===\n");

    // ------ index_select dim=0: select rows ------
    {
        // input: {5,3}
        //   row0: [1,2,3]  row1: [4,5,6]  row2: [7,8,9]
        //   row3: [10,11,12]  row4: [13,14,15]
        std::vector<float> hi = {1,2,3, 4,5,6, 7,8,9, 10,11,12, 13,14,15};
        std::vector<int64_t> hk = {0, 2, 4};  // select rows 0, 2, 4
        float* di = test::to_device(hi);
        int64_t* dk = test::to_device(hk);
        Tensor input = Tensor::from_ptr(di, {5, 3});
        Tensor idx = Tensor::from_blob(dk, {3}, DType::Int64);

        Tensor out = forge::index_select(input, 0, idx);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.size(0) == 3 && out.size(1) == 3,
                     "index_select_dim0: shape should be {3,3}");
        auto go = test::to_host(out.data<float>(), 9);
        std::vector<float> ref = {1,2,3, 7,8,9, 13,14,15};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "index_select_dim0"),
                     "index_select_dim0: data mismatch");

        out = Tensor(); idx = Tensor(); input = Tensor();
        cudaFree(di); cudaFree(dk);
    }

    // ------ index_select dim=1: select columns ------
    {
        // input: {3,5}
        std::vector<float> hi = {1,2,3,4,5, 6,7,8,9,10, 11,12,13,14,15};
        std::vector<int64_t> hk = {1, 3};  // select cols 1, 3
        float* di = test::to_device(hi);
        int64_t* dk = test::to_device(hk);
        Tensor input = Tensor::from_ptr(di, {3, 5});
        Tensor idx = Tensor::from_blob(dk, {2}, DType::Int64);

        Tensor out = forge::index_select(input, 1, idx);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.size(0) == 3 && out.size(1) == 2,
                     "index_select_dim1: shape should be {3,2}");
        auto go = test::to_host(out.data<float>(), 6);
        // row0: cols[1,3] = [2,4]  row1: [7,9]  row2: [12,14]
        std::vector<float> ref = {2,4, 7,9, 12,14};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "index_select_dim1"),
                     "index_select_dim1: data mismatch");

        out = Tensor(); idx = Tensor(); input = Tensor();
        cudaFree(di); cudaFree(dk);
    }

    // ------ gather 2D ------
    {
        // input: {3,4}, gather along dim=1
        // [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
        // index: {3,2} = [[0,2],[1,3],[2,0]]
        // out[i][j] = input[i][index[i][j]]
        std::vector<float> hi = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        std::vector<int64_t> hk = {0,2, 1,3, 2,0};
        float* di = test::to_device(hi);
        int64_t* dk = test::to_device(hk);
        Tensor input = Tensor::from_ptr(di, {3, 4});
        Tensor idx = Tensor::from_blob(dk, {3, 2}, DType::Int64);

        Tensor out = forge::gather(input, 1, idx);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.size(0) == 3 && out.size(1) == 2,
                     "gather_2d: shape should be {3,2}");
        auto go = test::to_host(out.data<float>(), 6);
        // row0: input[0][0]=1, input[0][2]=3
        // row1: input[1][1]=6, input[1][3]=8
        // row2: input[2][2]=11, input[2][0]=9
        std::vector<float> ref = {1,3, 6,8, 11,9};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "gather_2d"),
                     "gather_2d: data mismatch");

        out = Tensor(); idx = Tensor(); input = Tensor();
        cudaFree(di); cudaFree(dk);
    }

    // ------ gather 3D, dim=1 ------
    {
        // input: {2,3,2}
        // [[[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]]]
        // index: {2,2,2}, gather along dim=1
        // [[[ 0,1],[2,0]], [[1,2],[0,1]]]
        std::vector<float> hi = {1,2, 3,4, 5,6, 7,8, 9,10, 11,12};
        std::vector<int64_t> hk = {0,1, 2,0, 1,2, 0,1};
        float* di = test::to_device(hi);
        int64_t* dk = test::to_device(hk);
        Tensor input = Tensor::from_ptr(di, {2, 3, 2});
        Tensor idx = Tensor::from_blob(dk, {2, 2, 2}, DType::Int64);

        Tensor out = forge::gather(input, 1, idx);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.size(0) == 2 && out.size(1) == 2 && out.size(2) == 2,
                     "gather_3d: shape should be {2,2,2}");
        auto go = test::to_host(out.data<float>(), 8);
        // batch0: out[0][0][0] = input[0][idx=0][0] = 1
        //         out[0][0][1] = input[0][idx=1][1] = 4
        //         out[0][1][0] = input[0][idx=2][0] = 5
        //         out[0][1][1] = input[0][idx=0][1] = 2
        // batch1: out[1][0][0] = input[1][idx=1][0] = 9
        //         out[1][0][1] = input[1][idx=2][1] = 12
        //         out[1][1][0] = input[1][idx=0][0] = 7
        //         out[1][1][1] = input[1][idx=1][1] = 10
        std::vector<float> ref = {1,4, 5,2, 9,12, 7,10};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "gather_3d"),
                     "gather_3d: data mismatch");

        out = Tensor(); idx = Tensor(); input = Tensor();
        cudaFree(di); cudaFree(dk);
    }

    // ------ scatter 2D ------
    {
        // input: {3,4} all zeros
        // index: {3,2} = [[0,2],[1,3],[2,0]], src: {3,2} = [[10,20],[30,40],[50,60]]
        // scatter along dim=1: out[i][index[i][j]] = src[i][j]
        std::vector<float> hi(12, 0.0f);
        std::vector<int64_t> hk = {0,2, 1,3, 2,0};
        std::vector<float> hs = {10,20, 30,40, 50,60};
        float* di = test::to_device(hi);
        int64_t* dk = test::to_device(hk);
        float* ds = test::to_device(hs);
        Tensor input = Tensor::from_ptr(di, {3, 4});
        Tensor idx = Tensor::from_blob(dk, {3, 2}, DType::Int64);
        Tensor src = Tensor::from_ptr(ds, {3, 2});

        Tensor out = forge::scatter(input, 1, idx, src);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.size(0) == 3 && out.size(1) == 4,
                     "scatter_2d: shape should be {3,4}");
        auto go = test::to_host(out.data<float>(), 12);
        // row0: [10,0,20,0]  row1: [0,30,0,40]  row2: [60,0,50,0]
        std::vector<float> ref = {10,0,20,0, 0,30,0,40, 60,0,50,0};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "scatter_2d"),
                     "scatter_2d: data mismatch");

        out = Tensor(); src = Tensor(); idx = Tensor(); input = Tensor();
        cudaFree(di); cudaFree(dk); cudaFree(ds);
    }

    // ------ scatter_add_ (duplicate indices accumulate) ------
    {
        // self: {1,5} = [[0,0,0,0,0]]
        // index: {1,4} = [[1,1,3,3]]  (duplicates!)
        // src:   {1,4} = [[10,20,30,40]]
        // After scatter_add_ dim=1: self[0][1] += 10+20=30, self[0][3] += 30+40=70
        std::vector<float> hself(5, 0.0f);
        std::vector<int64_t> hk = {1,1,3,3};
        std::vector<float> hs = {10,20,30,40};
        float* dself = test::to_device(hself);
        int64_t* dk = test::to_device(hk);
        float* ds = test::to_device(hs);
        Tensor self = Tensor::from_ptr(dself, {1, 5});
        Tensor idx = Tensor::from_blob(dk, {1, 4}, DType::Int64);
        Tensor src = Tensor::from_ptr(ds, {1, 4});

        forge::scatter_add_(self, 1, idx, src);
        cudaDeviceSynchronize();
        auto go = test::to_host(self.data<float>(), 5);
        std::vector<float> ref = {0, 30, 0, 70, 0};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "scatter_add"),
                     "scatter_add: data mismatch");

        src = Tensor(); idx = Tensor(); self = Tensor();
        cudaFree(dself); cudaFree(dk); cudaFree(ds);
    }

    // ------ masked_fill_ (same shape) ------
    {
        std::vector<float> hd = {1, 2, 3, 4, 5, 6};
        std::vector<uint8_t> hm = {1, 0, 1, 0, 1, 0};
        float* dd = test::to_device(hd);
        uint8_t* dm; cudaMalloc(&dm, 6); cudaMemcpy(dm, hm.data(), 6, cudaMemcpyHostToDevice);
        Tensor data = Tensor::from_ptr(dd, {2, 3});
        Tensor mask = Tensor::from_blob(dm, {2, 3}, DType::Bool);

        forge::masked_fill_(data, mask, -99.0f);
        cudaDeviceSynchronize();
        auto go = test::to_host(data.data<float>(), 6);
        std::vector<float> ref = {-99, 2, -99, 4, -99, 6};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "masked_fill"),
                     "masked_fill: data mismatch");

        mask = Tensor(); data = Tensor();
        cudaFree(dd); cudaFree(dm);
    }

    // ------ masked_fill_ with broadcasting: {3,4} tensor, {4} mask ------
    {
        // data: {3,4}, mask: {4} = [T,F,F,T] — broadcasts along dim0
        std::vector<float> hd = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        std::vector<uint8_t> hm = {1, 0, 0, 1};
        float* dd = test::to_device(hd);
        uint8_t* dm; cudaMalloc(&dm, 4); cudaMemcpy(dm, hm.data(), 4, cudaMemcpyHostToDevice);
        Tensor data = Tensor::from_ptr(dd, {3, 4});
        Tensor mask = Tensor::from_blob(dm, {4}, DType::Bool);

        forge::masked_fill_(data, mask, 0.0f);
        cudaDeviceSynchronize();
        auto go = test::to_host(data.data<float>(), 12);
        // cols 0,3 filled with 0: [0,2,3,0, 0,6,7,0, 0,10,11,0]
        std::vector<float> ref = {0,2,3,0, 0,6,7,0, 0,10,11,0};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "masked_fill_broadcast"),
                     "masked_fill_broadcast: data mismatch");

        mask = Tensor(); data = Tensor();
        cudaFree(dd); cudaFree(dm);
    }

    // ------ masked_select ------
    {
        std::vector<float> hd = {10, 20, 30, 40, 50};
        std::vector<uint8_t> hm = {1, 0, 1, 0, 1};
        float* dd = test::to_device(hd);
        uint8_t* dm; cudaMalloc(&dm, 5); cudaMemcpy(dm, hm.data(), 5, cudaMemcpyHostToDevice);
        Tensor data = Tensor::from_ptr(dd, {5});
        Tensor mask = Tensor::from_blob(dm, {5}, DType::Bool);

        Tensor out = forge::masked_select(data, mask);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.numel() == 3, "masked_select: should have 3 elements");
        auto go = test::to_host(out.data<float>(), 3);
        // Elements where mask is true: 10, 30, 50
        // Note: atomicAdd doesn't guarantee order, so sort both
        std::sort(go.begin(), go.end());
        std::vector<float> ref = {10, 30, 50};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "masked_select"),
                     "masked_select: data mismatch");

        out = Tensor(); mask = Tensor(); data = Tensor();
        cudaFree(dd); cudaFree(dm);
    }

    // ------ zero_() ------
    {
        std::vector<float> hd = {1, 2, 3, 4};
        float* dd = test::to_device(hd);
        Tensor data = Tensor::from_ptr(dd, {4});

        data.zero_();
        cudaDeviceSynchronize();
        auto go = test::to_host(data.data<float>(), 4);
        std::vector<float> ref = {0, 0, 0, 0};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "zero_"),
                     "zero_: data mismatch");

        data = Tensor();
        cudaFree(dd);
    }

    // ------ copy_() ------
    {
        std::vector<float> ha = {10, 20, 30, 40, 50, 60};
        float* da = test::to_device(ha);
        Tensor src = Tensor::from_ptr(da, {2, 3});
        Tensor dst = Tensor::empty({2, 3});

        dst.copy_(src);
        cudaDeviceSynchronize();
        auto go = test::to_host(dst.data<float>(), 6);
        FORGE_ASSERT(test::check_arrays(go, ha, 0.0f, 0.0f, "copy_"),
                     "copy_: data mismatch");

        dst = Tensor(); src = Tensor();
        cudaFree(da);
    }

    // ------ index_select on non-contiguous (transposed) input ------
    {
        // Create {3,4}, transpose to {4,3}, then index_select dim=0
        std::vector<float> hi = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        float* di = test::to_device(hi);
        Tensor base = Tensor::from_ptr(di, {3, 4});
        Tensor input = base.transpose(0, 1);  // {4,3}, non-contiguous

        std::vector<int64_t> hk = {0, 2};
        int64_t* dk = test::to_device(hk);
        Tensor idx = Tensor::from_blob(dk, {2}, DType::Int64);

        Tensor out = forge::index_select(input, 0, idx);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.size(0) == 2 && out.size(1) == 3,
                     "index_select_noncontig: shape should be {2,3}");
        auto go = test::to_host(out.data<float>(), 6);
        // Transposed input rows:
        //   row0: [1,5,9]  row1: [2,6,10]  row2: [3,7,11]  row3: [4,8,12]
        // Select rows 0,2: [1,5,9, 3,7,11]
        std::vector<float> ref = {1,5,9, 3,7,11};
        FORGE_ASSERT(test::check_arrays(go, ref, 0.0f, 0.0f, "index_select_noncontig"),
                     "index_select_noncontig: data mismatch");

        out = Tensor(); idx = Tensor(); input = Tensor(); base = Tensor();
        cudaFree(di); cudaFree(dk);
    }

    // ------ Empty index tensor ------
    {
        std::vector<float> hi = {1, 2, 3, 4};
        float* di = test::to_device(hi);
        Tensor input = Tensor::from_ptr(di, {4});

        // Empty index: select nothing
        Tensor idx = Tensor::empty({0}, DType::Int64);
        Tensor out = forge::index_select(input, 0, idx);
        cudaDeviceSynchronize();
        FORGE_ASSERT(out.numel() == 0, "empty_index: should have 0 elements");

        out = Tensor(); idx = Tensor(); input = Tensor();
        cudaFree(di);
    }

    std::printf("ALL PASSED\n");
    return 0;
}
