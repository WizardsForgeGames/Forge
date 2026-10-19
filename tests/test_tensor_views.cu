// Forge GPU Compute Library -- Test
// Copyright (c) 2026 Darren Sander. MIT License.
//
// Tests all zero-copy view operations on Tensor.
#include <forge/tensor.h>
#include "test_helpers.cuh"

using namespace forge;

int main() {
    std::printf("=== Test: Tensor Views ===\n");

    // ------ view: reshape ------
    {
        Tensor t = Tensor::empty({12});
        t.fill_(1.0f);
        cudaDeviceSynchronize();

        Tensor v = t.view({3, 4});
        FORGE_ASSERT(v.numel() == 12, "view: numel should be 12");
        FORGE_ASSERT(v.ndim() == 2, "view: ndim should be 2");
        FORGE_ASSERT(v.size(0) == 3, "view: size(0) should be 3");
        FORGE_ASSERT(v.size(1) == 4, "view: size(1) should be 4");
        FORGE_ASSERT(v.data_ptr() == t.data_ptr(), "view: should share data_ptr");
        FORGE_ASSERT(v.is_contiguous(), "view: should be contiguous");
        // C-contiguous strides for {3,4} = [4, 1]
        FORGE_ASSERT(v.stride(0) == 4, "view: stride(0) should be 4");
        FORGE_ASSERT(v.stride(1) == 1, "view: stride(1) should be 1");

        std::printf("  PASS [view_reshape]\n");
    }

    // ------ transpose(0,1) ------
    {
        Tensor t = Tensor::empty({3, 4});
        Tensor tt = t.transpose(0, 1);
        FORGE_ASSERT(tt.size(0) == 4, "transpose: size(0) should be 4");
        FORGE_ASSERT(tt.size(1) == 3, "transpose: size(1) should be 3");
        // Original strides: [4, 1], after transpose: [1, 4]
        FORGE_ASSERT(tt.stride(0) == 1, "transpose: stride(0) should be 1");
        FORGE_ASSERT(tt.stride(1) == 4, "transpose: stride(1) should be 4");
        FORGE_ASSERT(!tt.is_contiguous(), "transpose: should be non-contiguous");
        FORGE_ASSERT(tt.data_ptr() == t.data_ptr(),
                     "transpose: should share data_ptr");

        std::printf("  PASS [transpose]\n");
    }

    // ------ slice(dim, start, length) ------
    {
        Tensor t = Tensor::empty({10});
        t.fill_(5.0f);
        cudaDeviceSynchronize();

        Tensor ts = t.slice(0, 2, 3); // elements [2,3,4]
        FORGE_ASSERT(ts.numel() == 3, "slice: numel should be 3");
        FORGE_ASSERT(ts.size(0) == 3, "slice: size(0) should be 3");
        // Offset: data_ptr should be shifted by 2 elements
        float* expected_ptr = t.data<float>() + 2;
        FORGE_ASSERT(ts.data<float>() == expected_ptr,
                     "slice: data_ptr should be offset by start");

        std::printf("  PASS [slice]\n");
    }

    // ------ unsqueeze(0) ------
    {
        Tensor t = Tensor::empty({3, 4});
        Tensor tu = t.unsqueeze(0);
        FORGE_ASSERT(tu.ndim() == 3, "unsqueeze(0): ndim should be 3");
        FORGE_ASSERT(tu.size(0) == 1, "unsqueeze(0): size(0) should be 1");
        FORGE_ASSERT(tu.size(1) == 3, "unsqueeze(0): size(1) should be 3");
        FORGE_ASSERT(tu.size(2) == 4, "unsqueeze(0): size(2) should be 4");
        FORGE_ASSERT(tu.data_ptr() == t.data_ptr(),
                     "unsqueeze(0): should share data_ptr");
        FORGE_ASSERT(tu.is_contiguous(), "unsqueeze(0): should be contiguous");

        std::printf("  PASS [unsqueeze_0]\n");
    }

    // ------ unsqueeze(-1) ------
    {
        Tensor t = Tensor::empty({3, 4});
        Tensor tu = t.unsqueeze(-1);
        FORGE_ASSERT(tu.ndim() == 3, "unsqueeze(-1): ndim should be 3");
        FORGE_ASSERT(tu.size(0) == 3, "unsqueeze(-1): size(0) should be 3");
        FORGE_ASSERT(tu.size(1) == 4, "unsqueeze(-1): size(1) should be 4");
        FORGE_ASSERT(tu.size(2) == 1, "unsqueeze(-1): size(2) should be 1");
        FORGE_ASSERT(tu.stride(2) == 1, "unsqueeze(-1): stride(2) should be 1");

        std::printf("  PASS [unsqueeze_neg1]\n");
    }

    // ------ squeeze(0): remove size-1 dim ------
    {
        Tensor t = Tensor::empty({1, 3, 4});
        Tensor ts = t.squeeze(0);
        FORGE_ASSERT(ts.ndim() == 2, "squeeze(0): ndim should be 2");
        FORGE_ASSERT(ts.size(0) == 3, "squeeze(0): size(0) should be 3");
        FORGE_ASSERT(ts.size(1) == 4, "squeeze(0): size(1) should be 4");
        FORGE_ASSERT(ts.data_ptr() == t.data_ptr(),
                     "squeeze(0): should share data_ptr");

        std::printf("  PASS [squeeze_0]\n");
    }

    // ------ squeeze non-1: no change ------
    {
        Tensor t = Tensor::empty({3, 4});
        Tensor ts = t.squeeze(0); // dim 0 has size 3, not 1
        FORGE_ASSERT(ts.ndim() == 2, "squeeze_non1: ndim should remain 2");
        FORGE_ASSERT(ts.size(0) == 3, "squeeze_non1: size(0) should remain 3");
        FORGE_ASSERT(ts.size(1) == 4, "squeeze_non1: size(1) should remain 4");

        std::printf("  PASS [squeeze_non1]\n");
    }

    // ------ expand: stride-0 broadcast ------
    {
        Tensor t = Tensor::empty({1, 4});
        Tensor te = t.expand({3, 4});
        FORGE_ASSERT(te.size(0) == 3, "expand: size(0) should be 3");
        FORGE_ASSERT(te.size(1) == 4, "expand: size(1) should be 4");
        FORGE_ASSERT(te.stride(0) == 0, "expand: stride(0) should be 0 (broadcast)");
        FORGE_ASSERT(te.stride(1) == 1, "expand: stride(1) should be 1");
        FORGE_ASSERT(!te.is_contiguous(), "expand: should be non-contiguous");
        FORGE_ASSERT(te.numel() == 12, "expand: numel should be 12");

        std::printf("  PASS [expand]\n");
    }

    // ------ reshape contiguous: returns view ------
    {
        Tensor t = Tensor::empty({12});
        Tensor r = t.reshape({3, 4});
        FORGE_ASSERT(r.numel() == 12, "reshape_contig: numel should be 12");
        FORGE_ASSERT(r.ndim() == 2, "reshape_contig: ndim should be 2");
        // Since input is contiguous, reshape returns a view sharing storage
        FORGE_ASSERT(r.storage().get() == t.storage().get(),
                     "reshape_contig: should share storage");

        std::printf("  PASS [reshape_contiguous]\n");
    }

    // ------ reshape non-contiguous: copies ------
    {
        Tensor t = Tensor::empty({3, 4});
        Tensor tt = t.transpose(0, 1); // {4,3} non-contiguous
        FORGE_ASSERT(!tt.is_contiguous(), "reshape_noncontig: transposed should be non-contiguous");

        Tensor r = tt.reshape({12});
        FORGE_ASSERT(r.numel() == 12, "reshape_noncontig: numel should be 12");
        FORGE_ASSERT(r.is_contiguous(), "reshape_noncontig: result should be contiguous");
        // Should have new storage (copy happened)
        FORGE_ASSERT(r.storage().get() != tt.storage().get(),
                     "reshape_noncontig: should have new storage (copy)");

        std::printf("  PASS [reshape_noncontiguous]\n");
    }

    // ------ View shares storage ------
    {
        Tensor t = Tensor::empty({3, 4});
        Tensor tt = t.transpose(0, 1);
        FORGE_ASSERT(tt.storage().get() == t.storage().get(),
                     "view_shares_storage: transpose should share storage");

        Tensor ts = t.slice(0, 0, 2);
        FORGE_ASSERT(ts.storage().get() == t.storage().get(),
                     "view_shares_storage: slice should share storage");

        Tensor tu = t.unsqueeze(0);
        FORGE_ASSERT(tu.storage().get() == t.storage().get(),
                     "view_shares_storage: unsqueeze should share storage");

        std::printf("  PASS [view_shares_storage]\n");
    }

    // ------ View is_view flag ------
    {
        Tensor t = Tensor::empty({3, 4});
        FORGE_ASSERT(!t.unsafeGetTensorImpl()->is_view(),
                     "is_view: base tensor should NOT be a view");

        Tensor tt = t.transpose(0, 1);
        FORGE_ASSERT(tt.unsafeGetTensorImpl()->is_view(),
                     "is_view: transposed should be a view");

        Tensor ts = t.slice(0, 0, 2);
        FORGE_ASSERT(ts.unsafeGetTensorImpl()->is_view(),
                     "is_view: sliced should be a view");

        std::printf("  PASS [view_is_view_flag]\n");
    }

    // ------ Chained views ------
    {
        Tensor t = Tensor::empty({4, 6});
        // transpose(0,1) -> {6,4} then slice(0, 0, 2) -> {2,4}
        Tensor v = t.transpose(0, 1).slice(0, 0, 2);
        FORGE_ASSERT(v.size(0) == 2, "chained: size(0) should be 2");
        FORGE_ASSERT(v.size(1) == 4, "chained: size(1) should be 4");
        FORGE_ASSERT(v.numel() == 8, "chained: numel should be 8");
        // After transpose {4,6} -> {6,4} strides: [1,6]
        // After slice dim0 [0,2): sizes {2,4} strides [1,6], offset 0
        FORGE_ASSERT(v.stride(0) == 1, "chained: stride(0) should be 1");
        FORGE_ASSERT(v.stride(1) == 6, "chained: stride(1) should be 6");
        FORGE_ASSERT(v.storage().get() == t.storage().get(),
                     "chained: should share storage through chain");

        std::printf("  PASS [chained_views]\n");
    }

    // ------ contiguous() on contiguous: returns self ------
    {
        Tensor t = Tensor::ones({5, 3});
        cudaDeviceSynchronize();
        FORGE_ASSERT(t.is_contiguous(), "contig_self: should be contiguous");

        Tensor tc = t.contiguous();
        FORGE_ASSERT(tc.data_ptr() == t.data_ptr(),
                     "contig_self: should return self (same data_ptr)");

        std::printf("  PASS [contiguous_on_contiguous]\n");
    }

    std::printf("ALL PASSED\n");
    return 0;
}
