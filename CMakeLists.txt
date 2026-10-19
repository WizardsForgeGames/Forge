cmake_minimum_required(VERSION 3.18)
project(Forge VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Use static CRT (/MT) so Forge links into /MT host applications
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)

# Target architectures (sm_80+ required: the sparse GEMM uses cp.async and mma.sp):
#   sm_80: A100 (Ampere)
#   sm_86: RTX 30-series (Ampere)
#   sm_89: RTX 40-series (Ada) -- the only tested target
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89" CACHE STRING "CUDA architectures" FORCE)
endif()

# Options
option(FORGE_BUILD_TESTS      "Build unit tests"  OFF)
option(FORGE_BUILD_BENCHMARKS "Build benchmarks"  OFF)

# ---------- Forge static library ----------
add_library(forge STATIC
    src/optim/adam.cu
    src/optim/grad_clip.cu
    src/optim/lr_schedule.cpp
    src/nn/activations.cu
    src/nn/softmax_ce.cu
    src/nn/fused_gemm_softmax_ce.cu
    src/nn/embedding.cu
    src/nn/bias.cu
    src/quant/quantize.cu
    src/quant/dequantize.cu
    src/gemm/int8_dense.cu
    src/gemm/int8_dp4a.cu
    src/gemm/int8_gemm.cu
    src/gemm/int8_sparse.cu
    src/gemm/fp32_tiled.cu
    src/nn/gating.cu
    src/nn/trilinear.cu
    src/nn/mse.cu
    src/nn/linear.cu
    src/nn/conv2d.cu
    src/nn/int8_conv2d.cu
    src/nn/int8_depthwise_conv2d.cu
    src/nn/int8_conv_transpose2d.cu
    src/nn/reduce.cu
    src/nn/rng.cu
    src/nn/fused_mlp.cu
    src/sparsity/prune_compress.cu
    src/sparsity/pack_metadata.cu
    src/nn/conv3d_winograd.cu
    src/sim/thermal.cu
    src/sim/thermal_datagen.cu
    src/sim/stefan.cu
    src/sim/stefan_datagen.cu
    src/graph.cu
    src/stream.cpp
    src/allocator.cu
    src/storage.cpp
    src/tensor_impl.cpp
    src/tensor.cpp
    src/tensor_kernels.cu
    src/tensor_ops.cu
    src/comparison_ops.cu
    src/reduction_ops.cu
    src/indexing_ops.cu
    src/concat_ops.cu
    src/autograd.cpp
    src/autograd_ops.cu
)

target_include_directories(forge
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/internal
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(forge PUBLIC CUDA::cudart CUDA::curand)

# NVCC flags
target_compile_options(forge PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<NOT:$<CONFIG:Debug>>:-O3>
        --use_fast_math
        --ptxas-options=-v
        --expt-relaxed-constexpr
    >
)

# Platform-specific CXX flags
if(MSVC)
    target_compile_options(forge PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:/O2>
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(forge PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    )
endif()

set_target_properties(forge PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ---------- Install ----------
include(GNUInstallDirs)
install(TARGETS forge EXPORT ForgeTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/forge/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/forge)
install(EXPORT ForgeTargets FILE ForgeTargets.cmake NAMESPACE Forge::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Forge)

# ---------- Tests ----------
if(FORGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------- Benchmarks ----------
if(FORGE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
