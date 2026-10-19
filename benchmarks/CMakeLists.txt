function(forge_add_bench BENCH_NAME BENCH_SOURCE)
    add_executable(${BENCH_NAME} ${BENCH_SOURCE})
    target_link_libraries(${BENCH_NAME} PRIVATE forge)
    set_target_properties(${BENCH_NAME} PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endfunction()

# Benchmarks will be added in Phase 7.
