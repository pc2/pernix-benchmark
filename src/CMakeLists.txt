find_package(PkgConfig)
pkg_search_module(BENCHMARK REQUIRED benchmark)

file(GLOB_RECURSE
        CP2K_SOURCE_FILES
        cp2k/*.cpp
        cp2k/*.f90
        cp2k/*.c
)

find_package(OpenMP)
function(add_pernix_benchmark variant compile_opts)
    file(GLOB_RECURSE
            SOURCE_FILES
            pernix/${variant}/*.cpp
            ${CMAKE_SOURCE_DIR}/external/pernix/include/*.h
    )
    message(STATUS ${SOURCE_FILES})

    add_executable(bench_pernix_${variant} main.cpp ${SOURCE_FILES})
    target_link_libraries(bench_pernix_${variant} PRIVATE ${BENCHMARK_LDFLAGS})
    target_compile_options(bench_pernix_${variant} PRIVATE ${compile_opts} ${ARGN} ${BENCHMARK_CFLAGS})
    target_include_directories(bench_pernix_${variant} PRIVATE ${CMAKE_SOURCE_DIR}/external/pernix/include)
    target_include_directories(bench_pernix_${variant} PRIVATE pernix/include)
endfunction()
add_pernix_benchmark(avx512vbmi -g -v -march=x86-64-v4 -mavx512vbmi)
add_pernix_benchmark(avx2 -g -v -march=x86-64-v3)
add_pernix_benchmark(bmi2 -g -v -march=x86-64-v3)
add_pernix_benchmark(fallback -g -v -march=x86-64)

# add_executable(bench_cp2k)
add_executable(bench_cp2k main.cpp)
target_sources(bench_cp2k PRIVATE ${CP2K_SOURCE_FILES})
target_link_libraries(bench_cp2k PRIVATE ${BENCHMARK_LDFLAGS} cp2k_compression)
target_include_directories(bench_cp2k PRIVATE "${CMAKE_BINARY_DIR}/fortran_modules/cp2k")
add_dependencies(bench_cp2k cp2k_compression)
target_compile_options(bench_cp2k PRIVATE ${BENCHMARK_CFLAGS} -g -march=native)
if (OpenMP_CXX_FOUND)
    target_link_libraries(bench_cp2k PUBLIC OpenMP::OpenMP_CXX)
endif ()