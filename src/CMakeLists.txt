find_package(PkgConfig)
pkg_search_module(BENCHMARK REQUIRED benchmark)

file(GLOB_RECURSE
        CP2K_SOURCE_FILES
        cp2k/*.cpp
        cp2k/*.f90
        cp2k/*.c
)

list(APPEND PERNIX_VARIANTS
        fallback
        avx2
        bmi2
        avx512vbmi
)

foreach (variant IN LISTS PERNIX_VARIANTS)
    file(GLOB_RECURSE
            SOURCE_FILES
            pernix/${variant}/*.cpp
    )

    add_executable(bench_pernix_${variant} main.cpp)
    target_sources(bench_pernix_${variant} PRIVATE ${SOURCE_FILES})
    target_link_libraries(bench_pernix_${variant} PRIVATE libcompression ${BENCHMARK_LDFLAGS})
    target_compile_options(bench_pernix_${variant} PRIVATE ${BENCHMARK_CFLAGS})
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(bench_pernix_${variant} PUBLIC OpenMP::OpenMP_CXX)
    endif ()
    target_include_directories(bench_pernix_${variant} PRIVATE pernix/include)
endforeach ()
target_compile_options(libcompression PUBLIC -v -g -march=native -funroll-loops)
target_compile_options(bench_pernix_fallback PUBLIC -g -march=x86-64)
target_compile_options(bench_pernix_avx2 PUBLIC -g -march=x86-64-v2)
target_compile_options(bench_pernix_bmi2 PUBLIC -g -march=x86-64-v2)
target_compile_options(bench_pernix_avx512vbmi PUBLIC -g -march=x86-64-v3 -mavx512vbmi)

# add_executable(bench_cp2k)
add_executable(bench_cp2k main.cpp)
target_sources(bench_cp2k PRIVATE ${CP2K_SOURCE_FILES})
target_link_libraries(bench_cp2k PRIVATE libcompression ${BENCHMARK_LDFLAGS} cp2k_compression)
target_include_directories(bench_cp2k PRIVATE "${CMAKE_BINARY_DIR}/fortran_modules/cp2k")
add_dependencies(bench_cp2k cp2k_compression)
target_compile_options(bench_cp2k PRIVATE ${BENCHMARK_CFLAGS} -g -march=native)
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(bench_cp2k PUBLIC OpenMP::OpenMP_CXX)
endif ()