find_package(PkgConfig)
pkg_search_module(BENCHMARK REQUIRED benchmark)

option(PERNIX_BENCHMARK_USE_SIMDE "Build Pernix benchmark targets with SIMDe instead of target-specific x86 flags" OFF)

set(PERNIX_BENCHMARK_TARGET_IS_X86 OFF)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86|i686)$")
    set(PERNIX_BENCHMARK_TARGET_IS_X86 ON)
endif ()

if (PERNIX_BENCHMARK_USE_SIMDE)
    pkg_search_module(SIMDE QUIET simde)

    set(PERNIX_BENCHMARK_SIMDE_INCLUDE_DIRS "")
    if (SIMDE_FOUND)
        list(APPEND PERNIX_BENCHMARK_SIMDE_INCLUDE_DIRS ${SIMDE_INCLUDE_DIRS})
    endif ()

    set(PERNIX_BENCHMARK_VENDORED_SIMDE_DIR "${CMAKE_SOURCE_DIR}/external/pernix/external/simde")
    if (EXISTS "${PERNIX_BENCHMARK_VENDORED_SIMDE_DIR}/simde/x86/avx2.h")
        list(APPEND PERNIX_BENCHMARK_SIMDE_INCLUDE_DIRS "${PERNIX_BENCHMARK_VENDORED_SIMDE_DIR}")
    endif ()

    if (NOT PERNIX_BENCHMARK_SIMDE_INCLUDE_DIRS)
        message(FATAL_ERROR "PERNIX_BENCHMARK_USE_SIMDE=ON requires SIMDe headers. "
                "Initialize the nested submodule with "
                "`git submodule update --init --recursive external/pernix/external/simde`, "
                "or install SIMDe so pkg-config can find it. "
                "Checked vendored path: ${PERNIX_BENCHMARK_VENDORED_SIMDE_DIR}/simde/x86/avx2.h")
    endif ()
endif ()

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
    if (PERNIX_BENCHMARK_USE_SIMDE)
        target_compile_definitions(bench_pernix_${variant} PRIVATE PERNIX_USE_SIMDE=1)
        target_compile_options(bench_pernix_${variant} PRIVATE -g -v ${BENCHMARK_CFLAGS})
        target_include_directories(bench_pernix_${variant} PRIVATE ${PERNIX_BENCHMARK_SIMDE_INCLUDE_DIRS})
    else ()
        target_compile_options(bench_pernix_${variant} PRIVATE ${compile_opts} ${ARGN} ${BENCHMARK_CFLAGS})
    endif ()
    target_include_directories(bench_pernix_${variant} PRIVATE ${CMAKE_SOURCE_DIR}/external/pernix/include)
    target_include_directories(bench_pernix_${variant} PRIVATE pernix/include)
endfunction()
if (PERNIX_BENCHMARK_USE_SIMDE OR PERNIX_BENCHMARK_TARGET_IS_X86)
    add_pernix_benchmark(avx512vbmi -g -v -march=x86-64-v4 -mavx512vbmi)
    add_pernix_benchmark(avx2 -g -v -march=x86-64-v3)
    add_pernix_benchmark(bmi2 -g -v -march=x86-64-v3)
endif ()
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
