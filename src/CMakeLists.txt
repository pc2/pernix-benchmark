find_package(PkgConfig)
pkg_search_module(BENCHMARK REQUIRED benchmark)

option(PERNIX_BENCHMARK_USE_SIMDE "Build Pernix benchmark targets with SIMDe instead of target-specific x86 flags" OFF)
option(PERNIX_BENCHMARK_ENABLE_CUDA "Build CUDA PCIe end-to-end benchmark targets" OFF)

set(PERNIX_BENCHMARK_TARGET_IS_X86 OFF)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86|i686)$")
    set(PERNIX_BENCHMARK_TARGET_IS_X86 ON)
endif ()

set(PERNIX_BENCHMARK_TARGET_IS_ARM64 OFF)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(PERNIX_BENCHMARK_TARGET_IS_ARM64 ON)
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

find_package(OpenMP)
function(add_pernix_benchmark variant compile_opts)
    file(GLOB_RECURSE
            SOURCE_FILES
            pernix/${variant}/*.cpp
            ${CMAKE_SOURCE_DIR}/external/pernix/include/*.h
    )
    message(STATUS ${SOURCE_FILES})

    add_executable(bench_pernix_${variant} main.cpp ${SOURCE_FILES})
    target_link_libraries(bench_pernix_${variant} PRIVATE
            pernix_benchmark_optimization
            ${BENCHMARK_LDFLAGS})
    if (PERNIX_BENCHMARK_USE_SIMDE)
        target_compile_definitions(bench_pernix_${variant} PRIVATE PERNIX_USE_SIMDE=1)
        target_compile_options(bench_pernix_${variant} PRIVATE -g ${BENCHMARK_CFLAGS})
        target_include_directories(bench_pernix_${variant} PRIVATE ${PERNIX_BENCHMARK_SIMDE_INCLUDE_DIRS})
    else ()
        target_compile_options(bench_pernix_${variant} PRIVATE ${compile_opts} ${ARGN} ${BENCHMARK_CFLAGS})
    endif ()
    target_include_directories(bench_pernix_${variant} PRIVATE ${CMAKE_SOURCE_DIR}/external/pernix/include)
    target_include_directories(bench_pernix_${variant} PRIVATE pernix/include)
endfunction()

function(add_pernix_pcie_benchmark variant compile_opts)
    add_executable(bench_pcie_${variant} main.cpp pcie/pcie_${variant}.cpp)
    target_link_libraries(bench_pcie_${variant} PRIVATE
            pernix_benchmark_optimization
            ${BENCHMARK_LDFLAGS}
            CUDA::cudart)
    target_compile_options(bench_pcie_${variant} PRIVATE ${compile_opts} ${ARGN} ${BENCHMARK_CFLAGS})
    target_include_directories(bench_pcie_${variant} PRIVATE
            ${CMAKE_SOURCE_DIR}/external/pernix/include
            pernix/include)
endfunction()
if (PERNIX_BENCHMARK_USE_SIMDE OR PERNIX_BENCHMARK_TARGET_IS_X86)
    add_pernix_benchmark(avx512vbmi -g -march=x86-64-v4 -mavx512vbmi)
    add_pernix_benchmark(avx2 -g -march=x86-64-v3)
    add_pernix_benchmark(bmi2 -g -march=x86-64-v3)
endif ()
if (PERNIX_BENCHMARK_TARGET_IS_ARM64)
    add_pernix_benchmark(neon -g -march=armv8-a+simd)
    add_pernix_benchmark(sve2 -g -march=armv8.2-a+simd+sve2)
endif ()
add_pernix_benchmark(fallback -g)

if (PERNIX_BENCHMARK_ENABLE_CUDA)
    if (NOT PERNIX_BENCHMARK_TARGET_IS_X86)
        message(FATAL_ERROR "CUDA PCIe benchmarks currently support x86 Pernix implementations only.")
    endif ()
    find_package(CUDAToolkit REQUIRED)
    add_pernix_pcie_benchmark(fallback -g)
    add_pernix_pcie_benchmark(avx2 -g -march=x86-64-v3)
    add_pernix_pcie_benchmark(bmi2 -g -march=x86-64-v3)
    add_pernix_pcie_benchmark(avx512vbmi -g -march=x86-64-v4 -mavx512vbmi)
endif ()

add_library(cp2k_benchmark_adapter STATIC cp2k/cp2k_benchmarks.f90)
target_link_libraries(cp2k_benchmark_adapter PRIVATE
        pernix_benchmark_optimization
        cp2k_compression)
target_include_directories(cp2k_benchmark_adapter PRIVATE "${CMAKE_BINARY_DIR}/fortran_modules/cp2k")
add_dependencies(cp2k_benchmark_adapter cp2k_compression)
target_compile_options(cp2k_benchmark_adapter PRIVATE
        $<$<COMPILE_LANGUAGE:Fortran>:-ffree-form -ffree-line-length-none -g -march=native>
)
set_target_properties(cp2k_benchmark_adapter PROPERTIES
        Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/fortran_modules/cp2k_benchmark_adapter"
)

add_executable(bench_cp2k main.cpp cp2k/cp2k_benchmarks.cpp)
target_link_libraries(bench_cp2k PRIVATE
        pernix_benchmark_optimization
        ${BENCHMARK_LDFLAGS}
        cp2k_benchmark_adapter)
target_include_directories(bench_cp2k PRIVATE ${CMAKE_SOURCE_DIR}/external/pernix/include)
target_include_directories(bench_cp2k PRIVATE pernix/include)
target_compile_options(bench_cp2k PRIVATE ${BENCHMARK_CFLAGS} -g -march=native)
if (OpenMP_CXX_FOUND)
    target_link_libraries(bench_cp2k PUBLIC OpenMP::OpenMP_CXX)
endif ()

if (BUILD_TESTING)
    add_executable(cp2k_benchmark_adapter_test ../tests/cp2k_benchmark_adapter_test.cpp)
    target_link_libraries(cp2k_benchmark_adapter_test PRIVATE cp2k_benchmark_adapter)
    add_test(NAME cp2k_benchmark_adapter_test COMMAND cp2k_benchmark_adapter_test)
endif ()
