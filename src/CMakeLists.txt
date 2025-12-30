find_package(PkgConfig)
pkg_search_module(BENCHMARK REQUIRED benchmark)

file(GLOB_RECURSE
        CP2K_SOURCE_FILES
        cp2k/*.cpp
        cp2k/*.f90
        cp2k/*.c
        # ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
)

# add_executable(bench_cp2k)
add_executable(bench_cp2k main.cpp)
target_sources(bench_cp2k PRIVATE ${CP2K_SOURCE_FILES})
target_link_libraries(bench_cp2k PRIVATE libcompression ${BENCHMARK_LDFLAGS} cp2k_compression)
target_include_directories(bench_cp2k PRIVATE "${CMAKE_BINARY_DIR}/fortran_modules/cp2k")
add_dependencies(bench_cp2k cp2k_compression)
target_compile_options(bench_cp2k PRIVATE ${BENCHMARK_CFLAGS})
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_cp2k PUBLIC OpenMP::OpenMP_CXX)
endif()