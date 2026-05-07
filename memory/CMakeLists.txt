cmake_minimum_required(VERSION 3.30)
project(membench LANGUAGES C CXX)

include(FetchContent)
include(CheckCXXSourceCompiles)

find_package(Threads REQUIRED)

FetchContent_Declare(
        cli11_proj
        QUIET
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.6.2
)
FetchContent_MakeAvailable(cli11_proj)

set(MEMBENCH_SOURCES
        src/main.cpp
        src/benchmark.cpp
        src/cpu.cpp

        src/kernel/load/load.cpp
        src/kernel/load/load_validator.cpp
        src/kernel/copy/copy.cpp
        src/kernel/copy/copy_validator.cpp
        src/kernel/ddot/ddot.cpp
        src/kernel/ddot/ddot_validator.cpp
        src/kernel/daxpy/daxpy.cpp
        src/kernel/daxpy/daxpy_validator.cpp
        src/kernel/store/store.cpp
        src/kernel/store/store_validator.cpp
        src/kernel/update/update.cpp
        src/kernel/update/update_validator.cpp
        src/kernel/triad/triad.cpp
        src/kernel/triad/triad_validator.cpp
)

set(MEMBENCH_NATIVE_FLAG "")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(MEMBENCH_NATIVE_FLAG "-march=native")
endif ()

set(CMAKE_REQUIRED_QUIET TRUE)
set(MEMBENCH_SAVED_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${MEMBENCH_NATIVE_FLAG}")

check_cxx_source_compiles("
    #include <immintrin.h>
    int main() {
        __m128i v = _mm_setzero_si128();
        return static_cast<int>(_mm_cvtsi128_si32(v));
    }
" MEMBENCH_HAS_SSE2)

check_cxx_source_compiles("
    #include <immintrin.h>
    int main() {
        __m256i v = _mm256_setzero_si256();
        return static_cast<int>(_mm256_extract_epi32(v, 0));
    }
" MEMBENCH_HAS_AVX2)

check_cxx_source_compiles("
    #include <immintrin.h>
    int main() {
        __m512i v = _mm512_setzero_si512();
        return static_cast<int>(_mm_cvtsi128_si32(_mm512_castsi512_si128(v)));
    }
" MEMBENCH_HAS_AVX512F)

check_cxx_source_compiles("
    #include <arm_neon.h>
    int main() {
        double data[2] = {};
        float64x2_t v = vld1q_f64(data);
        vst1q_f64(data, v);
        return 0;
    }
" MEMBENCH_HAS_NEON_F64)

check_cxx_source_compiles("
    #include <arm_sve.h>
    int main() {
        return static_cast<int>(svcntd());
    }
" MEMBENCH_HAS_SVE)

set(CMAKE_REQUIRED_FLAGS "${MEMBENCH_SAVED_REQUIRED_FLAGS}")

if (MEMBENCH_HAS_SSE2)
    list(APPEND MEMBENCH_SOURCES src/kernel/load/load_sse.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/copy/copy_sse.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/ddot/ddot_sse.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/daxpy/daxpy_sse.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/store/store_sse.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/update/update_sse.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/triad/triad_sse.cpp)
endif ()
if (MEMBENCH_HAS_AVX2)
    list(APPEND MEMBENCH_SOURCES src/kernel/load/load_avx2.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/copy/copy_avx2.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/ddot/ddot_avx2.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/daxpy/daxpy_avx2.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/store/store_avx2.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/update/update_avx2.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/triad/triad_avx2.cpp)
endif ()
if (MEMBENCH_HAS_AVX512F)
    list(APPEND MEMBENCH_SOURCES src/kernel/load/load_avx512.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/copy/copy_avx512.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/ddot/ddot_avx512.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/daxpy/daxpy_avx512.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/store/store_avx512.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/update/update_avx512.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/triad/triad_avx512.cpp)
endif ()
if (MEMBENCH_HAS_NEON_F64)
    list(APPEND MEMBENCH_SOURCES src/kernel/load/load_neon.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/copy/copy_neon.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/ddot/ddot_neon.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/daxpy/daxpy_neon.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/store/store_neon.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/update/update_neon.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/triad/triad_neon.cpp)
endif ()
if (MEMBENCH_HAS_SVE)
    list(APPEND MEMBENCH_SOURCES src/kernel/load/load_sve.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/copy/copy_sve.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/ddot/ddot_sve.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/daxpy/daxpy_sve.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/store/store_sve.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/update/update_sve.cpp)
    list(APPEND MEMBENCH_SOURCES src/kernel/triad/triad_sve.cpp)
endif ()

add_executable(membench ${MEMBENCH_SOURCES})

target_include_directories(membench PRIVATE include)
target_compile_features(membench PRIVATE cxx_std_20)
target_compile_options(membench PRIVATE -O3)
if (MEMBENCH_NATIVE_FLAG)
    target_compile_options(membench PRIVATE "${MEMBENCH_NATIVE_FLAG}")
endif ()
target_link_libraries(membench PRIVATE Threads::Threads)
target_link_libraries(membench PRIVATE CLI11::CLI11)
