#include <membench/benchmark.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void copy_avx512_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    double* dst = ctx.dst.data();
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 8 <= elements; i += 8) {
        const __m512i v = _mm512_load_si512(&src[i]);
        _mm512_store_si512(&dst[i], v);
    }

    for (; i < elements; ++i) {
        dst[i] = src[i];
    }
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void copy_avx512_nt_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    double* dst = ctx.dst.data();
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 8 <= elements; i += 8) {
        const __m512i v = _mm512_load_si512(&src[i]);
        _mm512_store_si512(&dst[i], v);
    }

    for (; i < elements; ++i) {
        dst[i] = src[i];
    }

    _mm_sfence();
}

static constexpr Kernel copy_avx512 = {
    "copy_avx512",
    KernelKind::copy,
    2 * sizeof(double),
    &copy_avx512_kernel
};

static constexpr Kernel copy_avx512_nt = {
    "copy_avx512_nt",
    KernelKind::copy,
    2 * sizeof(double),
    &copy_avx512_nt_kernel
};

static bool copy_avx512_registered = []() {
    global_registry().add_kernel(copy_avx512);
    global_registry().add_kernel(copy_avx512_nt);
    return true;
}();

} // namespace membench
