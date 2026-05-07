#include <membench/benchmark.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void triad_avx512_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    const double* src2 = ctx.src2.data();
    double* dst = ctx.dst.data();
    const __m512d alpha = _mm512_set1_pd(daxpy_alpha);
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 8 <= elements; i += 8) {
        const __m512d x = _mm512_load_pd(&src[i]);
        const __m512d y = _mm512_load_pd(&src2[i]);
        _mm512_store_pd(&dst[i], _mm512_add_pd(x, _mm512_mul_pd(alpha, y)));
    }

    for (; i < elements; ++i) {
        dst[i] = src[i] + daxpy_alpha * src2[i];
    }
}

static constexpr Kernel triad_avx512 = {
    "triad_avx512",
    KernelKind::triad,
    3 * sizeof(double),
    &triad_avx512_kernel
};

static bool triad_avx512_registered = []() {
    global_registry().add_kernel(triad_avx512);
    return true;
}();

} // namespace membench
