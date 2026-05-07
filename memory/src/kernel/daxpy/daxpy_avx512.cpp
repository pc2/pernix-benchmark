#include <membench/benchmark.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void daxpy_avx512_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    double* dst = ctx.dst.data();
    const __m512d alpha = _mm512_set1_pd(daxpy_alpha);

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (size_t i = 0; i < elements; i += 8) {
        const __m512d x = _mm512_load_pd(&src[i]);
        const __m512d y = _mm512_load_pd(&dst[i]);
        _mm512_store_pd(&dst[i], _mm512_add_pd(_mm512_mul_pd(alpha, x), y));
    }
}

static constexpr Kernel daxpy_avx512 = {
    "daxpy_avx512",
    KernelKind::daxpy,
    3 * sizeof(double),
    &daxpy_avx512_kernel,
};

static bool daxpy_avx512_registered = []() {
    global_registry().add_kernel(daxpy_avx512);
    return true;
}();

} // namespace membench
