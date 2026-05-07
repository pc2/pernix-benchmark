#include <membench/benchmark.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void triad_sse_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    const double* src2 = ctx.src2.data();
    double* dst = ctx.dst.data();
    const __m128d alpha = _mm_set1_pd(daxpy_alpha);
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 2 <= elements; i += 2) {
        const __m128d x = _mm_load_pd(&src[i]);
        const __m128d y = _mm_load_pd(&src2[i]);
        _mm_store_pd(&dst[i], _mm_add_pd(x, _mm_mul_pd(alpha, y)));
    }

    for (; i < elements; ++i) {
        dst[i] = src[i] + daxpy_alpha * src2[i];
    }
}

static constexpr Kernel triad_sse = {
    "triad_sse",
    KernelKind::triad,
    3 * sizeof(double),
    &triad_sse_kernel
};

static bool triad_sse_registered = []() {
    global_registry().add_kernel(triad_sse);
    return true;
}();

} // namespace membench
