#include <membench/benchmark.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void daxpy_sse_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    double* dst = ctx.dst.data();
    const __m128d alpha = _mm_set1_pd(daxpy_alpha);

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (size_t i = 0; i < elements; i += 2) {
        const __m128d x = _mm_load_pd(&src[i]);
        const __m128d y = _mm_load_pd(&dst[i]);
        _mm_store_pd(&dst[i], _mm_add_pd(_mm_mul_pd(alpha, x), y));
    }
}

static constexpr Kernel daxpy_sse = {
    "daxpy_sse",
    KernelKind::daxpy,
    3 * sizeof(double),
    &daxpy_sse_kernel,
};

static bool daxpy_sse_registered = []() {
    global_registry().add_kernel(daxpy_sse);
    return true;
}();

} // namespace membench
