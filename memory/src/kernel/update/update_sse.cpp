#include <membench/benchmark.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void update_sse_kernel(BenchmarkContext& ctx, size_t elements) {
    double* dst = ctx.dst.data();
    const __m128d alpha = _mm_set1_pd(daxpy_alpha);
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 2 <= elements; i += 2) {
        const __m128d y = _mm_load_pd(&dst[i]);
        _mm_store_pd(&dst[i], _mm_mul_pd(alpha, y));
    }

    for (; i < elements; ++i) {
        dst[i] = daxpy_alpha * dst[i];
    }
}

static constexpr Kernel update_sse = {
    "update_sse",
    KernelKind::update,
    2 * sizeof(double),
    &update_sse_kernel
};

static bool update_sse_registered = []() {
    global_registry().add_kernel(update_sse);
    return true;
}();

} // namespace membench
