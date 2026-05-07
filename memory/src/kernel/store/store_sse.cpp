#include <membench/benchmark.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void store_sse_kernel(BenchmarkContext& ctx, size_t elements) {
    double* dst = ctx.dst.data();
    const __m128d value = _mm_set1_pd(daxpy_alpha);
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 2 <= elements; i += 2) {
        _mm_store_pd(&dst[i], value);
    }

    for (; i < elements; ++i) {
        dst[i] = daxpy_alpha;
    }
}

static constexpr Kernel store_sse = {
    "store_sse",
    KernelKind::store,
    sizeof(double),
    &store_sse_kernel
};

static bool store_sse_registered = []() {
    global_registry().add_kernel(store_sse);
    return true;
}();

} // namespace membench
