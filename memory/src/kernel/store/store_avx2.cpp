#include <membench/benchmark.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void store_avx2_kernel(BenchmarkContext& ctx, size_t elements) {
    double* dst = ctx.dst.data();
    const __m256d value = _mm256_set1_pd(daxpy_alpha);
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 4 <= elements; i += 4) {
        _mm256_store_pd(&dst[i], value);
    }

    for (; i < elements; ++i) {
        dst[i] = daxpy_alpha;
    }
}

static constexpr Kernel store_avx2 = {
    "store_avx2",
    KernelKind::store,
    sizeof(double),
    &store_avx2_kernel
};

static bool store_avx2_registered = []() {
    global_registry().add_kernel(store_avx2);
    return true;
}();

} // namespace membench
