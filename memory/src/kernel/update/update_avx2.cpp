#include <membench/benchmark.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void update_avx2_kernel(BenchmarkContext& ctx, size_t elements) {
    double* dst = ctx.dst.data();
    const __m256d alpha = _mm256_set1_pd(daxpy_alpha);
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 4 <= elements; i += 4) {
        const __m256d y = _mm256_load_pd(&dst[i]);
        _mm256_store_pd(&dst[i], _mm256_mul_pd(alpha, y));
    }

    for (; i < elements; ++i) {
        dst[i] = daxpy_alpha * dst[i];
    }
}

static constexpr Kernel update_avx2 = {
    "update_avx2",
    KernelKind::update,
    2 * sizeof(double),
    &update_avx2_kernel
};

static bool update_avx2_registered = []() {
    global_registry().add_kernel(update_avx2);
    return true;
}();

} // namespace membench
