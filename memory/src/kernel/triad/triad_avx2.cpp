#include <membench/benchmark.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void triad_avx2_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    const double* src2 = ctx.src2.data();
    double* dst = ctx.dst.data();
    const __m256d alpha = _mm256_set1_pd(daxpy_alpha);
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 4 <= elements; i += 4) {
        const __m256d x = _mm256_load_pd(&src[i]);
        const __m256d y = _mm256_load_pd(&src2[i]);
        _mm256_store_pd(&dst[i], _mm256_add_pd(x, _mm256_mul_pd(alpha, y)));
    }

    for (; i < elements; ++i) {
        dst[i] = src[i] + daxpy_alpha * src2[i];
    }
}

static constexpr Kernel triad_avx2 = {
    "triad_avx2",
    KernelKind::triad,
    3 * sizeof(double),
    &triad_avx2_kernel
};

static bool triad_avx2_registered = []() {
    global_registry().add_kernel(triad_avx2);
    return true;
}();

} // namespace membench
