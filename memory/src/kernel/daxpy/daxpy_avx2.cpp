#include <membench/benchmark.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void daxpy_avx2_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    double* dst = ctx.dst.data();
    const __m256d alpha = _mm256_set1_pd(daxpy_alpha);

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (size_t i = 0; i < elements; i += 4) {
        const __m256d x = _mm256_load_pd(&src[i]);
        const __m256d y = _mm256_load_pd(&dst[i]);
        _mm256_store_pd(&dst[i], _mm256_add_pd(_mm256_mul_pd(alpha, x), y));
    }
}

static constexpr Kernel daxpy_avx2 = {
    "daxpy_avx2",
    KernelKind::daxpy,
    3 * sizeof(double),
    &daxpy_avx2_kernel,
};

static bool daxpy_avx2_registered = []() {
    global_registry().add_kernel(daxpy_avx2);
    return true;
}();

} // namespace membench
