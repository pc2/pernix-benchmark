#include <membench/benchmark.h>
#include <membench/utils.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void ddot_avx2_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    const double* src2 = ctx.src2.data();
    __m256d acc = _mm256_setzero_pd();
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 4 <= elements; i += 4) {
        const __m256d x = _mm256_load_pd(&src[i]);
        const __m256d y = _mm256_load_pd(&src2[i]);
        acc = _mm256_add_pd(acc, _mm256_mul_pd(x, y));
    }

    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, acc);
    double sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];

    for (; i < elements; ++i) {
        sum += src[i] * src2[i];
    }

    do_not_optimize(sum);
    ctx.result = sum;
}

static constexpr Kernel ddot_avx2 = {
    "ddot_avx2",
    KernelKind::ddot,
    2 * sizeof(double),
    &ddot_avx2_kernel
};

static bool ddot_avx2_registered = []() {
    global_registry().add_kernel(ddot_avx2);
    return true;
}();

} // namespace membench
