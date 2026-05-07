#include <membench/benchmark.h>
#include <membench/utils.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void ddot_avx512_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    const double* src2 = ctx.src2.data();
    __m512d acc = _mm512_setzero_pd();
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 8 <= elements; i += 8) {
        const __m512d x = _mm512_load_pd(&src[i]);
        const __m512d y = _mm512_load_pd(&src2[i]);
        acc = _mm512_add_pd(acc, _mm512_mul_pd(x, y));
    }

    alignas(64) double lanes[8];
    _mm512_store_pd(lanes, acc);
    double sum = lanes[0] + lanes[1] + lanes[2] + lanes[3]
        + lanes[4] + lanes[5] + lanes[6] + lanes[7];

    for (; i < elements; ++i) {
        sum += src[i] * src2[i];
    }

    do_not_optimize(sum);
    ctx.result = sum;
}

static constexpr Kernel ddot_avx512 = {
    "ddot_avx512",
    KernelKind::ddot,
    2 * sizeof(double),
    &ddot_avx512_kernel
};

static bool ddot_avx512_registered = []() {
    global_registry().add_kernel(ddot_avx512);
    return true;
}();

} // namespace membench
