#include <membench/benchmark.h>
#include <membench/utils.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void ddot_sse_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    const double* src2 = ctx.src2.data();
    __m128d acc = _mm_setzero_pd();
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 2 <= elements; i += 2) {
        const __m128d x = _mm_load_pd(&src[i]);
        const __m128d y = _mm_load_pd(&src2[i]);
        acc = _mm_add_pd(acc, _mm_mul_pd(x, y));
    }

    alignas(16) double lanes[2];
    _mm_store_pd(lanes, acc);
    double sum = lanes[0] + lanes[1];

    for (; i < elements; ++i) {
        sum += src[i] * src2[i];
    }

    do_not_optimize(sum);
    ctx.result = sum;
}

static constexpr Kernel ddot_sse = {
    "ddot_sse",
    KernelKind::ddot,
    2 * sizeof(double),
    &ddot_sse_kernel
};

static bool ddot_sse_registered = []() {
    global_registry().add_kernel(ddot_sse);
    return true;
}();

} // namespace membench
