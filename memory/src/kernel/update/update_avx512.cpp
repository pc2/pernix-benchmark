#include <membench/benchmark.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void update_avx512_kernel(BenchmarkContext& ctx, size_t elements) {
    double* dst = ctx.dst.data();
    const __m512d alpha = _mm512_set1_pd(daxpy_alpha);
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 8 <= elements; i += 8) {
        const __m512d y = _mm512_load_pd(&dst[i]);
        _mm512_store_pd(&dst[i], _mm512_mul_pd(alpha, y));
    }

    for (; i < elements; ++i) {
        dst[i] = daxpy_alpha * dst[i];
    }
}

static constexpr Kernel update_avx512 = {
    "update_avx512",
    KernelKind::update,
    2 * sizeof(double),
    &update_avx512_kernel
};

static bool update_avx512_registered = []() {
    global_registry().add_kernel(update_avx512);
    return true;
}();

} // namespace membench
