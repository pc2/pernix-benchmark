#include <membench/benchmark.h>
#include <membench/utils.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void load_avx512_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    __m512d acc = _mm512_setzero_pd();
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 8 <= elements; i += 8) {
        const __m512d v = _mm512_load_pd(&src[i]);
        acc = _mm512_add_pd(acc, v);
    }

    double sum = _mm512_reduce_add_pd(acc);
    for (; i < elements; ++i) {
        sum += src[i];
    }

    do_not_optimize(sum);
    ctx.result = sum;
}

static constexpr Kernel load_avx512 = {
    "load_avx512",
    KernelKind::load,
    sizeof(double),
    &load_avx512_kernel
};

static bool load_avx512_registered = []() {
    global_registry().add_kernel(load_avx512);
    return true;
}();

} // namespace membench
