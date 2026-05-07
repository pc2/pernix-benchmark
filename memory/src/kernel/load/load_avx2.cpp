#include <membench/benchmark.h>
#include <membench/utils.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void load_avx2_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    __m256d acc = _mm256_setzero_pd();
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 4 <= elements; i += 4) {
        const __m256d v = _mm256_load_pd(&src[i]);
        acc = _mm256_add_pd(acc, v);
    }

    alignas(32) double partial[4];
    _mm256_store_pd(partial, acc);
    double sum = partial[0] + partial[1] + partial[2] + partial[3];
    for (; i < elements; ++i) {
        sum += src[i];
    }

    do_not_optimize(sum);
    ctx.result = sum;
}

static constexpr Kernel load_avx2 = {
    "load_avx2",
    KernelKind::load,
    sizeof(double),
    &load_avx2_kernel
};

static bool load_avx2_registered = []() {
    global_registry().add_kernel(load_avx2);
    return true;
}();

} // namespace membench
