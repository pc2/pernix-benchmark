#include <membench/benchmark.h>
#include <membench/utils.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void load_sse_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    __m128d acc = _mm_setzero_pd();
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 2 <= elements; i += 2) {
        const __m128d v = _mm_load_pd(&src[i]);
        acc = _mm_add_pd(acc, v);
    }

    alignas(16) double partial[2];
    _mm_store_pd(partial, acc);
    double sum = partial[0] + partial[1];
    for (; i < elements; ++i) {
        sum += src[i];
    }

    do_not_optimize(sum);
    ctx.result = sum;
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void load_sse_nt_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    __m128d acc = _mm_setzero_pd();
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 2 <= elements; i += 2) {
        const __m128d v = _mm_load_pd(&src[i]);
        acc = _mm_add_pd(acc, v);
    }

    alignas(16) double partial[2];
    _mm_store_pd(partial, acc);
    double sum = partial[0] + partial[1];
    for (; i < elements; ++i) {
        sum += src[i];
    }

    do_not_optimize(sum);
    ctx.result = sum;
    _mm_sfence();
}

static constexpr Kernel load_sse = {
    "load_sse",
    KernelKind::load,
    sizeof(double),
    &load_sse_kernel
};

static constexpr Kernel load_sse_nt = {
    "load_sse_nt",
    KernelKind::load,
    sizeof(double),
    &load_sse_nt_kernel
};

static bool load_sse_registered = []() {
    global_registry().add_kernel(load_sse);
    global_registry().add_kernel(load_sse_nt);
    return true;
}();

} // namespace membench
