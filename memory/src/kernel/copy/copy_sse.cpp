#include <membench/benchmark.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void copy_sse_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    double* dst = ctx.dst.data();
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 2 <= elements; i += 2) {
        const __m128i v = _mm_load_si128(reinterpret_cast<const __m128i*>(&src[i]));
        _mm_store_si128(reinterpret_cast<__m128i*>(&dst[i]), v);
    }

    for (; i < elements; ++i) {
        dst[i] = src[i];
    }
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void copy_sse_nt_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    double* dst = ctx.dst.data();
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 2 <= elements; i += 2) {
        const __m128d v = _mm_load_pd(&src[i]);
        _mm_stream_pd(&dst[i], v);
    }

    for (; i < elements; ++i) {
        dst[i] = src[i];
    }

    _mm_sfence();
}

static constexpr Kernel copy_sse = {
    "copy_sse",
    KernelKind::copy,
    2 * sizeof(double),
    &copy_sse_kernel
};

static constexpr Kernel copy_sse_nt = {
    "copy_sse_nt",
    KernelKind::copy,
    2 * sizeof(double),
    &copy_sse_nt_kernel
};

static bool copy_sse_registered = []() {
    global_registry().add_kernel(copy_sse);
    global_registry().add_kernel(copy_sse_nt);
    return true;
}();

} // namespace membench
