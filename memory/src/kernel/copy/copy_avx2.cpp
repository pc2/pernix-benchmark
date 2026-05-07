#include <membench/benchmark.h>

#include <immintrin.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void copy_avx2_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    double* dst = ctx.dst.data();
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 4 <= elements; i += 4) {
        const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(&src[i]));
        _mm256_store_si256(reinterpret_cast<__m256i*>(&dst[i]), v);
    }

    for (; i < elements; ++i) {
        dst[i] = src[i];
    }
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void copy_avx2_nt_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    double* dst = ctx.dst.data();
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 4 <= elements; i += 4) {
        const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(&src[i]));
        _mm256_store_si256(reinterpret_cast<__m256i*>(&dst[i]), v);
    }

    for (; i < elements; ++i) {
        dst[i] = src[i];
    }

    _mm_sfence();
}

static constexpr Kernel copy_avx2 = {
    "copy_avx2",
    KernelKind::copy,
    2 * sizeof(double),
    &copy_avx2_kernel
};

static constexpr Kernel copy_avx2_nt = {
    "copy_avx2_nt",
    KernelKind::copy,
    2 * sizeof(double),
    &copy_avx2_nt_kernel
};

static bool copy_avx2_registered = []() {
    global_registry().add_kernel(copy_avx2);
    global_registry().add_kernel(copy_avx2_nt);
    return true;
}();

} // namespace membench
