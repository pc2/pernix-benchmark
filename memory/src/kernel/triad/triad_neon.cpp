#include <membench/benchmark.h>

#include <arm_neon.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void triad_neon_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    const double* src2 = ctx.src2.data();
    double* dst = ctx.dst.data();
    const float64x2_t alpha = vdupq_n_f64(daxpy_alpha);
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 2 <= elements; i += 2) {
        const float64x2_t x = vld1q_f64(&src[i]);
        const float64x2_t y = vld1q_f64(&src2[i]);
        vst1q_f64(&dst[i], vaddq_f64(x, vmulq_f64(alpha, y)));
    }

    for (; i < elements; ++i) {
        dst[i] = src[i] + daxpy_alpha * src2[i];
    }
}

static constexpr Kernel triad_neon = {
    "triad_neon",
    KernelKind::triad,
    3 * sizeof(double),
    &triad_neon_kernel
};

static bool triad_neon_registered = []() {
    global_registry().add_kernel(triad_neon);
    return true;
}();

} // namespace membench
