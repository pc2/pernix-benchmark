#include <membench/benchmark.h>

#include <arm_neon.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void daxpy_neon_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    double* dst = ctx.dst.data();
    const float64x2_t alpha = vdupq_n_f64(daxpy_alpha);
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 2 <= elements; i += 2) {
        const float64x2_t x = vld1q_f64(&src[i]);
        const float64x2_t y = vld1q_f64(&dst[i]);
        vst1q_f64(&dst[i], vfmaq_f64(y, alpha, x));
    }

    for (; i < elements; ++i) {
        dst[i] = daxpy_alpha * src[i] + dst[i];
    }
}

static constexpr Kernel daxpy_neon = {
    "daxpy_neon",
    KernelKind::daxpy,
    3 * sizeof(double),
    &daxpy_neon_kernel,
};

static bool daxpy_neon_registered = []() {
    global_registry().add_kernel(daxpy_neon);
    return true;
}();

} // namespace membench
