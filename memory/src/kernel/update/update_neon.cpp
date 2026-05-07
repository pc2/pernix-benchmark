#include <membench/benchmark.h>

#include <arm_neon.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void update_neon_kernel(BenchmarkContext& ctx, size_t elements) {
    double* dst = ctx.dst.data();
    const float64x2_t alpha = vdupq_n_f64(daxpy_alpha);
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 2 <= elements; i += 2) {
        const float64x2_t y = vld1q_f64(&dst[i]);
        vst1q_f64(&dst[i], vmulq_f64(alpha, y));
    }

    for (; i < elements; ++i) {
        dst[i] = daxpy_alpha * dst[i];
    }
}

static constexpr Kernel update_neon = {
    "update_neon",
    KernelKind::update,
    2 * sizeof(double),
    &update_neon_kernel
};

static bool update_neon_registered = []() {
    global_registry().add_kernel(update_neon);
    return true;
}();

} // namespace membench
