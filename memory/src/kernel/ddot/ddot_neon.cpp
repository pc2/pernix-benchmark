#include <membench/benchmark.h>
#include <membench/utils.h>

#include <arm_neon.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void ddot_neon_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    const double* src2 = ctx.src2.data();
    float64x2_t acc = vdupq_n_f64(0.0);
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 2 <= elements; i += 2) {
        const float64x2_t x = vld1q_f64(&src[i]);
        const float64x2_t y = vld1q_f64(&src2[i]);
        acc = vaddq_f64(acc, vmulq_f64(x, y));
    }

    double sum = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);

    for (; i < elements; ++i) {
        sum += src[i] * src2[i];
    }

    do_not_optimize(sum);
    ctx.result = sum;
}

static constexpr Kernel ddot_neon = {
    "ddot_neon",
    KernelKind::ddot,
    2 * sizeof(double),
    &ddot_neon_kernel
};

static bool ddot_neon_registered = []() {
    global_registry().add_kernel(ddot_neon);
    return true;
}();

} // namespace membench
