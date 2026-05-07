#include <membench/benchmark.h>
#include <membench/utils.h>

#include <arm_neon.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void load_neon_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    float64x2_t acc = vdupq_n_f64(0.0);
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 2 <= elements; i += 2) {
        const float64x2_t v = vld1q_f64(&src[i]);
        acc = vaddq_f64(acc, v);
    }

    double sum = vaddvq_f64(acc);
    for (; i < elements; ++i) {
        sum += src[i];
    }

    do_not_optimize(sum);
    ctx.result = sum;
}

static constexpr Kernel load_neon = {
    "load_neon",
    KernelKind::load,
    sizeof(double),
    &load_neon_kernel
};

static bool load_neon_registered = []() {
    global_registry().add_kernel(load_neon);
    return true;
}();

} // namespace membench
