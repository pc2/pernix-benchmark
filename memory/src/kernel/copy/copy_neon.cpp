#include <membench/benchmark.h>

#include <arm_neon.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void copy_neon_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    double* dst = ctx.dst.data();

    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 2 <= elements; i += 2) {
        const float64x2_t v = vld1q_f64(&src[i]);
        vst1q_f64(&dst[i], v);
    }

    for (; i < elements; ++i) {
        dst[i] = src[i];
    }
}

static constexpr Kernel copy_neon = {
    "copy_neon",
    KernelKind::copy,
    2 * sizeof(double),
    &copy_neon_kernel
};

static bool copy_neon_registered = []() {
    global_registry().add_kernel(copy_neon);
    return true;
}();

} // namespace membench
