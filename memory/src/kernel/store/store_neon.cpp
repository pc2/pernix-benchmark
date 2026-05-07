#include <membench/benchmark.h>

#include <arm_neon.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void store_neon_kernel(BenchmarkContext& ctx, size_t elements) {
    double* dst = ctx.dst.data();
    const float64x2_t value = vdupq_n_f64(daxpy_alpha);
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 2 <= elements; i += 2) {
        vst1q_f64(&dst[i], value);
    }

    for (; i < elements; ++i) {
        dst[i] = daxpy_alpha;
    }
}

static constexpr Kernel store_neon = {
    "store_neon",
    KernelKind::store,
    sizeof(double),
    &store_neon_kernel
};

static bool store_neon_registered = []() {
    global_registry().add_kernel(store_neon);
    return true;
}();

} // namespace membench
