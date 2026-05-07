#include <membench/benchmark.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void triad_scalar_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    const double* src2 = ctx.src2.data();
    double* dst = ctx.dst.data();

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (size_t i = 0; i < elements; ++i) {
        dst[i] = src[i] + daxpy_alpha * src2[i];
    }
}

static constexpr Kernel triad_scalar = {
    "triad_scalar",
    KernelKind::triad,
    3 * sizeof(double),
    &triad_scalar_kernel
};

static bool triad_scalar_registered = []() {
    global_registry().add_kernel(triad_scalar);
    return true;
}();

} // namespace membench
