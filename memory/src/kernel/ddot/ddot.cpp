#include <membench/benchmark.h>
#include <membench/utils.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void ddot_scalar_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    const double* src2 = ctx.src2.data();
    double sum = 0.0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (size_t i = 0; i < elements; ++i) {
        sum += src[i] * src2[i];
    }

    do_not_optimize(sum);
    ctx.result = sum;
}

static constexpr Kernel ddot_scalar = {
    "ddot_scalar",
    KernelKind::ddot,
    2 * sizeof(double),
    &ddot_scalar_kernel
};

static bool ddot_scalar_registered = []() {
    global_registry().add_kernel(ddot_scalar);
    return true;
}();

} // namespace membench
