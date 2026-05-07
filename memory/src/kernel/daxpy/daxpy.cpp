#include <membench/benchmark.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void daxpy_scalar_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    double* dst = ctx.dst.data();

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (size_t i = 0; i < elements; ++i) {
        dst[i] = daxpy_alpha * src[i] + dst[i];
    }
}

static constexpr Kernel daxpy_scalar = {
    "daxpy_scalar",
    KernelKind::daxpy,
    3 * sizeof(double),
    &daxpy_scalar_kernel,
};

static bool daxpy_scalar_registered = []() {
    global_registry().add_kernel(daxpy_scalar);
    return true;
}();

} // namespace membench
