#include <membench/benchmark.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void copy_scalar_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    double* dst = ctx.dst.data();

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (size_t i = 0; i < elements; ++i) {
        dst[i] = src[i];
    }
}

static constexpr Kernel copy_scalar = {
    "copy_scalar",
    KernelKind::copy,
    2 * sizeof(double),
    &copy_scalar_kernel
};

static bool copy_scalar_registered = []() {
    global_registry().add_kernel(copy_scalar);
    return true;
}();

} // namespace membench
