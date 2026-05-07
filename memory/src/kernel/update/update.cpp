#include <membench/benchmark.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void update_scalar_kernel(BenchmarkContext& ctx, size_t elements) {
    double* dst = ctx.dst.data();

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (size_t i = 0; i < elements; ++i) {
        dst[i] = daxpy_alpha * dst[i];
    }
}

static constexpr Kernel update_scalar = {
    "update_scalar",
    KernelKind::update,
    2 * sizeof(double),
    &update_scalar_kernel
};

static bool update_scalar_registered = []() {
    global_registry().add_kernel(update_scalar);
    return true;
}();

} // namespace membench
