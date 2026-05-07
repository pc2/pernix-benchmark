#include <membench/benchmark.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void store_scalar_kernel(BenchmarkContext& ctx, size_t elements) {
    double* dst = ctx.dst.data();

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (size_t i = 0; i < elements; ++i) {
        dst[i] = daxpy_alpha;
    }
}

static constexpr Kernel store_scalar = {
    "store_scalar",
    KernelKind::store,
    sizeof(double),
    &store_scalar_kernel
};

static bool store_scalar_registered = []() {
    global_registry().add_kernel(store_scalar);
    return true;
}();

} // namespace membench
