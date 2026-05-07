#include <membench/benchmark.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void load_scalar_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    double sum = 0.0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (size_t i = 0; i < elements; ++i) {
        sum += src[i];
    }

    ctx.result = sum;
}

static constexpr Kernel load_scalar = {
    "load_scalar",
    KernelKind::load,
    sizeof(double),
    &load_scalar_kernel
};

static bool load_scalar_registered = []() {
    global_registry().add_kernel(load_scalar);
    return true;
}();

} // namespace membench
