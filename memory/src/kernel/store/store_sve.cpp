#include <membench/benchmark.h>

#include <arm_sve.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void store_sve_kernel(BenchmarkContext& ctx, size_t elements) {
    double* dst = ctx.dst.data();
    const svfloat64_t value = svdup_f64(daxpy_alpha);
    const size_t vl = svcntd();
    const svbool_t pg_full = svptrue_b64();

    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + vl <= elements; i += vl) {
        svst1_f64(pg_full, &dst[i], value);
    }

    if (i < elements) {
        const svbool_t pg = svwhilelt_b64(i, elements);
        svst1_f64(pg, &dst[i], value);
    }
}

static constexpr Kernel store_sve = {
    "store_sve",
    KernelKind::store,
    sizeof(double),
    &store_sve_kernel
};

static bool store_sve_registered = []() {
    global_registry().add_kernel(store_sve);
    return true;
}();

} // namespace membench
