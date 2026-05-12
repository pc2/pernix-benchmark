#include <membench/benchmark.h>

#include <arm_sve.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void copy_sve_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    double* dst = ctx.dst.data();
    const size_t vl = svcntd();
    const svbool_t pg_full = svptrue_b64();

    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + vl <= elements; i += vl) {
        const svfloat64_t v = svld1_f64(pg_full, &src[i]);
        svst1_f64(pg_full, &dst[i], v);
    }

    if (i < elements) {
        const svbool_t pg = svwhilelt_b64(i, elements);
        const svfloat64_t v = svld1_f64(pg, &src[i]);
        svst1_f64(pg, &dst[i], v);
    }
}

static constexpr Kernel copy_sve = {
    "copy_sve",
    KernelKind::copy,
    2 * sizeof(double),
    &copy_sve_kernel
};

static bool copy_sve_registered = []() {
    global_registry().add_kernel(copy_sve);
    return true;
}();

} // namespace membench
