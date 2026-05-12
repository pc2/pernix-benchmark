#include <membench/benchmark.h>

#include <arm_sve.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void triad_sve_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    const double* src2 = ctx.src2.data();
    double* dst = ctx.dst.data();
    const svfloat64_t alpha = svdup_f64(daxpy_alpha);
    const size_t vl = svcntd();
    const svbool_t pg_full = svptrue_b64();

    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + vl <= elements; i += vl) {
        const svfloat64_t x = svld1_f64(pg_full, &src[i]);
        const svfloat64_t y = svld1_f64(pg_full, &src2[i]);
        svst1_f64(pg_full, &dst[i], svadd_f64_x(pg_full, x, svmul_f64_x(pg_full, alpha, y)));
    }

    if (i < elements) {
        const svbool_t pg = svwhilelt_b64(i, elements);
        const svfloat64_t x = svld1_f64(pg, &src[i]);
        const svfloat64_t y = svld1_f64(pg, &src2[i]);
        svst1_f64(pg, &dst[i], svadd_f64_x(pg, x, svmul_f64_x(pg, alpha, y)));
    }
}

static constexpr Kernel triad_sve = {
    "triad_sve",
    KernelKind::triad,
    3 * sizeof(double),
    &triad_sve_kernel
};

static bool triad_sve_registered = []() {
    global_registry().add_kernel(triad_sve);
    return true;
}();

} // namespace membench
