#include <membench/benchmark.h>

#include <arm_sve.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void daxpy_sve_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    double* dst = ctx.dst.data();

    const svfloat64_t alpha = svdup_f64(daxpy_alpha);
    const svbool_t pg_full = svptrue_b64();

    const size_t vl = svcntd();
    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + vl <= elements; i += vl) {
        const svfloat64_t x = svld1_f64(pg_full, &src[i]);
        const svfloat64_t y = svld1_f64(pg_full, &dst[i]);

        const svfloat64_t r = svadd_f64_x(pg_full, svmul_f64_x(pg_full, alpha, x), y);

        svst1_f64(pg_full, &dst[i], r);
    }

    if (i < elements) {
        const svbool_t pg = svwhilelt_b64(i, elements);
        const svfloat64_t x = svld1_f64(pg, &src[i]);
        const svfloat64_t y = svld1_f64(pg, &dst[i]);

        const svfloat64_t r = svadd_f64_x(pg, svmul_f64_x(pg, alpha, x), y);

        svst1_f64(pg, &dst[i], r);
    }
}

static constexpr Kernel daxpy_sve = {
    "daxpy_sve",
    KernelKind::daxpy,
    3 * sizeof(double),
    &daxpy_sve_kernel,
};

static bool daxpy_sve_registered = []() {
    global_registry().add_kernel(daxpy_sve);
    return true;
}();

} // namespace membench
