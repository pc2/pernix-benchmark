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

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (size_t i = 0; i < elements; i += svcntd()) {
        const svbool_t pg = svwhilelt_b64(i, elements);
        const svfloat64_t x = svld1_f64(pg, &src[i]);
        const svfloat64_t y = svld1_f64(pg, &dst[i]);
        svst1_f64(pg, &dst[i], svadd_f64_x(pg, svmul_f64_x(pg, alpha, x), y));
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
