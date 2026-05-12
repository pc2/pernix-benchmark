#include <membench/benchmark.h>

#include <arm_sve.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void update_sve_kernel(BenchmarkContext& ctx, size_t elements) {
    double* dst = ctx.dst.data();
    const svfloat64_t alpha = svdup_f64(daxpy_alpha);
    const size_t vl = svcntd();
    const svbool_t pg_full = svptrue_b64();

    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + vl <= elements; i += vl) {
        const svfloat64_t y = svld1_f64(pg_full, &dst[i]);
        svst1_f64(pg_full, &dst[i], svmul_f64_x(pg_full, alpha, y));
    }

    if (i < elements) {
        const svbool_t pg = svwhilelt_b64(i, elements);
        const svfloat64_t y = svld1_f64(pg, &dst[i]);
        svst1_f64(pg, &dst[i], svmul_f64_x(pg, alpha, y));
    }
}

static constexpr Kernel update_sve = {
    "update_sve",
    KernelKind::update,
    2 * sizeof(double),
    &update_sve_kernel
};

static bool update_sve_registered = []() {
    global_registry().add_kernel(update_sve);
    return true;
}();

} // namespace membench
