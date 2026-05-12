#include <membench/benchmark.h>
#include <membench/utils.h>

#include <arm_sve.h>

namespace membench {

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void load_sve_kernel(BenchmarkContext& ctx, size_t elements) {
    const double* src = ctx.src1.data();
    svfloat64_t acc = svdup_f64(0.0);
    const size_t vl = svcntd();
    const svbool_t pg_full = svptrue_b64();

    size_t i = 0;

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + vl <= elements; i += vl) {
        const svfloat64_t v = svld1_f64(pg_full, &src[i]);
        acc = svadd_f64_x(pg_full, acc, v);
    }

    if (i < elements) {
        const svbool_t pg = svwhilelt_b64(i, elements);
        const svfloat64_t v = svld1_f64(pg, &src[i]);
        acc = svadd_f64_m(pg, acc, v);
    }

    const double sum = svaddv_f64(pg_full, acc);
    do_not_optimize(sum);
    ctx.result = sum;
}

static constexpr Kernel load_sve = {
    "load_sve",
    KernelKind::load,
    sizeof(double),
    &load_sve_kernel
};

static bool load_sve_registered = []() {
    global_registry().add_kernel(load_sve);
    return true;
}();

} // namespace membench
