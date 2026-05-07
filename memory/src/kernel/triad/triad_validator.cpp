#include <membench/benchmark.h>

static bool validate_triad(const membench::BenchmarkContext& ctx) {
    const std::span<const double> src = ctx.src1.span();
    const std::span<const double> src2 = ctx.src2.span();
    const std::span<const double> dst = ctx.dst.span();

    for (size_t i = 0; i < src.size(); ++i) {
        const double expected = src[i] + membench::daxpy_alpha * src2[i];
        if (dst[i] != expected) {
            return false;
        }
    }

    return true;
}

static bool validate_triad_registered = []() {
    membench::global_registry().add_validator(membench::KernelKind::triad, validate_triad);
    return true;
}();
