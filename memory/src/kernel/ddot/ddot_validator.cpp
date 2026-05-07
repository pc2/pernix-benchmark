#include <membench/benchmark.h>

#include <algorithm>
#include <cmath>

static bool nearly_equal(const double actual, const double expected) {
    constexpr double relative_tolerance = 1e-12;
    const double scale = std::max(1.0, std::abs(expected));
    return std::abs(actual - expected) <= relative_tolerance * scale;
}

static bool validate_ddot(const membench::BenchmarkContext& ctx) {
    const std::span<const double> src = ctx.src1.span();
    const std::span<const double> src2 = ctx.src2.span();

    double sum = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        sum += src[i] * src2[i];
    }

    return nearly_equal(ctx.result, sum);
}

static bool validate_ddot_registered = []() {
    membench::global_registry().add_validator(membench::KernelKind::ddot, validate_ddot);
    return true;
}();
