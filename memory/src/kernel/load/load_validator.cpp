#include <membench/benchmark.h>

#include <algorithm>
#include <cmath>

static bool nearly_equal(const double actual, const double expected) {
    constexpr double relative_tolerance = 1e-12;
    const double scale = std::max(1.0, std::abs(expected));
    return std::abs(actual - expected) <= relative_tolerance * scale;
}

static bool validate_load(const membench::BenchmarkContext& ctx) {
    const std::span<const double> src = ctx.src1.span();

    double sum = 0;
    for (const double value : src) {
        sum += value;
    }

    return nearly_equal(ctx.result, sum);
}

static bool validate_load_registered = []() {
    membench::global_registry().add_validator(membench::KernelKind::load, validate_load);
    return true;
}();
