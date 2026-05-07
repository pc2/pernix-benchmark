#include <membench/benchmark.h>

#include <algorithm>
#include <cmath>

static bool nearly_equal(const double actual, const double expected) {
    constexpr double relative_tolerance = 1e-9;
    const double scale = std::max(1.0, std::abs(expected));
    return std::abs(actual - expected) <= relative_tolerance * scale;
}

static bool validate_daxpy(const membench::BenchmarkContext& ctx) {
    const std::span<const double> src = ctx.src1.span();
    const std::span<const double> dst = ctx.dst.span();
    const double iterations = static_cast<double>(ctx.config.iterations);

    for (size_t i = 0; i < src.size(); ++i) {
        const double expected = 1.0 + iterations * membench::daxpy_alpha * src[i];
        if (!nearly_equal(dst[i], expected)) {
            return false;
        }
    }

    return true;
}

static bool validate_daxpy_registered = []() {
    membench::global_registry().add_validator(membench::KernelKind::daxpy, validate_daxpy);
    return true;
}();
