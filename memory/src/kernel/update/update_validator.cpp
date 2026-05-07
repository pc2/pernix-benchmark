#include <membench/benchmark.h>

#include <cmath>

static bool validate_update(const membench::BenchmarkContext& ctx) {
    const std::span<const double> dst = ctx.dst.span();
    const double expected = std::pow(membench::daxpy_alpha, static_cast<double>(ctx.config.iterations));

    for (const double value : dst) {
        if (value != expected) {
            return false;
        }
    }

    return true;
}

static bool validate_update_registered = []() {
    membench::global_registry().add_validator(membench::KernelKind::update, validate_update);
    return true;
}();
