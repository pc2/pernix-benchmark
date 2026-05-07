#include <membench/benchmark.h>

static bool validate_store(const membench::BenchmarkContext& ctx) {
    const std::span<const double> dst = ctx.dst.span();

    for (const double value : dst) {
        if (value != membench::daxpy_alpha) {
            return false;
        }
    }

    return true;
}

static bool validate_store_registered = []() {
    membench::global_registry().add_validator(membench::KernelKind::store, validate_store);
    return true;
}();
