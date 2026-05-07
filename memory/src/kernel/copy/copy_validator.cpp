#include <membench/benchmark.h>

static bool validate_copy(const membench::BenchmarkContext& ctx) {
    const std::span<const double> src = ctx.src1.span();
    const std::span<const double> dst = ctx.dst.span();

    for (size_t i = 0; i < src.size(); ++i) {
        if (dst[i] != src[i]) {
            return false;
        }
    }

    return true;
}

static bool validate_copy_registered = []() {
    membench::global_registry().add_validator(membench::KernelKind::copy, validate_copy);
    return true;
}();
