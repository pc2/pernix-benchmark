#ifndef MEMBENCH_BENCHMARK_H
#define MEMBENCH_BENCHMARK_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "memory.h"

namespace membench {

constexpr double daxpy_alpha = 2.0;

struct BenchmarkConfig {
    size_t bytes;
    size_t iterations;
    size_t warmup_iterations;
    uint64_t seed;
    int32_t cpu_id; // if negative, do not set affinity
};

struct BenchmarkContext {
    explicit BenchmarkContext(const BenchmarkConfig& config)
        : config(config),
          src1(config.bytes / sizeof(double)),
          src2(config.bytes / sizeof(double)),
          dst(config.bytes / sizeof(double)) {
        std::mt19937_64 rng(config.seed);
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        for (double& value : src1.span()) {
            value = distribution(rng);
        }
        for (double& value : src2.span()) {
            value = distribution(rng);
        }
        for (double& value : dst.span()) {
            value = distribution(rng);
        }
    }

    BenchmarkConfig config;
    AlignedBuffer<double> src1;
    AlignedBuffer<double> src2;
    AlignedBuffer<double> dst;
    double result = std::numeric_limits<double>::quiet_NaN();
};

inline void reset_output(BenchmarkContext& ctx) {
    ctx.dst.fill(1.0);
}

using KernelFn = void (*)(BenchmarkContext& ctx, size_t elements);
using KernelValidationFn = bool (*)(const BenchmarkContext& ctx);

enum class KernelKind {
    load,
    store,
    copy,
    ddot,
    daxpy,
    update,
    triad,
    unknown,
};

struct Kernel {
    const std::string name{};
    const KernelKind kind{};
    size_t bytes_per_element{};
    KernelFn kernel{};
};

class BenchmarkRegistry {
public:
    void add_kernel(const Kernel& kernel);
    [[nodiscard]] const std::vector<Kernel>& kernels() const noexcept;

    void add_validator(KernelKind kind, KernelValidationFn validator);
    [[nodiscard]] KernelValidationFn find_validator(KernelKind kind) const noexcept;

    [[nodiscard]] const Kernel* find_kernel(const std::string& name) const noexcept;

    void run_kernel(const std::string& name, BenchmarkContext& ctx) const;

    void run_kernel(const Kernel& kernel, BenchmarkContext& ctx) const;

    void run_all(BenchmarkContext& ctx) const;
private:
    std::vector<Kernel> kernels_;
    std::unordered_map<KernelKind, KernelValidationFn> validators_;
};

// Access the global registry used by the simple benchmark runner.
BenchmarkRegistry& global_registry();

}

#endif  // MEMBENCH_BENCHMARK_H
