#ifndef MEMBENCH_BENCHMARK_H
#define MEMBENCH_BENCHMARK_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "memory.h"

namespace membench {

constexpr double daxpy_alpha = 2.0;

struct BenchmarkConfig {
    size_t bytes;
    size_t iterations; // zero means auto-calibrate per kernel
    size_t repetitions;
    size_t warmup_iterations;
    double target_seconds_per_repetition;
    uint64_t seed;
    int32_t cpu_id; // if negative, do not set affinity
    bool randomize_kernel_order;
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

struct BenchmarkStats {
    double min{};
    double median{};
    double mean{};
    double stdev{};
    double p95{};
};

struct BenchmarkResult {
    std::string kernel;
    size_t elements{};
    size_t bytes_per_element{};
    size_t iterations{};
    std::vector<double> time_seconds_samples;
    std::vector<double> time_per_iteration_seconds_samples;
    std::vector<double> bandwidth_bytes_per_second_samples;
    BenchmarkStats time_seconds;
    BenchmarkStats time_per_iteration_seconds;
    BenchmarkStats bandwidth_bytes_per_second;
    std::string bandwidth_unit;
    double bandwidth_unit_scale{};
    bool validation_passed{};
};

class BenchmarkRegistry {
public:
    void add_kernel(const Kernel& kernel);
    [[nodiscard]] const std::vector<Kernel>& kernels() const noexcept;

    void add_validator(KernelKind kind, KernelValidationFn validator);
    [[nodiscard]] KernelValidationFn find_validator(KernelKind kind) const noexcept;

    [[nodiscard]] const Kernel* find_kernel(const std::string& name) const noexcept;

    [[nodiscard]] std::optional<BenchmarkResult> run_kernel(const std::string& name, BenchmarkContext& ctx) const;

    [[nodiscard]] std::optional<BenchmarkResult> run_kernel(const Kernel& kernel, BenchmarkContext& ctx) const;

    [[nodiscard]] std::vector<BenchmarkResult> run_all(BenchmarkContext& ctx) const;
private:
    std::vector<Kernel> kernels_;
    std::unordered_map<KernelKind, KernelValidationFn> validators_;
};

// Access the global registry used by the simple benchmark runner.
BenchmarkRegistry& global_registry();

}

#endif  // MEMBENCH_BENCHMARK_H
