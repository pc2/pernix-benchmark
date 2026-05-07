
#include <membench/benchmark.h>
#include <membench/cpu.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string_view>
#include <utility>

namespace membench {
namespace {

struct ScaledUnit {
    double scale{};
    std::string_view unit{};
};

ScaledUnit choose_bandwidth_unit(const double bytes_per_second) {
    constexpr std::array<std::string_view, 5> units = {
        "B/s",
        "KiB/s",
        "MiB/s",
        "GiB/s",
        "TiB/s",
    };

    double scale = 1.0;
    size_t unit_index = 0;
    while (unit_index + 1 < units.size() && bytes_per_second / (scale * 1024.0) >= 1.0) {
        scale *= 1024.0;
        ++unit_index;
    }

    return {scale, units[unit_index]};
}

BenchmarkStats compute_stats(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());

    const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    const double mean = sum / static_cast<double>(samples.size());

    double squared_diff_sum = 0.0;
    for (const double sample : samples) {
        const double diff = sample - mean;
        squared_diff_sum += diff * diff;
    }
    const double variance = samples.size() > 1
        ? squared_diff_sum / static_cast<double>(samples.size() - 1)
        : 0.0;

    const size_t middle = samples.size() / 2;
    const double median = samples.size() % 2 == 0
        ? (samples[middle - 1] + samples[middle]) / 2.0
        : samples[middle];

    const size_t p95_index = static_cast<size_t>(
        std::ceil(0.95 * static_cast<double>(samples.size())) - 1.0);

    return {
        samples.front(),
        median,
        mean,
        std::sqrt(variance),
        samples[p95_index],
    };
}

void print_stats(const std::string_view label, const BenchmarkStats& stats, const std::string_view unit) {
    std::cout << "  " << label << " min: " << stats.min << ' ' << unit << "\n";
    std::cout << "  " << label << " median: " << stats.median << ' ' << unit << "\n";
    std::cout << "  " << label << " mean: " << stats.mean << ' ' << unit << "\n";
    std::cout << "  " << label << " stdev: " << stats.stdev << ' ' << unit << "\n";
    std::cout << "  " << label << " p95: " << stats.p95 << ' ' << unit << "\n";
}

void print_scaled_stats(const std::string_view label, const BenchmarkStats& stats, const ScaledUnit unit) {
    std::cout << "  " << label << " min: " << (stats.min / unit.scale) << ' ' << unit.unit << "\n";
    std::cout << "  " << label << " median: " << (stats.median / unit.scale) << ' ' << unit.unit << "\n";
    std::cout << "  " << label << " mean: " << (stats.mean / unit.scale) << ' ' << unit.unit << "\n";
    std::cout << "  " << label << " stdev: " << (stats.stdev / unit.scale) << ' ' << unit.unit << "\n";
    std::cout << "  " << label << " p95: " << (stats.p95 / unit.scale) << ' ' << unit.unit << "\n";
}

} // namespace

BenchmarkRegistry& global_registry() {
    static BenchmarkRegistry registry;
    return registry;
}

void BenchmarkRegistry::add_kernel(const Kernel& kernel) {
    kernels_.push_back(kernel);
}

const std::vector<Kernel>& BenchmarkRegistry::kernels() const noexcept {
    return kernels_;
}

void BenchmarkRegistry::add_validator(const KernelKind kind, const KernelValidationFn validator) {
    if (kind == KernelKind::unknown) {
        return;
    }

    validators_[kind] = validator;
}

KernelValidationFn BenchmarkRegistry::find_validator(const KernelKind kind) const noexcept {
    const auto it = validators_.find(kind);
    if (it == validators_.end()) {
        return nullptr;
    }
    return it->second;
}

const Kernel* BenchmarkRegistry::find_kernel(const std::string& name) const noexcept {
    for (const auto& k : kernels_) {
        if (k.name == name) {
            return &k;
        }
    }
    return nullptr;
}

std::optional<BenchmarkResult> BenchmarkRegistry::run_kernel(const std::string& name, BenchmarkContext& ctx) const {
    const Kernel* k = this->find_kernel(name);
    if (!k) {
        std::cerr << "Kernel not found: " << name << "\n";
        return std::nullopt;
    }

    return run_kernel(*k, ctx);
}

std::optional<BenchmarkResult> BenchmarkRegistry::run_kernel(const Kernel& kernel, BenchmarkContext& ctx) const {
    const BenchmarkConfig& config = ctx.config;
    const size_t elements = ctx.src1.size();
    if (config.iterations == 0) {
        std::cerr << "Kernel needs at least one measured iteration: " << kernel.name << "\n";
        return std::nullopt;
    }
    if (config.repetitions == 0) {
        std::cerr << "Kernel needs at least one measured repetition: " << kernel.name << "\n";
        return std::nullopt;
    }

    const size_t configured_elements = config.bytes / sizeof(double);
    if (configured_elements != elements) {
        std::cerr << "BenchmarkConfig bytes do not match context size for kernel: " << kernel.name << "\n";
        return std::nullopt;
    }

    if (config.cpu_id >= 0 && config.cpu_id < get_cpu_count()) {
        const auto [success, message] = set_cpu_affinity(config.cpu_id);
        if (!success) {
            std::cerr << "Failed to set CPU affinity: " << message << "\n";
        } else {
            std::cout << "Set CPU affinity to " << config.cpu_id << "\n";
        }
    }

    reset_output(ctx);
    for (size_t i = 0; i < config.warmup_iterations; ++i) {
        kernel.kernel(ctx, elements);
    }

    reset_output(ctx);
    ctx.result = std::numeric_limits<double>::quiet_NaN();
    using clock = std::chrono::steady_clock;

    std::vector<double> timing_samples;
    timing_samples.reserve(config.repetitions);

    for (size_t repetition = 0; repetition < config.repetitions; ++repetition) {
        reset_output(ctx);
        ctx.result = std::numeric_limits<double>::quiet_NaN();

        std::atomic_signal_fence(std::memory_order_seq_cst);
        const auto t0 = clock::now();
        for (size_t i = 0; i < config.iterations; ++i) {
            kernel.kernel(ctx, elements);
        }
        const auto t1 = clock::now();
        std::atomic_signal_fence(std::memory_order_seq_cst);

        const std::chrono::duration<double> duration = t1 - t0;
        const double seconds = duration.count();
        if (seconds <= 0.0) {
            std::cerr << "Kernel measured non-positive time\n";
            return std::nullopt;
        }
        timing_samples.push_back(seconds);
    }

    bool validation_passed = true;
    const KernelValidationFn validator = this->find_validator(kernel.kind);
    if (validator != nullptr) {
        if (!validator(ctx)) {
            std::cerr << "Validation failed for kernel: " << kernel.name << "\n";
            validation_passed = false;
        }
    }

    const double bytes = static_cast<double>(elements)
        * static_cast<double>(kernel.bytes_per_element)
        * static_cast<double>(config.iterations);
    std::vector<double> time_per_iteration_samples;
    time_per_iteration_samples.reserve(timing_samples.size());
    std::vector<double> bandwidth_samples;
    bandwidth_samples.reserve(timing_samples.size());

    for (const double seconds : timing_samples) {
        time_per_iteration_samples.push_back(seconds / static_cast<double>(config.iterations));
        bandwidth_samples.push_back(bytes / seconds);
    }

    const BenchmarkStats time_stats = compute_stats(timing_samples);
    const BenchmarkStats time_per_iteration_stats = compute_stats(time_per_iteration_samples);
    const BenchmarkStats bandwidth_stats = compute_stats(bandwidth_samples);
    const ScaledUnit bandwidth_unit = choose_bandwidth_unit(bandwidth_stats.mean);

    // Print a simple result summary
    std::cout << "Kernel: " << kernel.name << "\n";
    std::cout << "  Elements: " << elements << "\n";
    std::cout << "  Warmup iterations: " << config.warmup_iterations << "\n";
    std::cout << "  Iterations: " << config.iterations << "\n";
    std::cout << "  Repetitions: " << config.repetitions << "\n";
    std::cout << "  Seed: " << config.seed << "\n";
    print_stats("Time", time_stats, "s");
    print_stats("Time/iteration", time_per_iteration_stats, "s");
    print_scaled_stats("Bandwidth", bandwidth_stats, bandwidth_unit);

    if (config.cpu_id >= 0) {
        reset_cpu_affinity();
    }

    return BenchmarkResult{
        kernel.name,
        elements,
        kernel.bytes_per_element,
        timing_samples,
        time_per_iteration_samples,
        bandwidth_samples,
        time_stats,
        time_per_iteration_stats,
        bandwidth_stats,
        std::string(bandwidth_unit.unit),
        bandwidth_unit.scale,
        validation_passed,
    };
}

std::vector<BenchmarkResult> BenchmarkRegistry::run_all(BenchmarkContext& ctx) const {
    std::vector<const Kernel*> kernels;
    kernels.reserve(kernels_.size());
    for (const Kernel& kernel : kernels_) {
        kernels.push_back(&kernel);
    }

    if (ctx.config.randomize_kernel_order) {
        std::mt19937_64 rng(ctx.config.seed);
        std::shuffle(kernels.begin(), kernels.end(), rng);
    }

    std::vector<BenchmarkResult> results;
    results.reserve(kernels.size());
    for (const Kernel* kernel : kernels) {
        if (std::optional<BenchmarkResult> result = run_kernel(*kernel, ctx)) {
            results.push_back(std::move(*result));
        }
    }
    return results;
}

} // namespace membench
