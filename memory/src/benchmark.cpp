
#include <membench/benchmark.h>
#include <membench/cpu.h>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>
#include <string_view>

namespace membench {

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

void BenchmarkRegistry::run_kernel(const std::string& name, BenchmarkContext& ctx) const {
    const Kernel* k = this->find_kernel(name);
    if (!k) {
        std::cerr << "Kernel not found: " << name << "\n";
        return;
    }

    run_kernel(*k, ctx);
}

void BenchmarkRegistry::run_kernel(const Kernel& kernel, BenchmarkContext& ctx) const {
    const BenchmarkConfig& config = ctx.config;
    const size_t elements = ctx.src1.size();
    if (config.iterations == 0) {
        std::cerr << "Kernel needs at least one measured iteration: " << kernel.name << "\n";
        return;
    }

    const size_t configured_elements = config.bytes / sizeof(double);
    if (configured_elements != elements) {
        std::cerr << "BenchmarkConfig bytes do not match context size for kernel: " << kernel.name << "\n";
        return;
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
        return;
    }

    const KernelValidationFn validator = this->find_validator(kernel.kind);
    if (validator != nullptr) {
        if (!validator(ctx)) {
            std::cerr << "Validation failed for kernel: " << kernel.name << "\n";
        }
    }

    const double bytes = static_cast<double>(elements)
        * static_cast<double>(kernel.bytes_per_element)
        * static_cast<double>(config.iterations);
    const double bandwidth = bytes / seconds; // bytes per second

    // Print a simple result summary
    std::cout << "Kernel: " << kernel.name << "\n";
    std::cout << "  Elements: " << elements << "\n";
    std::cout << "  Warmup iterations: " << config.warmup_iterations << "\n";
    std::cout << "  Iterations: " << config.iterations << "\n";
    std::cout << "  Seed: " << config.seed << "\n";
    std::cout << "  Time(s): " << seconds << "\n";
    std::cout << "  Time/iteration(s): " << (seconds / static_cast<double>(config.iterations)) << "\n";
    std::cout << "  Bandwidth: " << (bandwidth / (1024.0*1024.0)) << " MiB/s\n";

    if (config.cpu_id >= 0) {
        reset_cpu_affinity();
    }
}

void BenchmarkRegistry::run_all(BenchmarkContext& ctx) const {
    for (const Kernel& k : kernels_) {
        run_kernel(k, ctx);
    }
}

} // namespace membench
