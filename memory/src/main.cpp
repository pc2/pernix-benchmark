#include <CLI/CLI.hpp>

#include <membench/benchmark.h>

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    membench::BenchmarkConfig config = {
        8 * 1024,
        10000000,
        1,
        100,
        33378,
        -1,
        false,
    };
    std::string kernel_name;
    bool list_kernels = false;

    CLI::App app{"Memory bandwidth benchmark"};
    app.add_option("--bytes", config.bytes, "Bytes per benchmark array");
    app.add_option("-i,--iterations", config.iterations, "Iterations per measured repetition");
    app.add_option("-r,--repetitions", config.repetitions, "Measured timing samples per kernel");
    app.add_option("--warmup-iterations", config.warmup_iterations, "Warmup iterations before measuring");
    app.add_option("--seed", config.seed, "Seed for data initialization and randomized kernel order");
    app.add_option("--cpu-id", config.cpu_id, "CPU id for affinity, or a negative value to disable affinity");
    app.add_option("-k,--kernel", kernel_name, "Run only one kernel by name");
    app.add_flag("--randomize-kernel-order", config.randomize_kernel_order, "Shuffle kernel order before run-all");
    app.add_flag("--list-kernels", list_kernels, "Print registered kernel names and exit");
    CLI11_PARSE(app, argc, argv);

    const membench::BenchmarkRegistry& registry = membench::global_registry();
    if (list_kernels) {
        for (const membench::Kernel& kernel : registry.kernels()) {
            std::cout << kernel.name << "\n";
        }
        return 0;
    }

    if (config.iterations == 0) {
        std::cerr << "iterations must be greater than zero\n";
        return 1;
    }
    if (config.repetitions == 0) {
        std::cerr << "repetitions must be greater than zero\n";
        return 1;
    }
    if (config.bytes < sizeof(double) || config.bytes % sizeof(double) != 0) {
        std::cerr << "bytes must be a positive multiple of " << sizeof(double) << "\n";
        return 1;
    }
    if (!kernel_name.empty() && registry.find_kernel(kernel_name) == nullptr) {
        std::cerr << "Kernel not found: " << kernel_name << "\n";
        return 1;
    }

    membench::BenchmarkContext ctx(config);

    if (!kernel_name.empty()) {
        registry.run_kernel(kernel_name, ctx);
    } else {
        registry.run_all(ctx);
    }

    return 0;
}
