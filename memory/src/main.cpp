#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include <membench/benchmark.h>

#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;

json stats_to_json(const membench::BenchmarkStats& stats) {
    return {
        {"min", stats.min},
        {"median", stats.median},
        {"mean", stats.mean},
        {"stdev", stats.stdev},
        {"p95", stats.p95},
    };
}

json result_to_json(const membench::BenchmarkResult& result) {
    return {
        {"kernel", result.kernel},
        {"elements", result.elements},
        {"bytes_per_element", result.bytes_per_element},
        {"validation_passed", result.validation_passed},
        {"time_seconds", stats_to_json(result.time_seconds)},
        {"time_per_iteration_seconds", stats_to_json(result.time_per_iteration_seconds)},
        {"bandwidth_bytes_per_second", stats_to_json(result.bandwidth_bytes_per_second)},
        {"bandwidth_display_unit", result.bandwidth_unit},
        {"bandwidth_display_scale", result.bandwidth_unit_scale},
        {"samples", {
            {"time_seconds", result.time_seconds_samples},
            {"time_per_iteration_seconds", result.time_per_iteration_seconds_samples},
            {"bandwidth_bytes_per_second", result.bandwidth_bytes_per_second_samples},
        }},
    };
}

json results_to_json(
    const membench::BenchmarkConfig& config,
    const std::vector<membench::BenchmarkResult>& results) {
    json result_items = json::array();
    for (const membench::BenchmarkResult& result : results) {
        result_items.push_back(result_to_json(result));
    }

    return {
        {"config", {
            {"bytes", config.bytes},
            {"iterations", config.iterations},
            {"repetitions", config.repetitions},
            {"warmup_iterations", config.warmup_iterations},
            {"seed", config.seed},
            {"cpu_id", config.cpu_id},
            {"randomize_kernel_order", config.randomize_kernel_order},
        }},
        {"results", std::move(result_items)},
    };
}

bool write_results_json(
    const std::string& path,
    const membench::BenchmarkConfig& config,
    const std::vector<membench::BenchmarkResult>& results) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Failed to open output file: " << path << "\n";
        return false;
    }

    out << results_to_json(config, results).dump(2) << "\n";
    if (!out) {
        std::cerr << "Failed to write output file: " << path << "\n";
        return false;
    }
    return true;
}

} // namespace

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
    std::string output_path;
    bool list_kernels = false;

    CLI::App app{"Memory bandwidth benchmark"};
    app.add_option("--bytes", config.bytes, "Bytes per benchmark array");
    app.add_option("-i,--iterations", config.iterations, "Iterations per measured repetition");
    app.add_option("-r,--repetitions", config.repetitions, "Measured timing samples per kernel");
    app.add_option("--warmup-iterations", config.warmup_iterations, "Warmup iterations before measuring");
    app.add_option("--seed", config.seed, "Seed for data initialization and randomized kernel order");
    app.add_option("--cpu-id", config.cpu_id, "CPU id for affinity, or a negative value to disable affinity");
    app.add_option("-k,--kernel", kernel_name, "Run only one kernel by name");
    app.add_option("-o,--output", output_path, "Write benchmark results as JSON to this file");
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

    std::vector<membench::BenchmarkResult> results;
    if (!kernel_name.empty()) {
        std::optional<membench::BenchmarkResult> result = registry.run_kernel(kernel_name, ctx);
        if (result) {
            results.push_back(std::move(*result));
        }
    } else {
        results = registry.run_all(ctx);
    }

    if (!output_path.empty() && !write_results_json(output_path, config, results)) {
        return 1;
    }

    return 0;
}
