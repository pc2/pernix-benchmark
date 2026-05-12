#include <membench/benchmark.h>

#include <CLI/CLI.hpp>
#include <cctype>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <membench/results_json.h>

namespace {

std::optional<size_t> parse_size_bytes(const std::string& value) {
    std::string compact;
    compact.reserve(value.size());
    for (const unsigned char c : value) {
        if (!std::isspace(c)) {
            compact.push_back(static_cast<char>(std::toupper(c)));
        }
    }

    if (compact.empty()) {
        return std::nullopt;
    }

    size_t number_end = 0;
    while (number_end < compact.size() && std::isdigit(static_cast<unsigned char>(compact[number_end]))) {
        ++number_end;
    }

    if (number_end == 0) {
        return std::nullopt;
    }

    size_t multiplier = 1;
    const std::string unit = compact.substr(number_end);
    if (unit.empty() || unit == "B") {
        multiplier = 1;
    } else if (unit == "KB") {
        multiplier = 1024;
    } else if (unit == "MB") {
        multiplier = 1024 * 1024;
    } else if (unit == "GB") {
        multiplier = 1024 * 1024 * 1024;
    } else {
        return std::nullopt;
    }

    size_t number = 0;
    for (size_t i = 0; i < number_end; ++i) {
        const size_t digit = static_cast<size_t>(compact[i] - '0');
        if (number > (std::numeric_limits<size_t>::max() - digit) / 10) {
            return std::nullopt;
        }
        number = number * 10 + digit;
    }

    if (number > std::numeric_limits<size_t>::max() / multiplier) {
        return std::nullopt;
    }
    return number * multiplier;
}

} // namespace

int main(int argc, char** argv) {
    membench::BenchmarkConfig config = {
        8 * 1024,
        0,
        1,
        100,
        0.1,
        33378,
        -1,
        false,
    };
    std::string bytes_value = "8KB";
    std::string kernel_name;
    std::string output_path;
    bool list_kernels = false;

    CLI::App app{"Memory bandwidth benchmark"};
    app.add_option("-b,--bytes", bytes_value, "Bytes per benchmark array, with optional B, KB, MB, or GB suffix");
    app.add_option("-i,--iterations", config.iterations, "Iterations per measured repetition, or 0 to auto-calibrate");
    app.add_option("-r,--repetitions", config.repetitions, "Measured timing samples per kernel");
    app.add_option("--warmup-iterations", config.warmup_iterations, "Warmup iterations before measuring");
    app.add_option("--target-seconds", config.target_seconds_per_repetition, "Target seconds per measured repetition when iterations is 0");
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

    const std::optional<size_t> parsed_bytes = parse_size_bytes(bytes_value);
    if (!parsed_bytes) {
        std::cerr << "bytes must be an integer with optional B, KB, MB, or GB suffix\n";
        return 1;
    }
    config.bytes = *parsed_bytes;

    if (config.repetitions == 0) {
        std::cerr << "repetitions must be greater than zero\n";
        return 1;
    }
    if (config.target_seconds_per_repetition <= 0.0) {
        std::cerr << "target seconds must be greater than zero\n";
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

    if (!output_path.empty() && !membench::write_results_json(output_path, config, results)) {
        return 1;
    }

    return 0;
}
