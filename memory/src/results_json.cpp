#include <membench/results_json.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <utility>

namespace membench {
namespace {

using json = nlohmann::json;

json stats_to_json(const BenchmarkStats& stats) {
    return {
        {"min", stats.min},
        {"median", stats.median},
        {"mean", stats.mean},
        {"stdev", stats.stdev},
        {"p95", stats.p95},
    };
}

json result_to_json(const BenchmarkResult& result) {
    return {
        {"kernel", result.kernel},
        {"elements", result.elements},
        {"bytes_per_element", result.bytes_per_element},
        {"iterations", result.iterations},
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
    const BenchmarkConfig& config,
    const std::vector<BenchmarkResult>& results) {
    json result_items = json::array();
    for (const BenchmarkResult& result : results) {
        result_items.push_back(result_to_json(result));
    }

    return {
        {"config", {
            {"bytes", config.bytes},
            {"iterations", config.iterations},
            {"repetitions", config.repetitions},
            {"warmup_iterations", config.warmup_iterations},
            {"target_seconds_per_repetition", config.target_seconds_per_repetition},
            {"seed", config.seed},
            {"cpu_id", config.cpu_id},
            {"randomize_kernel_order", config.randomize_kernel_order},
        }},
        {"results", std::move(result_items)},
    };
}

} // namespace

bool write_results_json(
    const std::string& path,
    const BenchmarkConfig& config,
    const std::vector<BenchmarkResult>& results) {
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

} // namespace membench
