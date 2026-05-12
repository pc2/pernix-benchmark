#ifndef MEMBENCH_RESULTS_JSON_H
#define MEMBENCH_RESULTS_JSON_H

#include <membench/benchmark.h>

#include <string>
#include <vector>

namespace membench {

bool write_results_json(
    const std::string& path,
    const BenchmarkConfig& config,
    const std::vector<BenchmarkResult>& results);

} // namespace membench

#endif  // MEMBENCH_RESULTS_JSON_H
