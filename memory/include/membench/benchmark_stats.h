#ifndef MEMBENCH_BENCHMARK_STATS_H
#define MEMBENCH_BENCHMARK_STATS_H

#include <membench/benchmark.h>

#include <vector>

namespace membench {

BenchmarkStats compute_stats(std::vector<double> samples);

} // namespace membench

#endif  // MEMBENCH_BENCHMARK_STATS_H
