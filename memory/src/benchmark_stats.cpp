#include <membench/benchmark_stats.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace membench {

BenchmarkStats compute_stats(std::vector<double> samples) {
    std::ranges::sort(samples);

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

} // namespace membench
