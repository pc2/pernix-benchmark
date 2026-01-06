#include <benchmark/benchmark.h>

using namespace benchmark;
using namespace benchmark::internal;

extern "C" {
double cp2k_compression(State& state, int iterations, int width, int nv);

double cp2k_decompression(State& state, int iterations, int width, int nv);
}

template <class... Args>
void BM_cp2k_compression(State& state, Args&&... args) {
    auto args_tuple                        = std::make_tuple(std::move(args)...);
    const int width                        = std::get<0>(args_tuple);
    constexpr uint32_t internal_iterations = 1 << 10; // 1024

    constexpr uint32_t nv = 1024;
    for (auto _ : state) {
        const double time = cp2k_compression(state, internal_iterations, width, nv);
        state.SetIterationTime(time);
    }

    const int64_t total_iterations = internal_iterations * state.iterations();

    const double bytes_processed = nv * static_cast<double>(total_iterations) * (8 + (static_cast<float>(width) / 8));

    state.SetBytesProcessed(static_cast<int64_t>(bytes_processed));
    // state.counters["compressed_bytes_per_second"] = 0;
}

template <class... Args>
void BM_cp2k_decompression(State& state, Args&&... args) {
    auto args_tuple                        = std::make_tuple(std::move(args)...);
    const int width                        = std::get<0>(args_tuple);
    constexpr uint32_t internal_iterations = 1 << 10; // 1024

    constexpr uint32_t nv = 1024;
    for (auto _ : state) {
        const double time = cp2k_decompression(state, internal_iterations, width, nv);
        state.SetIterationTime(time);
    }

    const int64_t total_iterations = internal_iterations * state.iterations();

    const double bytes_processed = nv * static_cast<double>(total_iterations) * (8 + (static_cast<float>(width) / 8));

    state.SetBytesProcessed(static_cast<int64_t>(bytes_processed));
    // state.counters["compressed_bytes_per_second"] = 0;
}

// BENCHMARK(BM_cp2k_compression);
// BENCHMARK(BM_cp2k_decompression);

BENCHMARK_CAPTURE(BM_cp2k_compression, width8, 8)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();
BENCHMARK_CAPTURE(BM_cp2k_decompression, width8, 8)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();

BENCHMARK_CAPTURE(BM_cp2k_compression, width9, 9)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();
BENCHMARK_CAPTURE(BM_cp2k_decompression, width9, 9)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();

BENCHMARK_CAPTURE(BM_cp2k_compression, width10, 10)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();
BENCHMARK_CAPTURE(BM_cp2k_decompression, width10, 10)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();

BENCHMARK_CAPTURE(BM_cp2k_compression, width11, 11)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();
BENCHMARK_CAPTURE(BM_cp2k_decompression, width11, 11)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();

BENCHMARK_CAPTURE(BM_cp2k_compression, width12, 12)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();
BENCHMARK_CAPTURE(BM_cp2k_decompression, width12, 12)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();

BENCHMARK_CAPTURE(BM_cp2k_compression, width13, 13)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();
BENCHMARK_CAPTURE(BM_cp2k_decompression, width13, 13)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();

BENCHMARK_CAPTURE(BM_cp2k_compression, width14, 14)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();
BENCHMARK_CAPTURE(BM_cp2k_decompression, width14, 14)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();

BENCHMARK_CAPTURE(BM_cp2k_compression, width15, 15)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();
BENCHMARK_CAPTURE(BM_cp2k_decompression, width15, 15)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();

BENCHMARK_CAPTURE(BM_cp2k_compression, width16, 16)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();
BENCHMARK_CAPTURE(BM_cp2k_decompression, width16, 16)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)->UseManualTime();