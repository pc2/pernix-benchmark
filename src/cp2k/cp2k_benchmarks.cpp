#include <benchmark/benchmark.h>

using namespace benchmark;
using namespace benchmark::internal;

extern "C" {
double cp2k_compression(State &state, int iterations, int width, int blocks);

double cp2k_decompression(State &state, int iterations, int width, int blocks);
}

template<class... Args>
void BM_cp2k_compression(State &state, Args &&... args) {
    auto args_tuple = std::make_tuple(std::move(args)...);
    const int width = std::get<0>(args_tuple);
    const int blocks = state.range(0);
    constexpr uint32_t internal_iterations = 1 << 4; // 16
    const int elements_per_block = 512 / width;

    for (auto _: state) {
        const double time = cp2k_compression(state, internal_iterations, width, blocks);
        state.SetIterationTime(time);
    }

    const int64_t total_iterations = internal_iterations * state.iterations();

    const double bytes_processed = (blocks * elements_per_block) * static_cast<double>(total_iterations) * (
                                       4 + (static_cast<float>(width) / 8));

    state.SetItemsProcessed(total_iterations * blocks);
    state.SetBytesProcessed(static_cast<int64_t>(bytes_processed));
}

template<class... Args>
void BM_cp2k_decompression(State &state, Args &&... args) {
    auto args_tuple = std::make_tuple(std::move(args)...);
    const int width = std::get<0>(args_tuple);
    const int blocks = state.range(0);
    constexpr uint32_t internal_iterations = 1 << 10; // 1024
    const int elements_per_block = 512 / width;

    for (auto _: state) {
        const double time = cp2k_decompression(state, internal_iterations, width, blocks);
        state.SetIterationTime(time);
    }

    const int64_t total_iterations = internal_iterations * state.iterations();

    const double bytes_processed = (blocks * elements_per_block) * static_cast<double>(total_iterations) * (
                                       4 + (static_cast<float>(width) / 8));

    state.SetItemsProcessed(total_iterations * blocks);
    state.SetBytesProcessed(static_cast<int64_t>(bytes_processed));
}

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
