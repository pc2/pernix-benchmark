#ifndef LIBCOMPRESSION_BENCHMARK_H
#define LIBCOMPRESSION_BENCHMARK_H

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>
#include <random>

namespace detail {
    inline size_t round_up_to(const size_t n, const size_t align) {
        return (n + (align - 1)) & ~(align - 1);
    }
}

template<uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
struct DecompressionBenchmarkSet {
    uint64_t number_of_blocks;

    alignas(64) uint8_t *input_ptr = nullptr;
    alignas(64) float_t *output_ptr = nullptr;
    std::vector<float_t> scales{};

    ~DecompressionBenchmarkSet() {
        if (input_ptr) std::free(input_ptr);
        if (output_ptr) std::free(output_ptr);
        input_ptr = nullptr;
        output_ptr = nullptr;
    }

    explicit DecompressionBenchmarkSet(const uint64_t number_of_blocks) : number_of_blocks(number_of_blocks) {
        constexpr int64_t elements_per_block = 512 / BIT_WIDTH;
        const int64_t total_bits = number_of_blocks * elements_per_block * BIT_WIDTH;
        const int64_t num_bytes = (total_bits + 7) / 8;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int8_t> dis{};
        std::uniform_real_distribution scale_dis(0.0001f, 1.0f);

        const size_t in_bytes = detail::round_up_to(static_cast<size_t>(num_bytes), 64);
        const size_t out_bytes =
                detail::round_up_to(number_of_blocks * elements_per_block * sizeof(float_t), 64);

        input_ptr = static_cast<uint8_t *>(std::aligned_alloc(64, in_bytes));
        output_ptr = static_cast<float_t *>(std::aligned_alloc(64, out_bytes));
        if (!input_ptr || !output_ptr) std::abort();

        scales.resize(number_of_blocks);

        for (auto &scale: scales) {
            scale = scale_dis(gen);
        }

        for (int64_t i = 0; i < num_bytes; ++i) {
            input_ptr[i] = static_cast<uint8_t>(dis(gen));
        }
    }
};

template<uint8_t BIT_WIDTH>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
struct CompressionBenchmarkSet {
    uint64_t number_of_blocks;

    alignas(64) float_t *input_ptr = nullptr;
    alignas(64) uint8_t *output_ptr = nullptr;
    std::vector<float_t> scales{};

    ~CompressionBenchmarkSet() {
        if (input_ptr) std::free(input_ptr);
        if (output_ptr) std::free(output_ptr);
        input_ptr = nullptr;
        output_ptr = nullptr;
    }

    explicit CompressionBenchmarkSet(const uint64_t number_of_blocks) : number_of_blocks(number_of_blocks) {
        constexpr int64_t elements_per_block = 512 / BIT_WIDTH;
        const int64_t total_bits = number_of_blocks * elements_per_block * BIT_WIDTH;
        const int64_t num_bytes = (total_bits + 7) / 8;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float_t> dis{};
        std::uniform_real_distribution scale_dis(0.0001f, 1.0f);

        const size_t out_bytes = detail::round_up_to(static_cast<size_t>(num_bytes), 64);
        const int64_t total_floats = number_of_blocks * elements_per_block;
        const size_t in_bytes =
                detail::round_up_to(number_of_blocks * elements_per_block * sizeof(float_t), 64);

        output_ptr = static_cast<uint8_t *>(std::aligned_alloc(64, out_bytes));
        input_ptr = static_cast<float_t *>(std::aligned_alloc(64, in_bytes));
        if (!output_ptr || !input_ptr) std::abort();

        scales.resize(number_of_blocks);

        for (auto &scale: scales) {
            scale = scale_dis(gen);
        }

        for (int64_t i = 0; i < total_floats; ++i) {
            input_ptr[i] = dis(gen);
        }
    }
};


#define BENCHMARK_DECOMPRESS_BLOCKS_REGISTER(name) \
    BENCHMARK(BM_##name)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)

#define BENCHMARK_DECOMPRESS_BLOCKS_FUNCTION(name, func, N, MEM)   \
    static void BM_##name##_##MEM##_##N(benchmark::State& state) { \
        BM_decompress_blocks<N, true, MEM>(state, func<N>);        \
    }                                                              \
    BENCHMARK_DECOMPRESS_BLOCKS_REGISTER(name##_##MEM##_##N);

#define BENCHMARK_COMPRESS_BLOCKS_REGISTER(name) \
    BENCHMARK(BM_##name)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)

#define BENCHMARK_COMPRESS_BLOCKS_FUNCTION(name, func, N, MEM)      \
    static void BM_##name##_##MEM##_##N(benchmark::State& state) {  \
        BM_compress_blocks<N, true, MEM>(state, func<N>);           \
    }                                                               \
    BENCHMARK_COMPRESS_BLOCKS_REGISTER(name##_##MEM##_##N);

template<uint8_t BIT_WIDTH, bool SIGN_VALUES, bool DISABLE_MEM>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__always_inline void BM_decompress_blocks(benchmark::State &state,
                                          const std::function<int(const uint8_t *, float_t, float_t *)> &
                                          decompress_function) {
    const size_t elements_per_block = 512 / BIT_WIDTH;
    const auto number_of_blocks = static_cast<size_t>(state.range(0));
    const auto benchmark_set = new DecompressionBenchmarkSet<BIT_WIDTH>(static_cast<int64_t>(number_of_blocks));

    const size_t bytes_read_per_block = (elements_per_block * BIT_WIDTH + 7) / 8;
    const size_t bytes_written_per_block = elements_per_block * sizeof(float_t);

    if constexpr (DISABLE_MEM) {
        alignas(64) thread_local uint8_t dummy_input[64] = {0};
        alignas(64) thread_local float_t dummy_output[elements_per_block * 1] = {0};

        for (auto _: state) {
            for (uint32_t block = 0; block < number_of_blocks; block++) {
                decompress_function(dummy_input, benchmark_set->scales[block], dummy_output);
                asm volatile("" ::"r"(dummy_input), "r"(dummy_output));
            }
        }
    } else {
        for (auto _: state) {
            alignas(64) const uint8_t *block_input = benchmark_set->input_ptr;
            alignas(64) float_t *block_output = benchmark_set->output_ptr;

            for (uint32_t block = 0; block < number_of_blocks; block++) {
                decompress_function(block_input + block * 64, benchmark_set->scales[block],
                                    block_output + block * elements_per_block);
                benchmark::DoNotOptimize(block_input);
                benchmark::DoNotOptimize(benchmark_set->scales.data());
                benchmark::DoNotOptimize(block_output);
                // block_input += 32;
                // block_output += elements_per_block;
                benchmark::ClobberMemory();
            }
        }
    }
    const auto iters = static_cast<uint64_t>(state.iterations());
    const auto blocks = static_cast<uint64_t>(number_of_blocks);

    const uint64_t bytes_per_block = bytes_read_per_block + bytes_written_per_block;
    state.SetBytesProcessed(static_cast<int64_t>(iters * blocks * bytes_per_block));

    const auto items = static_cast<int64_t>(iters * blocks * elements_per_block);
    state.SetItemsProcessed(items);

    delete benchmark_set;
}

template<uint8_t BIT_WIDTH, bool SIGN_VALUES, bool DISABLE_MEM>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__always_inline void BM_compress_blocks(benchmark::State &state,
                                        const std::function<int(const float_t *, float_t, uint8_t *)> &
                                        compress_function) {
    const size_t elements_per_block = 512 / BIT_WIDTH;
    const auto number_of_blocks = static_cast<size_t>(state.range(0));
    auto benchmark_set = new CompressionBenchmarkSet<BIT_WIDTH>(static_cast<int64_t>(number_of_blocks));

    const size_t bytes_read_per_block = elements_per_block * sizeof(float_t);
    const size_t bytes_written_per_block = (elements_per_block * BIT_WIDTH + 7) / 8;

    if constexpr (DISABLE_MEM) {
        alignas(64) thread_local float_t dummy_input[elements_per_block * 1] = {0};
        alignas(64) thread_local uint8_t dummy_output[64] = {0};

        for (auto _: state) {
            for (uint32_t block = 0; block < number_of_blocks; block++) {
                compress_function(dummy_input, benchmark_set->scales[block], dummy_output);
                asm volatile("" ::"r"(dummy_input), "r"(dummy_output));
            }
        }
    } else {
        for (auto _: state) {
            alignas(64) const float_t *block_input = benchmark_set->input_ptr;
            alignas(64) uint8_t *block_output = benchmark_set->output_ptr;

            for (uint32_t block = 0; block < number_of_blocks; block++) {
                compress_function(block_input + block * elements_per_block, benchmark_set->scales[block],
                                  block_output + block * 64);
                benchmark::DoNotOptimize(block_input);
                benchmark::DoNotOptimize(benchmark_set->scales.data());
                benchmark::DoNotOptimize(block_output);
                // block_input += elements_per_block;
                // block_output += 32;
                benchmark::ClobberMemory();
            }
        }
    }
    const auto iters = static_cast<uint64_t>(state.iterations());
    const auto blocks = static_cast<uint64_t>(number_of_blocks);

    const uint64_t bytes_per_block = bytes_read_per_block + bytes_written_per_block;
    state.SetBytesProcessed(static_cast<int64_t>(iters * blocks * bytes_per_block));

    const auto items = static_cast<int64_t>(iters * blocks * elements_per_block);
    state.SetItemsProcessed(items);

    delete benchmark_set;
}

#endif //LIBCOMPRESSION_BENCHMARK_H
