#ifndef LIBCOMPRESSION_BENCHMARK_H
#define LIBCOMPRESSION_BENCHMARK_H

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>
#include <iostream>
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
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int8_t> dis{};
        std::uniform_real_distribution scale_dis(0.0001f, 1.0f);

        const size_t in_bytes = detail::round_up_to(64u * number_of_blocks, 64);
        const size_t out_bytes = detail::round_up_to(number_of_blocks * elements_per_block * sizeof(float_t), 64);

        input_ptr = static_cast<uint8_t *>(std::aligned_alloc(64, in_bytes));
        output_ptr = static_cast<float_t *>(std::aligned_alloc(64, out_bytes));
        if (!input_ptr || !output_ptr) std::abort();

        scales.resize(number_of_blocks);

        for (auto &scale: scales) {
            scale = scale_dis(gen);
        }

        for (int64_t i = 0; i < (64u * number_of_blocks); ++i) {
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
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float_t> dis{};
        std::uniform_real_distribution scale_dis(1.0f, 10000.0f);

        const size_t out_bytes = detail::round_up_to(64 * number_of_blocks, 64);
        const size_t in_bytes = detail::round_up_to(elements_per_block * number_of_blocks * sizeof(float_t), 64);

        output_ptr = static_cast<uint8_t *>(std::aligned_alloc(64, out_bytes));
        input_ptr = static_cast<float_t *>(std::aligned_alloc(64, in_bytes));
        if (!output_ptr || !input_ptr) std::abort();

        scales.resize(number_of_blocks);

        for (auto &scale: scales) {
            scale = scale_dis(gen);
        }

        for (int64_t i = 0; i < (number_of_blocks * elements_per_block); ++i) {
            input_ptr[i] = dis(gen);
        }
    }
};

template<uint8_t BIT_WIDTH, bool SIGN_VALUES, bool DISABLE_MEM>
class BenchmarkDecompressor {
public:
    virtual ~BenchmarkDecompressor() = default;

    __always_inline virtual int decompress(const uint8_t *, const float_t, float_t *) = 0;
};

template<uint8_t BIT_WIDTH, bool DISABLE_MEM>
class BenchmarkCompressor {
public:
    virtual ~BenchmarkCompressor() = default;

    __always_inline virtual int compress(const float_t *, const float_t, uint8_t *) = 0;
};

#define BENCHMARK_DECOMPRESS_BLOCKS_REGISTER(name) \
    BENCHMARK(BM_##name)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)

#define BENCHMARK_COMPRESS_BLOCKS_REGISTER(name) \
    BENCHMARK(BM_##name)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)


template<uint8_t BIT_WIDTH, bool SIGN_VALUES, bool DISABLE_MEM, typename Decompressor>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__always_inline void BM_decompress_blocks(benchmark::State &state) {
    const size_t elements_per_block = 512 / BIT_WIDTH;
    const auto number_of_blocks = static_cast<size_t>(state.range(0));
    std::unique_ptr<BenchmarkDecompressor<BIT_WIDTH, SIGN_VALUES, DISABLE_MEM> > decompressor = std::make_unique<
        Decompressor>();

    const auto benchmark_set = new DecompressionBenchmarkSet<BIT_WIDTH>(static_cast<int64_t>(number_of_blocks));

    const size_t bytes_read_per_block = (elements_per_block * BIT_WIDTH + 7) / 8;
    const size_t bytes_written_per_block = elements_per_block * sizeof(float_t);

    double sum = 0;
    if constexpr (DISABLE_MEM) {
        alignas(64) thread_local uint8_t dummy_input[64] = {0};
        alignas(64) thread_local float_t dummy_output[elements_per_block * 1] = {0};
        thread_local float_t scale = 1.0f;

        for (auto _: state) {
            for (uint32_t block = 0; block < number_of_blocks; block++) {
                decompressor->decompress(dummy_input, scale, dummy_output);
                sum += dummy_output[0];
                asm volatile("" ::"r"(dummy_input), "r"(dummy_output));
            }
        }
    } else {
        for (auto _: state) {
            alignas(64) const uint8_t *block_input = benchmark_set->input_ptr;
            alignas(64) float_t *block_output = benchmark_set->output_ptr;

            for (uint32_t block = 0; block < number_of_blocks; block++) {
                decompressor->decompress(block_input, benchmark_set->scales[block], block_output);
                benchmark::DoNotOptimize(block_input);
                benchmark::DoNotOptimize(benchmark_set->scales.data());
                benchmark::DoNotOptimize(block_output);
                block_input += 64;
                block_output += elements_per_block;
                sum += block_output[0];
                benchmark::ClobberMemory();
            }
        }
    }
    const auto iters = static_cast<uint64_t>(state.iterations());
    const auto blocks = static_cast<uint64_t>(number_of_blocks);

    if constexpr (DISABLE_MEM) {
        state.SetBytesProcessed(static_cast<int64_t>(iters * blocks * bytes_read_per_block));
    } else {
        state.SetBytesProcessed(
            static_cast<int64_t>(iters * blocks * (bytes_read_per_block + bytes_written_per_block)));
    }

    const auto items = static_cast<int64_t>(iters * blocks);
    state.SetItemsProcessed(items);
    state.counters["sum"] = sum;

    delete benchmark_set;
}

template<uint8_t BIT_WIDTH, bool SIGN_VALUES, bool DISABLE_MEM, typename Compressor>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24)
__always_inline void BM_compress_blocks(benchmark::State &state) {
    const size_t elements_per_block = 512 / BIT_WIDTH;
    const auto number_of_blocks = static_cast<size_t>(state.range(0));
    std::unique_ptr<BenchmarkCompressor<BIT_WIDTH, DISABLE_MEM> > compressor = std::make_unique<Compressor>();
    auto benchmark_set = new CompressionBenchmarkSet<BIT_WIDTH>(static_cast<int64_t>(number_of_blocks));

    const size_t bytes_read_per_block = elements_per_block * sizeof(float_t);
    const size_t bytes_written_per_block = (elements_per_block * BIT_WIDTH + 7) / 8;

    uint64_t sum = 0;
    if constexpr (DISABLE_MEM) {
        alignas(64) thread_local float_t dummy_input[elements_per_block * 1] = {0};
        alignas(64) thread_local uint8_t dummy_output[64] = {0};
        thread_local float_t scale = 1.0f;

        for (auto _: state) {
            for (uint32_t block = 0; block < number_of_blocks; block++) {
                compressor->compress(dummy_input, scale, dummy_output);
                sum += dummy_output[0];
                asm volatile("" ::"r"(dummy_input), "r"(dummy_output));
            }
        }
    } else {
        for (auto _: state) {
            alignas(64) const float_t *block_input = benchmark_set->input_ptr;
            alignas(64) uint8_t *block_output = benchmark_set->output_ptr;

            for (uint32_t block = 0; block < number_of_blocks; block++) {
                compressor->compress(block_input, benchmark_set->scales[block], block_output);
                benchmark::DoNotOptimize(block_input);
                benchmark::DoNotOptimize(benchmark_set->scales.data());
                benchmark::DoNotOptimize(block_output);
                block_input += elements_per_block;
                block_output += 64;
                sum += block_output[0];
                benchmark::ClobberMemory();
            }
        }
    }

    const auto iters = static_cast<uint64_t>(state.iterations());

    if constexpr (DISABLE_MEM) {
        state.SetBytesProcessed(static_cast<int64_t>(iters * number_of_blocks * bytes_written_per_block));
    } else {
        state.SetBytesProcessed(
            static_cast<int64_t>(iters * number_of_blocks * (bytes_read_per_block + bytes_written_per_block)));
    }

    state.SetItemsProcessed(static_cast<int64_t>(iters * number_of_blocks));
    state.counters["sum"] = sum;

    delete benchmark_set;
}


#endif //LIBCOMPRESSION_BENCHMARK_H
