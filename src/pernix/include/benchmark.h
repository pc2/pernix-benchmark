#ifndef LIBCOMPRESSION_BENCHMARK_H
#define LIBCOMPRESSION_BENCHMARK_H

#include <benchmark/benchmark.h>
#include <pernix/simd_compat.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

namespace detail {
    inline size_t round_up_to(const size_t n, const size_t align) {
        return (n + (align - 1)) & ~(align - 1);
    }
}

template<uint8_t BIT_WIDTH, typename ValueT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<ValueT>)
struct DecompressionBenchmarkSet {
    uint64_t number_of_blocks;

    alignas(64) uint8_t *input_ptr = nullptr;
    alignas(64) ValueT *output_ptr = nullptr;
    ValueT scale{};

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
        std::uniform_real_distribution<ValueT> scale_dis(static_cast<ValueT>(0.0001), static_cast<ValueT>(1.0));

        const size_t in_bytes = detail::round_up_to(64u * number_of_blocks, 64);
        const size_t out_bytes = detail::round_up_to(number_of_blocks * elements_per_block * sizeof(ValueT), 64);

        input_ptr = static_cast<uint8_t *>(std::aligned_alloc(64, in_bytes));
        output_ptr = static_cast<ValueT *>(std::aligned_alloc(64, out_bytes));
        if (!input_ptr || !output_ptr) std::abort();

        scale = scale_dis(gen);

        for (int64_t i = 0; i < (64u * number_of_blocks); ++i) {
            input_ptr[i] = static_cast<uint8_t>(dis(gen));
        }
    }
};

template<uint8_t BIT_WIDTH, typename ValueT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<ValueT>)
struct CompressionBenchmarkSet {
    uint64_t number_of_blocks;

    alignas(64) ValueT *input_ptr = nullptr;
    alignas(64) uint8_t *output_ptr = nullptr;
    ValueT scale{};

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
        std::uniform_real_distribution<ValueT> dis{};
        std::uniform_real_distribution<ValueT> scale_dis(static_cast<ValueT>(1.0), static_cast<ValueT>(10000.0));

        const size_t out_bytes = detail::round_up_to(64 * number_of_blocks, 64);
        const size_t in_bytes = detail::round_up_to(elements_per_block * number_of_blocks * sizeof(ValueT), 64);

        output_ptr = static_cast<uint8_t *>(std::aligned_alloc(64, out_bytes));
        input_ptr = static_cast<ValueT *>(std::aligned_alloc(64, in_bytes));
        if (!output_ptr || !input_ptr) std::abort();

        scale = scale_dis(gen);

        for (int64_t i = 0; i < (number_of_blocks * elements_per_block); ++i) {
            input_ptr[i] = dis(gen);
        }
    }
};

template<uint8_t BIT_WIDTH, bool SIGN_VALUES, bool DISABLE_MEM, typename ValueT>
class BenchmarkDecompressor {
public:
    virtual ~BenchmarkDecompressor() = default;

    virtual int decompress_blocks(const uint8_t *, ValueT, ValueT *, uint32_t) = 0;
};

template<uint8_t BIT_WIDTH, bool DISABLE_MEM, typename ValueT>
class BenchmarkCompressor {
public:
    virtual ~BenchmarkCompressor() = default;

    virtual int compress_blocks(const ValueT *, ValueT, uint8_t *, uint32_t) = 0;
};

#define BENCHMARK_DECOMPRESS_BLOCKS_REGISTER(name) \
    BENCHMARK(BM_##name)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)

#define BENCHMARK_COMPRESS_BLOCKS_REGISTER(name) \
    BENCHMARK(BM_##name)->RangeMultiplier(2)->Range(1 << 0, 1 << 22)


template<uint8_t BIT_WIDTH, bool SIGN_VALUES, bool DISABLE_MEM, typename ValueT, typename Decompressor>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<ValueT>)
__always_inline void BM_decompress_blocks(benchmark::State &state) {
    const size_t elements_per_block = 512 / BIT_WIDTH;
    const auto number_of_blocks = static_cast<size_t>(state.range(0));
    std::unique_ptr<BenchmarkDecompressor<BIT_WIDTH, SIGN_VALUES, DISABLE_MEM, ValueT> > decompressor = std::make_unique<
        Decompressor>();

    const auto benchmark_set = new DecompressionBenchmarkSet<BIT_WIDTH, ValueT>(static_cast<int64_t>(number_of_blocks));

    const size_t bytes_read_per_block = (elements_per_block * BIT_WIDTH + 7) / 8;
    const size_t bytes_written_per_block = elements_per_block * sizeof(ValueT);

    constexpr size_t core_batch_blocks = 1024;
    double sum = 0;
    if constexpr (DISABLE_MEM) {
        const size_t batch_blocks = std::min(number_of_blocks, core_batch_blocks);
        DecompressionBenchmarkSet<BIT_WIDTH, ValueT> core_set(batch_blocks);
        thread_local ValueT scale = static_cast<ValueT>(1.0);

        for (auto _: state) {
            for (size_t processed = 0; processed < number_of_blocks; processed += batch_blocks) {
                const auto blocks_this_call = static_cast<uint32_t>(std::min(batch_blocks, number_of_blocks - processed));
                decompressor->decompress_blocks(core_set.input_ptr, scale, core_set.output_ptr, blocks_this_call);
                sum += core_set.output_ptr[0];
                asm volatile("" ::"r"(core_set.input_ptr), "r"(core_set.output_ptr));
            }
        }
    } else {
        for (auto _: state) {
            decompressor->decompress_blocks(benchmark_set->input_ptr, benchmark_set->scale, benchmark_set->output_ptr,
                                             static_cast<uint32_t>(number_of_blocks));
            sum += benchmark_set->output_ptr[0];
            benchmark::DoNotOptimize(benchmark_set->input_ptr);
            benchmark::DoNotOptimize(benchmark_set->output_ptr);
            benchmark::ClobberMemory();
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

template<uint8_t BIT_WIDTH, bool SIGN_VALUES, bool DISABLE_MEM, typename ValueT, typename Compressor>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<ValueT>)
__always_inline void BM_compress_blocks(benchmark::State &state) {
    const size_t elements_per_block = 512 / BIT_WIDTH;
    const auto number_of_blocks = static_cast<size_t>(state.range(0));
    std::unique_ptr<BenchmarkCompressor<BIT_WIDTH, DISABLE_MEM, ValueT> > compressor = std::make_unique<Compressor>();
    auto benchmark_set = new CompressionBenchmarkSet<BIT_WIDTH, ValueT>(static_cast<int64_t>(number_of_blocks));

    const size_t bytes_read_per_block = elements_per_block * sizeof(ValueT);
    const size_t bytes_written_per_block = (elements_per_block * BIT_WIDTH + 7) / 8;

    constexpr size_t core_batch_blocks = 1024;
    uint64_t sum = 0;
    if constexpr (DISABLE_MEM) {
        const size_t batch_blocks = std::min(number_of_blocks, core_batch_blocks);
        CompressionBenchmarkSet<BIT_WIDTH, ValueT> core_set(batch_blocks);
        thread_local auto scale = static_cast<ValueT>(1.0);

        for (auto _: state) {
            for (size_t processed = 0; processed < number_of_blocks; processed += batch_blocks) {
                const auto blocks_this_call = static_cast<uint32_t>(std::min(batch_blocks, number_of_blocks - processed));
                compressor->compress_blocks(core_set.input_ptr, scale, core_set.output_ptr, blocks_this_call);
                sum += core_set.output_ptr[0];
                asm volatile("" ::"r"(core_set.input_ptr), "r"(core_set.output_ptr));
            }
        }
    } else {
        for (auto _: state) {
            compressor->compress_blocks(benchmark_set->input_ptr, benchmark_set->scale, benchmark_set->output_ptr,
                                        static_cast<uint32_t>(number_of_blocks));
            sum += benchmark_set->output_ptr[0];
            benchmark::DoNotOptimize(benchmark_set->input_ptr);
            benchmark::DoNotOptimize(benchmark_set->output_ptr);
            benchmark::ClobberMemory();
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
    state.counters["sum"] = static_cast<double>(sum);

    delete benchmark_set;
}


#endif //LIBCOMPRESSION_BENCHMARK_H
