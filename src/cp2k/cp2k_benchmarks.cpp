#include <benchmark.h>

#include <cstdint>
#include <type_traits>

extern "C" {
void cp2k_compress_block_f32(const float* input, float scale, std::int64_t* packed, int bit_width);
void cp2k_decompress_block_f32(const std::int64_t* packed, float scale, float* output, int bit_width);
void cp2k_compress_block_f64(const double* input, double scale, std::int64_t* packed, int bit_width);
void cp2k_decompress_block_f64(const std::int64_t* packed, double scale, double* output, int bit_width);
}

template <std::uint8_t BIT_WIDTH, bool DISABLE_MEM, typename ValueT>
class BenchmarkCompressorCP2K : public BenchmarkCompressor<BIT_WIDTH, DISABLE_MEM, ValueT> {
    static_assert(std::is_same_v<ValueT, float> || std::is_same_v<ValueT, double>);

public:
    int compress_blocks(const ValueT* input, const ValueT scale, std::uint8_t* output, const std::uint32_t blocks) override {
        constexpr std::size_t elements_per_block = 512 / BIT_WIDTH;
        for (std::uint32_t block = 0; block < blocks; ++block) {
            if constexpr (std::is_same_v<ValueT, float>) {
                cp2k_compress_block_f32(input, scale, reinterpret_cast<std::int64_t*>(output), BIT_WIDTH);
            } else {
                cp2k_compress_block_f64(input, scale, reinterpret_cast<std::int64_t*>(output), BIT_WIDTH);
            }
            input += elements_per_block;
            output += 64;
        }
        return 0;
    }
};

template <std::uint8_t BIT_WIDTH, bool DISABLE_MEM, typename ValueT>
class BenchmarkDecompressorCP2K : public BenchmarkDecompressor<BIT_WIDTH, true, DISABLE_MEM, ValueT> {
    static_assert(std::is_same_v<ValueT, float> || std::is_same_v<ValueT, double>);

public:
    int decompress_blocks(const std::uint8_t* input, const ValueT scale, ValueT* output, const std::uint32_t blocks) override {
        constexpr std::size_t elements_per_block = 512 / BIT_WIDTH;
        for (std::uint32_t block = 0; block < blocks; ++block) {
            if constexpr (std::is_same_v<ValueT, float>) {
                cp2k_decompress_block_f32(reinterpret_cast<const std::int64_t*>(input), scale, output, BIT_WIDTH);
            } else {
                cp2k_decompress_block_f64(reinterpret_cast<const std::int64_t*>(input), scale, output, BIT_WIDTH);
            }
            input += 64;
            output += elements_per_block;
        }
        return 0;
    }
};

#define BENCHMARK_CP2K_FOR_MODE(N, TAG, TYPE, MEM)                                                \
    static void BM_compress_cp2k##TAG##_##MEM##_##N(benchmark::State& state) {                    \
        BM_compress_blocks<N, true, MEM, TYPE, BenchmarkCompressorCP2K<N, MEM, TYPE>>(state);     \
    }                                                                                             \
    BENCHMARK_COMPRESS_BLOCKS_REGISTER(compress_cp2k##TAG##_##MEM##_##N, N, MEM, TYPE);           \
    static void BM_decompress_cp2k##TAG##_##MEM##_##N(benchmark::State& state) {                  \
        BM_decompress_blocks<N, true, MEM, TYPE, BenchmarkDecompressorCP2K<N, MEM, TYPE>>(state); \
    }                                                                                             \
    BENCHMARK_DECOMPRESS_BLOCKS_REGISTER(decompress_cp2k##TAG##_##MEM##_##N, N, MEM, TYPE);

#define BENCHMARK_CP2K_FOR_TYPE(N, TAG, TYPE)    \
    BENCHMARK_CP2K_FOR_MODE(N, TAG, TYPE, true); \
    BENCHMARK_CP2K_FOR_MODE(N, TAG, TYPE, false)

#define BENCHMARK_CP2K(N)                   \
    BENCHMARK_CP2K_FOR_TYPE(N, f32, float); \
    BENCHMARK_CP2K_FOR_TYPE(N, f64, double)

#define CP2K_FOR_EACH_BIT_WIDTH(M) \
    M(1);                          \
    M(2);                          \
    M(3);                          \
    M(4);                          \
    M(5);                          \
    M(6);                          \
    M(7);                          \
    M(8);                          \
    M(9);                          \
    M(10);                         \
    M(11);                         \
    M(12);                         \
    M(13);                         \
    M(14);                         \
    M(15);                         \
    M(16);                         \
    M(17);                         \
    M(18);                         \
    M(19);                         \
    M(20);                         \
    M(21);                         \
    M(22);                         \
    M(23);                         \
    M(24)

CP2K_FOR_EACH_BIT_WIDTH(BENCHMARK_CP2K);

#undef CP2K_FOR_EACH_BIT_WIDTH
#undef BENCHMARK_CP2K
#undef BENCHMARK_CP2K_FOR_TYPE
#undef BENCHMARK_CP2K_FOR_MODE
