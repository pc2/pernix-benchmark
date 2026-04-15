#include <benchmark.h>
#include <pernix/pernix.h>


template<uint8_t BIT_WIDTH, bool DISABLE_MEM, typename ValueT>
class BenchmarkDecompressorAVX2 : public BenchmarkDecompressor<BIT_WIDTH, true, DISABLE_MEM, ValueT> {
public:
    int decompress(const uint8_t *input, const ValueT scale, ValueT *output) override {
        return pernix::mm256_decompress_block_avx2<BIT_WIDTH>(input, scale, output);
    }
};

#define BENCHMARK_DECOMPRESS_BLOCKS_AVX2_FUNCTION(N, MEM, TYPE, TAG) \
static void BM_decompress_##TAG##_##MEM##_##N(benchmark::State& state) { \
BM_decompress_blocks<N, true, MEM, TYPE, BenchmarkDecompressorAVX2<N, MEM, TYPE>>(state); \
}                                                              \
BENCHMARK_DECOMPRESS_BLOCKS_REGISTER(decompress_##TAG##_##MEM##_##N);

#define PERNIX_FOR_EACH_BIT_WIDTH(M, MEM, TYPE, TAG) \
M(1, MEM, TYPE, TAG); \
M(2, MEM, TYPE, TAG); \
M(3, MEM, TYPE, TAG); \
M(4, MEM, TYPE, TAG); \
M(5, MEM, TYPE, TAG); \
M(6, MEM, TYPE, TAG); \
M(7, MEM, TYPE, TAG); \
M(8, MEM, TYPE, TAG); \
M(9, MEM, TYPE, TAG); \
M(10, MEM, TYPE, TAG); \
M(11, MEM, TYPE, TAG); \
M(12, MEM, TYPE, TAG); \
M(13, MEM, TYPE, TAG); \
M(14, MEM, TYPE, TAG); \
M(15, MEM, TYPE, TAG); \
M(16, MEM, TYPE, TAG); \
M(17, MEM, TYPE, TAG); \
M(18, MEM, TYPE, TAG); \
M(19, MEM, TYPE, TAG); \
M(20, MEM, TYPE, TAG); \
M(21, MEM, TYPE, TAG); \
M(22, MEM, TYPE, TAG); \
M(23, MEM, TYPE, TAG); \
M(24, MEM, TYPE, TAG)

PERNIX_FOR_EACH_BIT_WIDTH(BENCHMARK_DECOMPRESS_BLOCKS_AVX2_FUNCTION, true, float, avx2f32)
PERNIX_FOR_EACH_BIT_WIDTH(BENCHMARK_DECOMPRESS_BLOCKS_AVX2_FUNCTION, false, float, avx2f32)
PERNIX_FOR_EACH_BIT_WIDTH(BENCHMARK_DECOMPRESS_BLOCKS_AVX2_FUNCTION, true, double, avx2f64)
PERNIX_FOR_EACH_BIT_WIDTH(BENCHMARK_DECOMPRESS_BLOCKS_AVX2_FUNCTION, false, double, avx2f64)

#undef PERNIX_FOR_EACH_BIT_WIDTH
#undef BENCHMARK_DECOMPRESS_BLOCKS_AVX2_FUNCTION
