#include <benchmark.h>
#include <pernix/pernix.h>

template<uint8_t BIT_WIDTH, bool DISABLE_MEM, typename ValueT>
class BenchmarkCompressorBMI2 : public BenchmarkCompressor<BIT_WIDTH, DISABLE_MEM, ValueT> {
public:
    int compress(const ValueT *input, const ValueT scale, uint8_t *output) override {
        return pernix::mm256_compress_block_bmi2<BIT_WIDTH>(input, scale, output);
    }
};

#define BENCHMARK_COMPRESS_BLOCKS_BMI2_FUNCTION(N, MEM, TYPE, TAG) \
static void BM_compress_##TAG##_##MEM##_##N(benchmark::State& state) { \
BM_compress_blocks<N, true, MEM, TYPE, BenchmarkCompressorBMI2<N, MEM, TYPE>>(state); \
}                                                              \
BENCHMARK_COMPRESS_BLOCKS_REGISTER(compress_##TAG##_##MEM##_##N);

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

PERNIX_FOR_EACH_BIT_WIDTH(BENCHMARK_COMPRESS_BLOCKS_BMI2_FUNCTION, true, float, bmi2f32)
PERNIX_FOR_EACH_BIT_WIDTH(BENCHMARK_COMPRESS_BLOCKS_BMI2_FUNCTION, false, float, bmi2f32)
PERNIX_FOR_EACH_BIT_WIDTH(BENCHMARK_COMPRESS_BLOCKS_BMI2_FUNCTION, true, double, bmi2f64)
PERNIX_FOR_EACH_BIT_WIDTH(BENCHMARK_COMPRESS_BLOCKS_BMI2_FUNCTION, false, double, bmi2f64)

#undef PERNIX_FOR_EACH_BIT_WIDTH
#undef BENCHMARK_COMPRESS_BLOCKS_BMI2_FUNCTION
