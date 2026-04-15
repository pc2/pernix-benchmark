#include <benchmark.h>
#include <pernix/pernix.h>

template<uint8_t BIT_WIDTH, bool DISABLE_MEM, typename ValueT>
class BenchmarkCompressorAVX512VBMI : public BenchmarkCompressor<BIT_WIDTH, DISABLE_MEM, ValueT> {
public:
    __always_inline int compress(const ValueT *input, const ValueT scale, uint8_t *output) override {
        return pernix::mm512_compress_block_avx512vbmi<BIT_WIDTH>(input, scale, output);
    }
};

#define BENCHMARK_COMPRESS_BLOCKS_AVX512VBMI_FUNCTION(N, MEM, TYPE, TAG) \
    static void BM_compress_##TAG##_##MEM##_##N(benchmark::State& state) { \
        BM_compress_blocks<N, true, MEM, TYPE, BenchmarkCompressorAVX512VBMI<N, MEM, TYPE>>(state); \
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

PERNIX_FOR_EACH_BIT_WIDTH(BENCHMARK_COMPRESS_BLOCKS_AVX512VBMI_FUNCTION, true, float, avx512vbmif32)
PERNIX_FOR_EACH_BIT_WIDTH(BENCHMARK_COMPRESS_BLOCKS_AVX512VBMI_FUNCTION, false, float, avx512vbmif32)
PERNIX_FOR_EACH_BIT_WIDTH(BENCHMARK_COMPRESS_BLOCKS_AVX512VBMI_FUNCTION, true, double, avx512vbmif64)
PERNIX_FOR_EACH_BIT_WIDTH(BENCHMARK_COMPRESS_BLOCKS_AVX512VBMI_FUNCTION, false, double, avx512vbmif64)

#undef PERNIX_FOR_EACH_BIT_WIDTH
#undef BENCHMARK_COMPRESS_BLOCKS_AVX512VBMI_FUNCTION
