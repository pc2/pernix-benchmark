#include <pcie_benchmark.h>
#include <pernix/fallback/scalar_compression.h>
#include <pernix/fallback/scalar_decompression.h>

struct FallbackCompressor { template <std::uint8_t Width, typename T> int operator()(const T* input, T scale, std::uint8_t* output, std::uint32_t blocks) const { return pernix::compress_blocks_fallback<Width, 64>(input, scale, output, blocks); } };
struct FallbackDecompressor { template <std::uint8_t Width, typename T> int operator()(const std::uint8_t* input, T scale, T* output, std::uint32_t blocks) const { return pernix::decompress_blocks_fallback<Width, true, 64>(input, scale, output, blocks); } };

#define REGISTER(W, T, TAG) \
    static void BM_pcie_h2d_##TAG##_##W(benchmark::State& s) { pernix_benchmark::pcie::BM_pcie_h2d<W, T, FallbackCompressor>(s); } \
    static void BM_pcie_d2h_##TAG##_##W(benchmark::State& s) { pernix_benchmark::pcie::BM_pcie_d2h<W, T, FallbackDecompressor>(s); } \
    PERNIX_PCIE_PAYLOADS(BM_pcie_h2d_##TAG##_##W); PERNIX_PCIE_PAYLOADS(BM_pcie_d2h_##TAG##_##W)
#define WIDTHS(T, TAG) REGISTER(1,T,TAG); REGISTER(2,T,TAG); REGISTER(3,T,TAG); REGISTER(4,T,TAG); REGISTER(5,T,TAG); REGISTER(6,T,TAG); REGISTER(7,T,TAG); REGISTER(8,T,TAG); REGISTER(9,T,TAG); REGISTER(10,T,TAG); REGISTER(11,T,TAG); REGISTER(12,T,TAG); REGISTER(13,T,TAG); REGISTER(14,T,TAG); REGISTER(15,T,TAG); REGISTER(16,T,TAG); REGISTER(17,T,TAG); REGISTER(18,T,TAG); REGISTER(19,T,TAG); REGISTER(20,T,TAG); REGISTER(21,T,TAG); REGISTER(22,T,TAG); REGISTER(23,T,TAG); REGISTER(24,T,TAG)
WIDTHS(float, fallbackf32); WIDTHS(double, fallbackf64);
