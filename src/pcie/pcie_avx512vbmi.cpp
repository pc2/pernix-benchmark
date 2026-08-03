#include <pcie_benchmark.h>
#include <pernix/x86/avx512vbmi/avx512vbmi_compression.h>
#include <pernix/x86/avx512vbmi/avx512vbmi_decompression.h>
struct Avx512vbmiCompressor { template <std::uint8_t W, typename T> int operator()(const T* i,T s,std::uint8_t* o,std::uint32_t b) const { return pernix::mm512_compress_blocks_avx512vbmi<W,64>(i,s,o,b); } };
struct Avx512vbmiDecompressor { template <std::uint8_t W, typename T> int operator()(const std::uint8_t* i,T s,T* o,std::uint32_t b) const { return pernix::mm512_decompress_blocks_avx512vbmi<W,true,64>(i,s,o,b); } };
#define REGISTER(W,T,TAG) static void BM_pcie_h2d_##TAG##_##W(benchmark::State& s){pernix_benchmark::pcie::BM_pcie_h2d<W,T,Avx512vbmiCompressor>(s);} static void BM_pcie_d2h_##TAG##_##W(benchmark::State& s){pernix_benchmark::pcie::BM_pcie_d2h<W,T,Avx512vbmiDecompressor>(s);} PERNIX_PCIE_PAYLOADS(BM_pcie_h2d_##TAG##_##W,W,T); PERNIX_PCIE_PAYLOADS(BM_pcie_d2h_##TAG##_##W,W,T)
#define WIDTHS(T,TAG) REGISTER(1,T,TAG);REGISTER(2,T,TAG);REGISTER(3,T,TAG);REGISTER(4,T,TAG);REGISTER(5,T,TAG);REGISTER(6,T,TAG);REGISTER(7,T,TAG);REGISTER(8,T,TAG);REGISTER(9,T,TAG);REGISTER(10,T,TAG);REGISTER(11,T,TAG);REGISTER(12,T,TAG);REGISTER(13,T,TAG);REGISTER(14,T,TAG);REGISTER(15,T,TAG);REGISTER(16,T,TAG);REGISTER(17,T,TAG);REGISTER(18,T,TAG);REGISTER(19,T,TAG);REGISTER(20,T,TAG);REGISTER(21,T,TAG);REGISTER(22,T,TAG);REGISTER(23,T,TAG);REGISTER(24,T,TAG)
WIDTHS(float,avx512vbmif32); WIDTHS(double,avx512vbmif64);
