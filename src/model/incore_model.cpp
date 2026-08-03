#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

#if defined(PERNIX_MODEL_ISA_AVX2)
#include <pernix/x86/avx2/avx2_compression.h>
#include <pernix/x86/avx2/avx2_decompression.h>
#define PERNIX_MODEL_ISA_TOKEN avx2
#define PERNIX_MODEL_COMPRESS(width, input, scale, output) \
    pernix::mm256_compress_block_avx2<width, 64>(input, scale, output)
#define PERNIX_MODEL_DECOMPRESS(width, input, scale, output) \
    pernix::mm256_decompress_block_avx2<width, true, 64>(input, scale, output)
#elif defined(PERNIX_MODEL_ISA_AVX512VBMI)
#include <pernix/x86/avx512vbmi/avx512vbmi_compression.h>
#include <pernix/x86/avx512vbmi/avx512vbmi_decompression.h>
#define PERNIX_MODEL_ISA_TOKEN avx512vbmi
#define PERNIX_MODEL_COMPRESS(width, input, scale, output) \
    pernix::mm512_compress_block_avx512vbmi<width, 64>(input, scale, output)
#define PERNIX_MODEL_DECOMPRESS(width, input, scale, output) \
    pernix::mm512_decompress_block_avx512vbmi<width, true, 64>(input, scale, output)
#else
#error "Select PERNIX_MODEL_ISA_AVX2 or PERNIX_MODEL_ISA_AVX512VBMI"
#endif

#define PERNIX_STRINGIFY_IMPL(value) #value
#define PERNIX_STRINGIFY(value) PERNIX_STRINGIFY_IMPL(value)
#define PERNIX_JOIN_IMPL(left, right) left##right
#define PERNIX_JOIN(left, right) PERNIX_JOIN_IMPL(left, right)
#define PERNIX_MODEL_NAME_IMPL(operation, isa, width) model_##operation##_##isa##_##width
#define PERNIX_MODEL_NAME(operation, isa, width) \
    PERNIX_MODEL_NAME_IMPL(operation, isa, width)

#if defined(PERNIX_MODEL_MCA)
#define PERNIX_MCA_BEGIN(name) \
    asm volatile("# LLVM-MCA-BEGIN " PERNIX_STRINGIFY(name) ::: "memory")
#define PERNIX_MCA_END(name) \
    asm volatile("# LLVM-MCA-END " PERNIX_STRINGIFY(name) ::: "memory")
#else
#define PERNIX_MCA_BEGIN(name)
#define PERNIX_MCA_END(name)
#endif

using Kernel = int (*)(const void*, float, void*);

#define PERNIX_DEFINE_WIDTH(width) \
    extern "C" __attribute__((noinline, used)) int \
    PERNIX_MODEL_NAME(compression, PERNIX_MODEL_ISA_TOKEN, width)( \
        const void* input, const float scale, void* output) { \
        PERNIX_MCA_BEGIN(PERNIX_MODEL_NAME(compression, PERNIX_MODEL_ISA_TOKEN, width)); \
        const int status = PERNIX_MODEL_COMPRESS(width, input, scale, output); \
        PERNIX_MCA_END(PERNIX_MODEL_NAME(compression, PERNIX_MODEL_ISA_TOKEN, width)); \
        return status; \
    } \
    extern "C" __attribute__((noinline, used)) int \
    PERNIX_MODEL_NAME(decompression, PERNIX_MODEL_ISA_TOKEN, width)( \
        const void* input, const float scale, void* output) { \
        PERNIX_MCA_BEGIN(PERNIX_MODEL_NAME(decompression, PERNIX_MODEL_ISA_TOKEN, width)); \
        const int status = PERNIX_MODEL_DECOMPRESS(width, input, scale, output); \
        PERNIX_MCA_END(PERNIX_MODEL_NAME(decompression, PERNIX_MODEL_ISA_TOKEN, width)); \
        return status; \
    }

#define PERNIX_FOR_EACH_POSTER_WIDTH(macro) \
    macro(2)  macro(3)  macro(4)  macro(5)  macro(6)  macro(7)  \
    macro(8)  macro(9)  macro(10) macro(11) macro(12) macro(13) \
    macro(14) macro(15) macro(16) macro(17) macro(18) macro(19) \
    macro(20) macro(21) macro(22) macro(23) macro(24)

PERNIX_FOR_EACH_POSTER_WIDTH(PERNIX_DEFINE_WIDTH)

struct KernelEntry {
    int width;
    Kernel compression;
    Kernel decompression;
};

#define PERNIX_KERNEL_ENTRY(width) \
    KernelEntry{width, \
        PERNIX_MODEL_NAME(compression, PERNIX_MODEL_ISA_TOKEN, width), \
        PERNIX_MODEL_NAME(decompression, PERNIX_MODEL_ISA_TOKEN, width)},

constexpr std::array kernels = {
    PERNIX_FOR_EACH_POSTER_WIDTH(PERNIX_KERNEL_ENTRY)
};

extern "C" __attribute__((noinline)) int control_kernel(
    const void* input, const float, void* output) {
    asm volatile("" : : "r"(input), "r"(output) : "memory");
    return 0;
}

int main(const int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "usage: probe OPERATION WIDTH ITERATIONS [--control]\n";
        return 2;
    }
    const std::string_view operation(argv[1]);
    const int width = std::atoi(argv[2]);
    const std::uint64_t iterations = std::strtoull(argv[3], nullptr, 10);
    const bool control = argc == 5 && std::string_view(argv[4]) == "--control";
    if ((operation != "compression" && operation != "decompression") ||
        width < 2 || width > 24 || iterations == 0) {
        std::cerr << "invalid probe arguments\n";
        return 2;
    }

    const auto entry = kernels[static_cast<std::size_t>(width - 2)];
    Kernel kernel = control ? control_kernel
                            : operation == "compression" ? entry.compression
                                                           : entry.decompression;
    alignas(64) std::array<std::byte, 2048> input{};
    alignas(64) std::array<std::byte, 2048> output{};
    std::uint64_t status = 0;
    for (std::uint64_t iteration = 0; iteration < iterations; ++iteration) {
        status += static_cast<unsigned>(kernel(input.data(), 1.0F, output.data()));
    }
    std::cout << (status + std::to_integer<unsigned char>(output[0])) << '\n';
    return 0;
}

