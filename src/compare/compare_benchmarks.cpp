#include <benchmark/benchmark.h>
#include <libxs_pack.h>

#if defined(PERNIX_COMPARE_AVX2)
#include <pernix/x86/avx2/avx2_compression.h>
#include <pernix/x86/avx2/avx2_decompression.h>
#elif defined(PERNIX_COMPARE_AVX512VBMI)
#include <pernix/x86/avx512vbmi/avx512vbmi_compression.h>
#include <pernix/x86/avx512vbmi/avx512vbmi_decompression.h>
#else
#include <pernix/fallback/scalar_compression.h>
#include <pernix/fallback/scalar_decompression.h>
#endif

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {
enum class Profile { dense, sparse };

template <int Width>
void fill_values(std::vector<float>& floats, std::vector<std::int32_t>& integers, Profile profile) {
    const std::uint64_t seed = 0xD1B54A32D192ED03ULL ^ floats.size() ^
                               (static_cast<std::uint64_t>(Width) << 32) ^
                               static_cast<std::uint64_t>(profile == Profile::sparse);
    std::mt19937_64 rng(seed);
    // The signed libxs I32 and Pernix decoders share only zero at one bit.
    const std::int32_t lo = Width == 1 ? 0 : -(std::int32_t{1} << (Width - 1));
    const std::int32_t hi = Width == 1 ? 0 : (std::int32_t{1} << (Width - 1)) - 1;
    for (std::size_t i = 0; i < floats.size(); ++i) {
        std::int32_t value;
        if (profile == Profile::sparse && i % 16 < 12) {
            value = 0;
        } else if (profile == Profile::sparse) {
            const auto small = static_cast<std::int32_t>(rng() % 15) - 7;
            value = std::clamp(small, lo, hi);
        } else {
            const auto span = static_cast<std::uint64_t>(static_cast<std::int64_t>(hi) - lo + 1);
            value = lo + static_cast<std::int32_t>(rng() % span);
        }
        integers[i] = value;
        floats[i] = static_cast<float>(value);
    }
}

template <int Width>
int pernix_compress(const float* input, std::uint8_t* output, std::uint32_t blocks) {
#if defined(PERNIX_COMPARE_AVX2)
    return pernix::mm256_compress_blocks_avx2<Width, 64>(input, 1.0f, output, blocks);
#elif defined(PERNIX_COMPARE_AVX512VBMI)
    return pernix::mm512_compress_blocks_avx512vbmi<Width, 64>(input, 1.0f, output, blocks);
#else
    return pernix::compress_blocks_fallback<Width, 64>(input, 1.0f, output, blocks);
#endif
}

template <int Width>
int pernix_decompress(const std::uint8_t* input, float* output, std::uint32_t blocks) {
#if defined(PERNIX_COMPARE_AVX2)
    return pernix::mm256_decompress_blocks_avx2<Width, true, 64>(input, 1.0f, output, blocks);
#elif defined(PERNIX_COMPARE_AVX512VBMI)
    return pernix::mm512_decompress_blocks_avx512vbmi<Width, true, 64>(input, 1.0f, output, blocks);
#else
    return pernix::decompress_blocks_fallback<Width, true, 64>(input, 1.0f, output, blocks);
#endif
}

template <int Width, Profile DataProfile, bool Compress, bool UseLibxs>
void run(benchmark::State& state) {
    const auto count = static_cast<std::size_t>(state.range(0));
    constexpr std::size_t per_block = 512 / Width;
    const auto blocks = static_cast<std::uint32_t>(count / per_block);
    const auto elements = blocks * per_block;
    std::vector<float> floats(elements);
    std::vector<std::int32_t> integers(elements);
    fill_values<Width>(floats, integers, DataProfile);
    std::vector<float> decoded_floats(elements);
    std::vector<std::int32_t> decoded_integers(elements);
    std::vector<std::uint8_t> pernix_stream(blocks * 64);
    const auto libxs_bound = libxs_vpack_bound(elements, Width, 8);
    std::vector<std::uint64_t> libxs_stream(libxs_bound);
    std::size_t libxs_words = libxs_vpack(libxs_stream.data(), integers.data(), LIBXS_DATATYPE_I32,
                                           elements, Width, 8);
    if (libxs_words == 0) {
        state.SkipWithError("libxs_vpack setup failed");
        return;
    }
    if (pernix_compress<Width>(floats.data(), pernix_stream.data(), blocks) != 0) {
        state.SkipWithError("Pernix compression setup failed"); return;
    }
    if constexpr (!Compress && UseLibxs) {
        const auto consumed = libxs_vunpack(decoded_integers.data(), libxs_stream.data(), libxs_words,
                                            LIBXS_DATATYPE_I32, elements);
        if (consumed != libxs_words || decoded_integers != integers) {
            state.SkipWithError("libxs round-trip verification failed"); return;
        }
    } else if constexpr (!Compress) {
        if (pernix_decompress<Width>(pernix_stream.data(), decoded_floats.data(), blocks) != 0) {
            state.SkipWithError("Pernix decompression setup failed"); return;
        }
        if (!std::equal(floats.begin(), floats.end(), decoded_floats.begin())) {
            state.SkipWithError("Pernix round-trip verification failed"); return;
        }
    }

    const std::size_t stream_bytes = UseLibxs ? libxs_words * sizeof(std::uint64_t) : blocks * 64;
    const std::size_t input_bytes = elements * sizeof(float);
    for (auto _ : state) {
        if constexpr (UseLibxs && Compress) {
            libxs_words = libxs_vpack(libxs_stream.data(), integers.data(), LIBXS_DATATYPE_I32, elements, Width, 8);
            benchmark::DoNotOptimize(libxs_words);
            benchmark::DoNotOptimize(libxs_stream.data());
        } else if constexpr (UseLibxs && !Compress) {
            libxs_words = libxs_vunpack(decoded_integers.data(), libxs_stream.data(), libxs_words,
                                        LIBXS_DATATYPE_I32, elements);
            benchmark::DoNotOptimize(libxs_words);
            benchmark::DoNotOptimize(decoded_integers.data());
        } else if constexpr (Compress) {
            auto written = pernix_compress<Width>(floats.data(), pernix_stream.data(), blocks);
            benchmark::DoNotOptimize(written);
            benchmark::DoNotOptimize(pernix_stream.data());
        } else {
            auto read = pernix_decompress<Width>(pernix_stream.data(), decoded_floats.data(), blocks);
            benchmark::DoNotOptimize(read);
            benchmark::DoNotOptimize(decoded_floats.data());
        }
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * elements);
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * (input_bytes + stream_bytes));
    state.counters["elements"] = static_cast<double>(elements);
    state.counters["stream_bytes"] = static_cast<double>(stream_bytes);
    state.counters["bits_per_value"] = 8.0 * static_cast<double>(stream_bytes) / elements;
}

template <int Width, Profile P>
void register_cases(const char* profile) {
    const std::string suffix = std::string(profile) + "_w" + std::to_string(Width);
    for (const auto& [name, compress, libxs] : {
             std::tuple{"pernix_compress_", true, false}, {"pernix_decompress_", false, false},
             {"libxs_compress_", true, true}, {"libxs_decompress_", false, true}}) {
        benchmark::RegisterBenchmark((std::string("BM_compare_") + name + suffix).c_str(),
            [=](benchmark::State& state) {
                if (libxs && compress) run<Width, P, true, true>(state);
                else if (libxs) run<Width, P, false, true>(state);
                else if (compress) run<Width, P, true, false>(state);
                else run<Width, P, false, false>(state);
            })->Arg(1 << 12)->Arg(1 << 18);
    }
}

template <Profile P>
void register_widths(const char* profile) {
#define REGISTER(W) register_cases<W, P>(profile)
    REGISTER(1); REGISTER(2); REGISTER(3); REGISTER(4); REGISTER(5); REGISTER(6);
    REGISTER(7); REGISTER(8); REGISTER(9); REGISTER(10); REGISTER(11); REGISTER(12);
    REGISTER(13); REGISTER(14); REGISTER(15); REGISTER(16); REGISTER(17); REGISTER(18);
    REGISTER(19); REGISTER(20); REGISTER(21); REGISTER(22); REGISTER(23); REGISTER(24);
#undef REGISTER
}
}  // namespace

[[maybe_unused]] static const bool registered = [] {
    register_widths<Profile::dense>("dense");
    register_widths<Profile::sparse>("sparse");
    return true;
}();
