#include <benchmark/benchmark.h>
#include <libxs_pack.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(PERNIX_HFX_AVX512)
#include <pernix/x86/avx512vbmi/avx512vbmi_compression.h>
#include <pernix/x86/avx512vbmi/avx512vbmi_decompression.h>
#elif defined(PERNIX_HFX_AVX2)
#include <pernix/x86/avx2/avx2_compression.h>
#include <pernix/x86/avx2/avx2_decompression.h>
#else
#include <pernix/fallback/scalar_compression.h>
#include <pernix/fallback/scalar_decompression.h>
#endif

extern "C" {
std::size_t cp2k_hfx_pack_fixed(const std::int64_t* values, std::size_t ndata, int nbits,
                               std::int64_t* packed);
std::size_t cp2k_hfx_unpack_fixed(const std::int64_t* packed, std::size_t ndata, int nbits,
                                 std::int64_t* values);
}

namespace {
constexpr std::size_t cache_values = 1024;
constexpr int group_size = 8;
#if defined(PERNIX_HFX_AVX512)
constexpr const char* pernix_implementation = "pernix_avx512vbmi";
#elif defined(PERNIX_HFX_AVX2)
constexpr const char* pernix_implementation = "pernix_avx2";
#elif defined(PERNIX_HFX_SCALAR)
constexpr const char* pernix_implementation = "pernix_scalar";
#else
constexpr const char* pernix_implementation = "pernix_fallback";
#endif

struct Fixture {
    std::string name;
    std::vector<std::size_t> dimensions;
    std::vector<double> values;
};

std::uint64_t splitmix64(std::uint64_t& state) {
    std::uint64_t value = (state += 0x9E3779B97F4A7C15ULL);
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

Fixture make_synthetic_fixture() {
    constexpr std::vector<std::size_t>::size_type x = 64;
    constexpr std::vector<std::size_t>::size_type y = 32;
    constexpr std::vector<std::size_t>::size_type z = 16;
    Fixture result{"synthetic_hfx_sparse_tail", {x, y, z}, std::vector<double>(x * y * z)};
    std::uint64_t state = 0x4350324B48465831ULL;
    for (double& value : result.values) {
        const auto selector = splitmix64(state) % 100;
        if (selector < 70) {
            value = 0.0;
        } else if (selector < 96) {
            value = static_cast<double>(splitmix64(state) % 63) - 31.0;
        } else {
            value = static_cast<double>(splitmix64(state) % (1ULL << 24)) - (1ULL << 23);
        }
    }
    return result;
}

const Fixture& fixture() {
    static const Fixture loaded = make_synthetic_fixture();
    return loaded;
}

template <int Width>
std::vector<std::int64_t> quantize_fixture(const double scale) {
    const auto& source = fixture().values;
    std::vector<std::int64_t> values(source.size());
    const std::int64_t range = std::int64_t{1} << Width;
    const std::int64_t minimum = Width == 1 ? 0 : -(range / 2);
    const std::int64_t maximum = Width == 1 ? 1 : range / 2 - 1;
    for (std::size_t i = 0; i < source.size(); ++i) {
        const double scaled = source[i] * scale;
        values[i] = std::isnan(scaled)
                        ? 0
                        : std::llrint(std::clamp(scaled, static_cast<double>(minimum),
                                                static_cast<double>(maximum)));
    }
    return values;
}

template <int Width>
constexpr std::size_t pernix_values_per_block() {
    return 512 / Width;
}

template <int Width>
void pernix_pack(const std::vector<std::int64_t>& values, std::vector<std::uint64_t>& packed) {
    constexpr std::size_t per_block = pernix_values_per_block<Width>();
    const auto blocks = (values.size() + per_block - 1) / per_block;
    static thread_local std::vector<float> input;
    input.resize(blocks * per_block);
    std::fill(input.begin(), input.end(), 0.0F);
    for (std::size_t i = 0; i < values.size(); ++i) input[i] = static_cast<float>(values[i]);
    packed.resize(blocks * 8);
#if defined(PERNIX_HFX_AVX512)
    const int status = pernix::mm512_compress_blocks_avx512vbmi<Width, 64>(
        input.data(), 1.0F, reinterpret_cast<std::uint8_t*>(packed.data()), static_cast<std::uint32_t>(blocks));
#elif defined(PERNIX_HFX_AVX2)
    const int status = pernix::mm256_compress_blocks_avx2<Width, 64>(
        input.data(), 1.0F, reinterpret_cast<std::uint8_t*>(packed.data()), static_cast<std::uint32_t>(blocks));
#else
    const int status = pernix::compress_blocks_fallback<Width, 64>(
        input.data(), 1.0F, reinterpret_cast<std::uint8_t*>(packed.data()), static_cast<std::uint32_t>(blocks));
#endif
    if (status != 0) throw std::runtime_error("Pernix HFX compression failed");
}

template <int Width>
void pernix_unpack(const std::uint64_t* packed, const std::size_t words,
                   std::vector<std::int64_t>& values) {
    constexpr std::size_t per_block = pernix_values_per_block<Width>();
    const auto blocks = words / 8;
    static thread_local std::vector<float> output;
    output.resize(blocks * per_block);
#if defined(PERNIX_HFX_AVX512)
    const int status = pernix::mm512_decompress_blocks_avx512vbmi<Width, true, 64>(
        reinterpret_cast<const std::uint8_t*>(packed), 1.0F, output.data(), static_cast<std::uint32_t>(blocks));
#elif defined(PERNIX_HFX_AVX2)
    const int status = pernix::mm256_decompress_blocks_avx2<Width, true, 64>(
        reinterpret_cast<const std::uint8_t*>(packed), 1.0F, output.data(), static_cast<std::uint32_t>(blocks));
#else
    const int status = pernix::decompress_blocks_fallback<Width, true, 64>(
        reinterpret_cast<const std::uint8_t*>(packed), 1.0F, output.data(), static_cast<std::uint32_t>(blocks));
#endif
    if (status != 0) throw std::runtime_error("Pernix HFX decompression failed");
    values.resize(cache_values);
    for (std::size_t i = 0; i < cache_values; ++i) values[i] = static_cast<std::int64_t>(output[i]);
}

class HfxContainer {
public:
    explicit HfxContainer(const std::size_t maximum_words)
        : maximum_nodes_(maximum_words / cache_values + 1) {
        nodes_.reserve(maximum_nodes_ + 1);
    }

    void reset() {
        nodes_.clear();
        write_cursor_ = 0;
        read_cursor_ = 0;
        words_written_ = 0;
    }

    void reset_read() { read_cursor_ = 0; }

    void put(const std::uint64_t* words, std::size_t count) {
        while (count > 0) {
            const std::size_t node = write_cursor_ / cache_values;
            if (node >= maximum_nodes_) throw std::runtime_error("HFX container capacity exceeded");
            if (node == nodes_.size()) nodes_.emplace_back(new std::uint64_t[cache_values]);
            const std::size_t offset = write_cursor_ % cache_values;
            const std::size_t chunk = std::min(count, cache_values - offset);
            std::copy_n(words, chunk, nodes_[node].get() + offset);
            words += chunk;
            count -= chunk;
            write_cursor_ += chunk;
            words_written_ += chunk;
            const std::size_t next_node = write_cursor_ / cache_values;
            if (write_cursor_ % cache_values == 0 && next_node == nodes_.size() &&
                nodes_.size() < maximum_nodes_) {
                nodes_.emplace_back(new std::uint64_t[cache_values]);
            }
        }
    }

    void put_word(const std::uint64_t word) { put(&word, 1); }

    void get(std::uint64_t* destination, std::size_t count) {
        if (count > words_written_ - read_cursor_) throw std::runtime_error("Truncated HFX container");
        while (count > 0) {
            const std::size_t node = read_cursor_ / cache_values;
            const std::size_t offset = read_cursor_ % cache_values;
            const std::size_t chunk = std::min(count, cache_values - offset);
            std::copy_n(nodes_[node].get() + offset, chunk, destination);
            destination += chunk;
            count -= chunk;
            read_cursor_ += chunk;
        }
    }

    std::size_t words_written() const { return words_written_; }
    std::size_t allocated_bytes() const { return nodes_.size() * cache_values * sizeof(std::uint64_t); }

private:
    std::size_t maximum_nodes_ = 0;
    std::vector<std::unique_ptr<std::uint64_t[]>> nodes_;
    std::size_t write_cursor_ = 0;
    std::size_t read_cursor_ = 0;
    std::size_t words_written_ = 0;
};

template <int Width, bool Libxs, bool Pernix>
std::size_t scratch_bound() {
    if constexpr (Pernix) {
        constexpr auto per_block = pernix_values_per_block<Width>();
        return ((cache_values + per_block - 1) / per_block) * 8;
    } else if constexpr (Libxs) {
        return libxs_vpack_bound(cache_values, Width, group_size);
    } else {
        return (cache_values * Width + 63) / 64;
    }
}

template <int Width, bool Libxs, bool Pernix>
void pack_records(const std::vector<std::int64_t>& values, HfxContainer& container,
                  std::vector<std::int64_t>& cache, std::vector<std::uint64_t>& scratch) {
    container.reset();
    for (std::size_t offset = 0; offset < values.size(); offset += cache_values) {
        std::fill(cache.begin(), cache.end(), 0);
        const auto count = std::min(cache_values, values.size() - offset);
        std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(offset), count, cache.begin());
        if constexpr (Pernix) {
            pernix_pack<Width>(cache, scratch);
            container.put(scratch.data(), scratch.size());
        } else if constexpr (Libxs) {
            scratch.resize(libxs_vpack_bound(cache_values, Width, group_size));
            const auto written = libxs_vpack(scratch.data(), cache.data(), LIBXS_DATATYPE_I64,
                                              cache_values, Width, group_size);
            if (written == 0 || written > scratch.size()) throw std::runtime_error("LIBXS HFX compression failed");
            scratch.resize(written);
            container.put_word(written);
            container.put(scratch.data(), written);
        } else {
            scratch.resize(scratch_bound<Width, Libxs, Pernix>());
            const auto written = cp2k_hfx_pack_fixed(cache.data(), cache_values, Width,
                                                      reinterpret_cast<std::int64_t*>(scratch.data()));
            if (written != scratch_bound<Width, Libxs, Pernix>()) throw std::runtime_error("CP2K HFX compression failed");
            container.put(scratch.data(), written);
        }
    }
}

template <int Width, bool Libxs, bool Pernix>
void unpack_records(HfxContainer& container, const std::size_t input_count,
                    std::vector<std::int64_t>& decoded,
                    std::vector<std::uint64_t>& scratch, std::vector<std::int64_t>& cache) {
    container.reset_read();
    decoded.clear();
    for (std::size_t offset = 0; offset < input_count; offset += cache_values) {
        std::size_t words = 0;
        if constexpr (Pernix) {
            words = scratch_bound<Width, Libxs, Pernix>();
            scratch.resize(words);
            container.get(scratch.data(), words);
            pernix_unpack<Width>(scratch.data(), words, cache);
        } else if constexpr (Libxs) {
            std::uint64_t record_words = 0;
            container.get(&record_words, 1);
            words = static_cast<std::size_t>(record_words);
            scratch.resize(words);
            container.get(scratch.data(), words);
            cache.resize(cache_values);
            const auto consumed = libxs_vunpack(cache.data(), scratch.data(), words, LIBXS_DATATYPE_I64, cache_values);
            if (consumed != words) throw std::runtime_error("LIBXS HFX record did not unpack completely");
        } else {
            words = scratch_bound<Width, Libxs, Pernix>();
            scratch.resize(words);
            container.get(scratch.data(), words);
            cache.resize(cache_values);
            const auto consumed = cp2k_hfx_unpack_fixed(reinterpret_cast<const std::int64_t*>(scratch.data()),
                                                         cache_values, Width, cache.data());
            if (consumed != words) throw std::runtime_error("CP2K HFX record did not unpack completely");
        }
        const auto remaining = input_count - decoded.size();
        const auto count = std::min(cache_values, remaining);
        decoded.insert(decoded.end(), cache.begin(), cache.begin() + static_cast<std::ptrdiff_t>(count));
    }
}

template <int Width, bool Libxs, bool Pernix, bool Compress>
void run(benchmark::State& state, const double scale, const char* implementation) {
    const auto input = quantize_fixture<Width>(scale);
    const auto record_count = (input.size() + cache_values - 1) / cache_values;
    const auto input_bytes = cache_values * record_count * sizeof(std::int64_t);
    const auto maximum_words = record_count *
        (scratch_bound<Width, Libxs, Pernix>() + (Libxs ? 1 : 0));
    HfxContainer container(maximum_words);
    std::vector<std::int64_t> cache(cache_values);
    std::vector<std::uint64_t> scratch(scratch_bound<Width, Libxs, Pernix>());
    std::vector<std::int64_t> decoded;
    decoded.reserve(record_count * cache_values);
    pack_records<Width, Libxs, Pernix>(input, container, cache, scratch);
    const auto packed_bytes = container.words_written() * sizeof(std::uint64_t);
    const auto container_bytes = container.allocated_bytes();
    unpack_records<Width, Libxs, Pernix>(container, input.size(), decoded, scratch, cache);
    if (!std::equal(input.begin(), input.end(), decoded.begin())) {
        state.SkipWithError("HFX codec record round-trip verification failed");
        return;
    }
    if constexpr (Compress) {
        for (auto _ : state) {
            pack_records<Width, Libxs, Pernix>(input, container, cache, scratch);
            benchmark::DoNotOptimize(container.words_written());
            benchmark::ClobberMemory();
        }
    } else {
        for (auto _ : state) {
            unpack_records<Width, Libxs, Pernix>(container, input.size(), decoded, scratch, cache);
            benchmark::DoNotOptimize(decoded.data());
            benchmark::ClobberMemory();
        }
    }
    const auto iterations = static_cast<std::int64_t>(state.iterations());
    state.SetItemsProcessed(iterations * static_cast<std::int64_t>(input.size()));
    state.SetBytesProcessed(iterations * static_cast<std::int64_t>(input_bytes + packed_bytes));
    state.counters["packed_bytes"] = static_cast<double>(packed_bytes);
    state.counters["container_bytes"] = static_cast<double>(container_bytes);
    state.counters["bits_per_value"] = 8.0 * static_cast<double>(packed_bytes) /
                                       static_cast<double>(input_bytes / sizeof(std::int64_t));
    state.counters["compression_ratio"] = static_cast<double>(packed_bytes) / static_cast<double>(input_bytes);
    state.counters["fixture_values"] = static_cast<double>(input.size());
    state.counters["fixture_ndims"] = static_cast<double>(fixture().dimensions.size());
    for (std::size_t axis = 0; axis < fixture().dimensions.size(); ++axis) {
        state.counters["dimension_" + std::to_string(axis)] = static_cast<double>(fixture().dimensions[axis]);
    }
    state.counters["scale"] = scale;
    state.counters["group_size"] = Libxs ? group_size : 0;
    state.SetLabel(std::string(implementation) + ":" + fixture().name);
}

template <int Width, bool Libxs, bool Pernix>
void register_codec(const char* codec, const char* direction, const double scale) {
    const std::string stem = "BM_hfx_" + std::string(direction) + "_" + codec + "_w" + std::to_string(Width);
    const auto suffix = fixture().name;
    benchmark::RegisterBenchmark((stem + "_" + suffix).c_str(), [scale](benchmark::State& state) {
        run<Width, Libxs, Pernix, true>(state, scale, "compression");
    })->Arg(static_cast<std::int64_t>(fixture().values.size()));
    const std::string decompress_stem = "BM_hfx_decompress_" + std::string(codec) + "_w" + std::to_string(Width);
    benchmark::RegisterBenchmark((decompress_stem + "_" + suffix).c_str(), [scale](benchmark::State& state) {
        run<Width, Libxs, Pernix, false>(state, scale, "decompression");
    })->Arg(static_cast<std::int64_t>(fixture().values.size()));
}

template <int Width>
void register_width(const double scale) {
    register_codec<Width, false, false>("cp2k_original", "compress", scale);
    register_codec<Width, true, false>("libxs_group8", "compress", scale);
    register_codec<Width, false, true>(pernix_implementation, "compress", scale);
}

[[maybe_unused]] const bool registered = [] {
    const char* scale_text = std::getenv("PERNIX_HFX_SCALE");
    if (scale_text == nullptr || *scale_text == '\0') {
        throw std::runtime_error("PERNIX_HFX_SCALE must be a finite positive quantization scale");
    }
    char* end = nullptr;
    const double scale = std::strtod(scale_text, &end);
    if (end == scale_text || *end != '\0' || !std::isfinite(scale) || scale <= 0.0) {
        throw std::runtime_error("PERNIX_HFX_SCALE must be a finite positive quantization scale");
    }
    (void)fixture();
#define REGISTER(W) register_width<W>(scale)
    REGISTER(2); REGISTER(3); REGISTER(4); REGISTER(5); REGISTER(6);
    REGISTER(7); REGISTER(8); REGISTER(9); REGISTER(10); REGISTER(11); REGISTER(12);
    REGISTER(13); REGISTER(14); REGISTER(15); REGISTER(16); REGISTER(17); REGISTER(18);
    REGISTER(19); REGISTER(20); REGISTER(21); REGISTER(22); REGISTER(23); REGISTER(24);
#undef REGISTER
    return true;
}();
}  // namespace
