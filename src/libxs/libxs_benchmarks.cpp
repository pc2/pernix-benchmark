#include <benchmark/benchmark.h>
#include <libxs_pack.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

enum class ValueProfile : std::uint64_t {
    broad = 0,
    sparse = 1,
};

template <typename T>
constexpr libxs_data_t data_type() {
    if constexpr (std::is_same_v<T, std::int8_t>) return LIBXS_DATATYPE_I8;
    if constexpr (std::is_same_v<T, std::uint8_t>) return LIBXS_DATATYPE_U8;
    if constexpr (std::is_same_v<T, std::int16_t>) return LIBXS_DATATYPE_I16;
    if constexpr (std::is_same_v<T, std::uint16_t>) return LIBXS_DATATYPE_U16;
    if constexpr (std::is_same_v<T, std::int32_t>) return LIBXS_DATATYPE_I32;
    if constexpr (std::is_same_v<T, std::uint32_t>) return LIBXS_DATATYPE_U32;
    if constexpr (std::is_same_v<T, std::int64_t>) return LIBXS_DATATYPE_I64;
    return LIBXS_DATATYPE_U64;
}

template <typename T>
void fill_values(std::vector<T>& values, const ValueProfile profile) {
    constexpr bool is_signed = std::is_signed_v<T>;
    constexpr unsigned bit_count = sizeof(T) * 8;
    constexpr std::uint64_t value_mask = [] {
        if constexpr (bit_count == 64) {
            return ~std::uint64_t{0};
        } else {
            return (std::uint64_t{1} << bit_count) - 1;
        }
    }();
    constexpr std::uint64_t sign_bit = std::uint64_t{1} << (bit_count - 1);
    const std::uint64_t seed = 0x9E3779B97F4A7C15ULL ^ values.size() ^
                               (sizeof(T) << 24) ^ (is_signed ? 0x1000ULL : 0ULL) ^
                               static_cast<std::uint64_t>(profile);
    std::mt19937_64 generator(seed);

    if constexpr (is_signed) {
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (profile == ValueProfile::sparse && index % 16 < 8) {
                values[index] = 0;
            } else if (profile == ValueProfile::sparse) {
                values[index] = static_cast<T>(static_cast<std::int64_t>(generator() % 255) - 127);
            } else {
                const std::uint64_t bits = generator() & value_mask;
                if ((bits & sign_bit) == 0) {
                    values[index] = static_cast<T>(bits);
                } else {
                    const auto lower_magnitude = static_cast<std::int64_t>((~bits) & (sign_bit - 1));
                    values[index] = static_cast<T>(-1 - lower_magnitude);
                }
            }
        }
    } else {
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (profile == ValueProfile::sparse && index % 16 < 8) {
                values[index] = 0;
            } else {
                const std::uint64_t value = profile == ValueProfile::broad
                                                ? generator() & value_mask
                                                : generator() & 0xFF;
                values[index] = static_cast<T>(value);
            }
        }
    }
}

template <typename T>
bool pack_input(const std::vector<T>& values, std::vector<std::uint64_t>& packed,
                const std::size_t bound, const int group, std::size_t& words) {
    if (bound == 0) return false;
    packed.resize(bound);
    words = libxs_vpack(packed.data(), values.data(), data_type<T>(), values.size(),
                        static_cast<int>(sizeof(T) * 8), group);
    return words != 0 && words <= bound;
}

template <typename T>
void BM_libxs_compress(benchmark::State& state, const int group, const ValueProfile profile) {
    const auto count = static_cast<std::size_t>(state.range(0));
    std::vector<T> values(count);
    fill_values(values, profile);

    const int maxbits = static_cast<int>(sizeof(T) * 8);
    const std::size_t bound = libxs_vpack_bound(count, maxbits, group);
    std::vector<std::uint64_t> packed;
    std::size_t words = 0;
    if (!pack_input(values, packed, bound, group, words)) {
        state.SkipWithError("libxs_vpack returned an invalid word count");
        return;
    }

    const std::size_t packed_bytes = words * sizeof(std::uint64_t);
    const std::size_t input_bytes = count * sizeof(T);
    std::vector<T> decoded(count);
    const std::size_t consumed =
        libxs_vunpack(decoded.data(), packed.data(), words, data_type<T>(), count);
    if (consumed != words || !std::equal(values.begin(), values.end(), decoded.begin())) {
        state.SkipWithError("libxs_vpack round-trip verification failed");
        return;
    }

    for (auto _ : state) {
        std::size_t written =
            libxs_vpack(packed.data(), values.data(), data_type<T>(), count, maxbits, group);
        benchmark::DoNotOptimize(written);
        benchmark::DoNotOptimize(packed.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(input_bytes + packed_bytes));
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(count));
    state.counters["packed_bytes"] = static_cast<double>(packed_bytes);
    state.counters["bits_per_value"] = 8.0 * static_cast<double>(packed_bytes) / static_cast<double>(count);
    state.counters["compression_ratio"] = static_cast<double>(packed_bytes) / static_cast<double>(input_bytes);
}

template <typename T>
void BM_libxs_decompress(benchmark::State& state, const int group, const ValueProfile profile) {
    const auto count = static_cast<std::size_t>(state.range(0));
    std::vector<T> values(count);
    fill_values(values, profile);

    const int maxbits = static_cast<int>(sizeof(T) * 8);
    const std::size_t bound = libxs_vpack_bound(count, maxbits, group);
    std::vector<std::uint64_t> packed;
    std::size_t words = 0;
    if (!pack_input(values, packed, bound, group, words)) {
        state.SkipWithError("libxs_vpack setup returned an invalid word count");
        return;
    }

    const std::size_t packed_bytes = words * sizeof(std::uint64_t);
    const std::size_t output_bytes = count * sizeof(T);
    std::vector<T> decoded(count);
    const std::size_t consumed =
        libxs_vunpack(decoded.data(), packed.data(), words, data_type<T>(), count);
    if (consumed != words || !std::equal(values.begin(), values.end(), decoded.begin())) {
        state.SkipWithError("libxs_vunpack round-trip verification failed");
        return;
    }

    for (auto _ : state) {
        std::size_t read =
            libxs_vunpack(decoded.data(), packed.data(), words, data_type<T>(), count);
        benchmark::DoNotOptimize(read);
        benchmark::DoNotOptimize(decoded.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(packed_bytes + output_bytes));
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(count));
    state.counters["packed_bytes"] = static_cast<double>(packed_bytes);
    state.counters["bits_per_value"] = 8.0 * static_cast<double>(packed_bytes) / static_cast<double>(count);
    state.counters["compression_ratio"] = static_cast<double>(packed_bytes) / static_cast<double>(output_bytes);
}

template <typename T>
void register_value_type(const char* type_name) {
    for (const auto [profile_name, profile] : {
             std::pair{"broad", ValueProfile::broad},
             std::pair{"sparse", ValueProfile::sparse},
         }) {
        for (int group = 1; group <= 32; ++group) {
            const std::string suffix = std::string(type_name) + "_" + profile_name + "_g" +
                                       std::to_string(group);
            benchmark::RegisterBenchmark(("BM_libxs_compress_" + suffix).c_str(),
                                         [group, profile](benchmark::State& state) {
                                             BM_libxs_compress<T>(state, group, profile);
                                         })
                ->Arg(1 << 12)
                ->Arg((1 << 18) + 3);
            benchmark::RegisterBenchmark(("BM_libxs_decompress_" + suffix).c_str(),
                                         [group, profile](benchmark::State& state) {
                                             BM_libxs_decompress<T>(state, group, profile);
                                         })
                ->Arg(1 << 12)
                ->Arg((1 << 18) + 3);
        }
    }
}

}  // namespace

[[maybe_unused]] static const bool registered_libxs_benchmarks = [] {
    register_value_type<std::int8_t>("i8");
    register_value_type<std::uint8_t>("u8");
    register_value_type<std::int16_t>("i16");
    register_value_type<std::uint16_t>("u16");
    register_value_type<std::int32_t>("i32");
    register_value_type<std::uint32_t>("u32");
    register_value_type<std::int64_t>("i64");
    register_value_type<std::uint64_t>("u64");
    return true;
}();
