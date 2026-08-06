#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

extern "C" {
void cp2k_compress_block_f32(const float* input, float scale, std::int64_t* packed, int bit_width);
void cp2k_decompress_block_f32(const std::int64_t* packed, float scale, float* output, int bit_width);
void cp2k_compress_block_f64(const double* input, double scale, std::int64_t* packed, int bit_width);
void cp2k_decompress_block_f64(const std::int64_t* packed, double scale, double* output, int bit_width);
}

namespace {
constexpr int block_bits = 512;

template <typename ValueT>
std::int64_t quantize(const ValueT input, const ValueT scale, const int bit_width) {
    if (std::isnan(input * scale)) {
        return 0;
    }
    const std::int64_t range_size = std::int64_t{1} << bit_width;
    const std::int64_t minimum    = bit_width == 1 ? 0 : -(range_size / 2);
    const std::int64_t maximum    = bit_width == 1 ? 1 : range_size / 2 - 1;
    return std::llrint(std::clamp(input * scale, static_cast<ValueT>(minimum), static_cast<ValueT>(maximum)));
}

template <typename ValueT>
void compress(const ValueT* input, ValueT scale, std::int64_t* packed, int bit_width) {
    if constexpr (std::is_same_v<ValueT, float>) {
        cp2k_compress_block_f32(input, scale, packed, bit_width);
    } else {
        cp2k_compress_block_f64(input, scale, packed, bit_width);
    }
}

template <typename ValueT>
void decompress(const std::int64_t* packed, ValueT scale, ValueT* output, int bit_width) {
    if constexpr (std::is_same_v<ValueT, float>) {
        cp2k_decompress_block_f32(packed, scale, output, bit_width);
    } else {
        cp2k_decompress_block_f64(packed, scale, output, bit_width);
    }
}

template <typename ValueT>
bool test_precision() {
    alignas(64) std::array<ValueT, block_bits> input{};
    alignas(64) std::array<ValueT, block_bits> output{};
    alignas(64) std::array<std::int64_t, 8> packed{};

    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<ValueT>(static_cast<int>(index % 19) - 9) / static_cast<ValueT>(3);
    }
    input[0] = std::numeric_limits<ValueT>::quiet_NaN();
    input[1] = std::numeric_limits<ValueT>::infinity();
    input[2] = -std::numeric_limits<ValueT>::infinity();

    constexpr ValueT compression_scale   = static_cast<ValueT>(2.25);
    constexpr ValueT decompression_scale = static_cast<ValueT>(0.125);
    constexpr ValueT sentinel            = static_cast<ValueT>(-1234567);

    for (int bit_width = 1; bit_width <= 24; ++bit_width) {
        const int elements = block_bits / bit_width;
        output.fill(sentinel);
        packed.fill(0);

        compress(input.data(), compression_scale, packed.data(), bit_width);
        decompress(packed.data(), decompression_scale, output.data(), bit_width);

        for (int index = 0; index < elements; ++index) {
            const ValueT expected = static_cast<ValueT>(quantize(input[index], compression_scale, bit_width)) * decompression_scale;
            if (output[index] != expected) {
                std::cerr << (std::is_same_v<ValueT, float> ? "f32" : "f64") << " width " << bit_width << ", element " << index
                          << ": expected " << expected << ", got " << output[index] << '\n';
                return false;
            }
        }
        if (elements < block_bits && output[elements] != sentinel) {
            std::cerr << (std::is_same_v<ValueT, float> ? "f32" : "f64") << " width " << bit_width << " wrote beyond its logical output\n";
            return false;
        }
    }

    return true;
}
}  // namespace

int main() {
    return test_precision<float>() && test_precision<double>() ? 0 : 1;
}
