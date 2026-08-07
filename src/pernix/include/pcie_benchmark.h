#ifndef PERNIX_PCIE_BENCHMARK_H
#define PERNIX_PCIE_BENCHMARK_H

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace pernix_benchmark::pcie {

constexpr std::uint64_t kBlockBytes          = 64;
constexpr std::uint64_t kPosterOriginalBytes = std::uint64_t{1} << 30;
constexpr std::size_t kPipelineChunkBytes    = std::size_t{1} << 20;
constexpr std::size_t kPipelineDepth         = 2;
// Some original Pernix kernels implement partial input reads with masked vector
// loads.  The mask prevents inactive lanes from being accessed, but the active
// lane containing the final payload byte can extend a few bytes beyond the
// logical 64-byte block.  Keep one full vector of mapped storage after the last
// block; this padding is never included in a PCIe transfer or in the reported
// compressed byte count.
constexpr std::size_t kPackedGuardBytes = 64;

inline void check_cuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(const std::size_t elements) : elements_(elements) {
        if (elements_ != 0)
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&data_), elements_ * sizeof(T), cudaHostAllocDefault), "cudaHostAlloc");
    }

    ~PinnedBuffer() {
        if (data_) cudaFreeHost(data_);
    }
    PinnedBuffer(const PinnedBuffer&)            = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& other) noexcept
        : elements_(std::exchange(other.elements_, 0)), data_(std::exchange(other.data_, nullptr)) {}
    T* data() { return data_; }
    const T* data() const { return data_; }

private:
    std::size_t elements_;
    T* data_ = nullptr;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(const std::size_t bytes) { check_cuda(cudaMalloc(&data_, bytes), "cudaMalloc"); }
    ~DeviceBuffer() {
        if (data_) cudaFree(data_);
    }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* data() { return data_; }
    std::uint8_t* bytes() { return static_cast<std::uint8_t*>(data_); }

private:
    void* data_ = nullptr;
};

class Event {
public:
    Event() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~Event() {
        if (event_) cudaEventDestroy(event_);
    }
    Event(const Event&) = delete;
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

class Stream {
public:
    Stream() { check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"); }
    ~Stream() {
        if (stream_) cudaStreamDestroy(stream_);
    }
    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

template <std::uint8_t BIT_WIDTH, typename ValueT>
    requires(BIT_WIDTH >= 1 && BIT_WIDTH <= 24 && std::is_floating_point_v<ValueT>)
struct Fixture {
    static constexpr std::size_t kElementsPerBlock = 512 / BIT_WIDTH;
    const std::size_t blocks;
    const std::size_t payload_bytes;
    const std::size_t staging_bytes;
    PinnedBuffer<ValueT> input;
    std::array<PinnedBuffer<std::uint8_t>, kPipelineDepth> packed;
    PinnedBuffer<ValueT> output;
    ValueT scale;
    DeviceBuffer device;

    explicit Fixture(const std::size_t payload)
        : blocks(payload / kBlockBytes),
          payload_bytes(payload),
          staging_bytes(std::min(payload, kPipelineChunkBytes)),
          input(blocks * kElementsPerBlock),
          packed{PinnedBuffer<std::uint8_t>(staging_bytes + kPackedGuardBytes),
                 PinnedBuffer<std::uint8_t>(staging_bytes + kPackedGuardBytes)},
          output(blocks * kElementsPerBlock),
          device(blocks * kElementsPerBlock * sizeof(ValueT)) {
        for (auto& buffer : packed) {
            std::fill_n(buffer.data(), staging_bytes + kPackedGuardBytes, std::uint8_t{0});
        }
        std::mt19937 generator(0xC0FFEE);
        std::uniform_real_distribution<ValueT> input_distribution(static_cast<ValueT>(-1), static_cast<ValueT>(1));
        std::uniform_real_distribution<ValueT> scale_distribution(static_cast<ValueT>(0.0001), static_cast<ValueT>(1));
        for (std::size_t index = 0; index < blocks * kElementsPerBlock; ++index) input.data()[index] = input_distribution(generator);
        scale = scale_distribution(generator);
    }

    std::size_t useful_bytes() const { return blocks * kElementsPerBlock * sizeof(ValueT); }
    std::size_t chunks() const { return (payload_bytes + staging_bytes - 1) / staging_bytes; }
    std::size_t chunk_payload_bytes(const std::size_t chunk) const {
        return std::min(staging_bytes, payload_bytes - chunk * staging_bytes);
    }
    std::size_t chunk_block_offset(const std::size_t chunk) const { return chunk * staging_bytes / kBlockBytes; }
};

template <std::uint8_t BIT_WIDTH, typename ValueT>
constexpr std::size_t poster_payload_bytes() {
    constexpr std::size_t elements_per_block = 512 / BIT_WIDTH;
    constexpr std::size_t blocks             = kPosterOriginalBytes / (sizeof(ValueT) * elements_per_block);
    return blocks * kBlockBytes;
}

template <std::uint8_t BIT_WIDTH, typename ValueT>
double measure_uncompressed_copy(Fixture<BIT_WIDTH, ValueT>& fixture, const cudaMemcpyKind direction) {
    Event start;
    Event stop;
    double total_milliseconds             = 0;
    std::uint64_t repetitions             = 0;
    constexpr double minimum_milliseconds = 250.0;
    if (direction == cudaMemcpyHostToDevice) {
        check_cuda(cudaMemcpyAsync(fixture.device.data(), fixture.input.data(), fixture.useful_bytes(), direction),
                   "cudaMemcpyAsync uncompressed H2D warm-up");
    } else {
        check_cuda(cudaMemcpyAsync(fixture.output.data(), fixture.device.data(), fixture.useful_bytes(), direction),
                   "cudaMemcpyAsync uncompressed D2H warm-up");
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize baseline warm-up");
    do {
        check_cuda(cudaEventRecord(start.get()), "cudaEventRecord baseline start");
        if (direction == cudaMemcpyHostToDevice) {
            check_cuda(cudaMemcpyAsync(fixture.device.data(), fixture.input.data(), fixture.useful_bytes(), direction),
                       "cudaMemcpyAsync uncompressed H2D");
        } else {
            check_cuda(cudaMemcpyAsync(fixture.output.data(), fixture.device.data(), fixture.useful_bytes(), direction),
                       "cudaMemcpyAsync uncompressed D2H");
        }
        check_cuda(cudaEventRecord(stop.get()), "cudaEventRecord baseline stop");
        check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize baseline");
        float elapsed_milliseconds = 0;
        check_cuda(cudaEventElapsedTime(&elapsed_milliseconds, start.get(), stop.get()), "cudaEventElapsedTime baseline");
        total_milliseconds += elapsed_milliseconds;
        ++repetitions;
    } while (total_milliseconds < minimum_milliseconds);
    return total_milliseconds / 1000.0 / static_cast<double>(repetitions);
}

inline void publish_metrics(benchmark::State& state, const std::uint64_t iterations, const std::size_t useful_bytes,
                            const std::size_t payload_bytes, const std::size_t chunks, const double dma_milliseconds,
                            const double codec_seconds, const double uncompressed_seconds, const double sum) {
    state.SetBytesProcessed(static_cast<std::int64_t>(iterations * useful_bytes));
    state.SetItemsProcessed(static_cast<std::int64_t>(iterations));
    state.counters["pcie_payload_bytes"] = static_cast<double>(payload_bytes);
    state.counters["dma_seconds"]        = dma_milliseconds / 1000.0;
    state.counters["dma_bytes_per_second"] =
        dma_milliseconds == 0 ? 0.0 : static_cast<double>(iterations * payload_bytes) / (dma_milliseconds / 1000.0);
    state.counters["original_bytes"]                     = static_cast<double>(useful_bytes);
    state.counters["compressed_bytes"]                   = static_cast<double>(payload_bytes);
    state.counters["codec_seconds"]                      = codec_seconds / static_cast<double>(iterations);
    state.counters["uncompressed_seconds"]               = uncompressed_seconds;
    state.counters["pinned_memory"]                      = 1.0;
    state.counters["async_copy"]                         = 1.0;
    state.counters["streams"]                            = 1.0;
    state.counters["overlap"]                            = chunks > 1 ? 1.0 : 0.0;
    state.counters["pipeline_chunk_bytes"]               = static_cast<double>(kPipelineChunkBytes);
    state.counters["pipeline_depth"]                     = static_cast<double>(kPipelineDepth);
    state.counters["allocation_initialization_excluded"] = 1.0;
    state.counters["scale_inside_payload"]               = 0.0;
    state.counters["sum"]                                = sum;
}

template <std::uint8_t BIT_WIDTH, typename ValueT, typename Compressor>
void BM_pcie_h2d(benchmark::State& state) {
    const std::size_t payload_bytes = static_cast<std::size_t>(state.range(0));
    Fixture<BIT_WIDTH, ValueT> fixture(payload_bytes);
    Stream stream;
    std::array<Event, kPipelineDepth> starts;
    std::array<Event, kPipelineDepth> stops;
    std::array<bool, kPipelineDepth> pending{};
    double dma_milliseconds = 0;
    double codec_seconds    = 0;
    double sum              = 0;
    Compressor compressor;
    const double uncompressed_seconds = measure_uncompressed_copy(fixture, cudaMemcpyHostToDevice);
    for (auto _ : state) {
        for (std::size_t chunk = 0; chunk < fixture.chunks(); ++chunk) {
            const std::size_t slot = chunk % kPipelineDepth;
            if (pending[slot]) {
                check_cuda(cudaEventSynchronize(stops[slot].get()), "cudaEventSynchronize H2D staging slot");
                float elapsed = 0;
                check_cuda(cudaEventElapsedTime(&elapsed, starts[slot].get(), stops[slot].get()), "cudaEventElapsedTime H2D chunk");
                dma_milliseconds += elapsed;
                sum += fixture.packed[slot].data()[0];
                pending[slot] = false;
            }

            const std::size_t chunk_bytes  = fixture.chunk_payload_bytes(chunk);
            const std::size_t chunk_blocks = chunk_bytes / kBlockBytes;
            const std::size_t block_offset = fixture.chunk_block_offset(chunk);
            const auto codec_start         = std::chrono::steady_clock::now();
            compressor.template operator()<BIT_WIDTH>(fixture.input.data() + block_offset * fixture.kElementsPerBlock, fixture.scale,
                                                      fixture.packed[slot].data(), static_cast<std::uint32_t>(chunk_blocks));
            codec_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - codec_start).count();

            check_cuda(cudaEventRecord(starts[slot].get(), stream.get()), "cudaEventRecord H2D chunk start");
            check_cuda(cudaMemcpyAsync(fixture.device.bytes() + block_offset * kBlockBytes, fixture.packed[slot].data(), chunk_bytes,
                                       cudaMemcpyHostToDevice, stream.get()),
                       "cudaMemcpyAsync H2D chunk");
            check_cuda(cudaEventRecord(stops[slot].get(), stream.get()), "cudaEventRecord H2D chunk stop");
            pending[slot] = true;
        }
        for (std::size_t slot = 0; slot < kPipelineDepth; ++slot) {
            if (!pending[slot]) continue;
            check_cuda(cudaEventSynchronize(stops[slot].get()), "cudaEventSynchronize H2D drain");
            float elapsed = 0;
            check_cuda(cudaEventElapsedTime(&elapsed, starts[slot].get(), stops[slot].get()), "cudaEventElapsedTime H2D drain");
            dma_milliseconds += elapsed;
            sum += fixture.packed[slot].data()[0];
            pending[slot] = false;
        }
        benchmark::DoNotOptimize(sum);
        benchmark::ClobberMemory();
    }
    publish_metrics(state, state.iterations(), fixture.useful_bytes(), fixture.payload_bytes, fixture.chunks(), dma_milliseconds,
                    codec_seconds, uncompressed_seconds, sum);
}

template <std::uint8_t BIT_WIDTH, typename ValueT, typename Decompressor>
void BM_pcie_d2h(benchmark::State& state) {
    const std::size_t payload_bytes = static_cast<std::size_t>(state.range(0));
    Fixture<BIT_WIDTH, ValueT> fixture(payload_bytes);
    Stream stream;
    check_cuda(cudaMemset(fixture.device.data(), 0, fixture.payload_bytes), "cudaMemset compressed device payload");
    std::array<Event, kPipelineDepth> starts;
    std::array<Event, kPipelineDepth> stops;
    double dma_milliseconds = 0;
    double codec_seconds    = 0;
    double sum              = 0;
    Decompressor decompressor;
    const double uncompressed_seconds = measure_uncompressed_copy(fixture, cudaMemcpyDeviceToHost);
    for (auto _ : state) {
        auto enqueue = [&](const std::size_t chunk) {
            const std::size_t slot         = chunk % kPipelineDepth;
            const std::size_t chunk_bytes  = fixture.chunk_payload_bytes(chunk);
            const std::size_t block_offset = fixture.chunk_block_offset(chunk);
            check_cuda(cudaEventRecord(starts[slot].get(), stream.get()), "cudaEventRecord D2H chunk start");
            check_cuda(cudaMemcpyAsync(fixture.packed[slot].data(), fixture.device.bytes() + block_offset * kBlockBytes, chunk_bytes,
                                       cudaMemcpyDeviceToHost, stream.get()),
                       "cudaMemcpyAsync D2H chunk");
            check_cuda(cudaEventRecord(stops[slot].get(), stream.get()), "cudaEventRecord D2H chunk stop");
        };

        enqueue(0);
        for (std::size_t chunk = 0; chunk < fixture.chunks(); ++chunk) {
            const std::size_t slot = chunk % kPipelineDepth;
            check_cuda(cudaEventSynchronize(stops[slot].get()), "cudaEventSynchronize D2H chunk");
            float elapsed = 0;
            check_cuda(cudaEventElapsedTime(&elapsed, starts[slot].get(), stops[slot].get()), "cudaEventElapsedTime D2H chunk");
            dma_milliseconds += elapsed;

            if (chunk + 1 < fixture.chunks()) enqueue(chunk + 1);

            const std::size_t chunk_bytes  = fixture.chunk_payload_bytes(chunk);
            const std::size_t chunk_blocks = chunk_bytes / kBlockBytes;
            const std::size_t block_offset = fixture.chunk_block_offset(chunk);
            const auto codec_start         = std::chrono::steady_clock::now();
            decompressor.template operator()<BIT_WIDTH>(fixture.packed[slot].data(), fixture.scale,
                                                        fixture.output.data() + block_offset * fixture.kElementsPerBlock,
                                                        static_cast<std::uint32_t>(chunk_blocks));
            codec_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - codec_start).count();
            sum += fixture.output.data()[block_offset * fixture.kElementsPerBlock];
        }
        benchmark::DoNotOptimize(sum);
        benchmark::ClobberMemory();
    }
    publish_metrics(state, state.iterations(), fixture.useful_bytes(), fixture.payload_bytes, fixture.chunks(), dma_milliseconds,
                    codec_seconds, uncompressed_seconds, sum);
}

}  // namespace pernix_benchmark::pcie

#define PERNIX_PCIE_PAYLOADS(name, width, value_type) \
    BENCHMARK(name)->Arg(4096)->Arg(1048576)->Arg(pernix_benchmark::pcie::poster_payload_bytes<width, value_type>())

#endif
