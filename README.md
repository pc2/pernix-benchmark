# Pernix Benchmark Framework

This repository provides a [Google Benchmark](https://github.com/google/benchmark)-based framework for measuring [Pernix](https://github.com/pc2/pernix) compression and decompression implementations. It also includes comparable CP2K packing benchmarks, CUDA PCIe end-to-end benchmarks, and instruction-throughput modeling tools.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- A C, C++, and Fortran toolchain
- Google Benchmark
- LLVM-MCA for instruction modeling during a complete run on supported x86 hosts
- The CUDA Toolkit and a supported NVIDIA GPU for the optional PCIe benchmarks

Clone the repository with its submodules:

```console
git clone --recursive https://github.com/pc2/pernix-benchmark.git
cd pernix-benchmark/benchmark-tools
```

## Quickstart

Create the benchmark-tools environment and run every Pernix implementation supported by the host, followed by CP2K:

```console
uv sync
uv run pernix-bench run
```

Run a specific Pernix implementation, CP2K alone, or the optional CUDA PCIe suite:

```console
uv run pernix-bench run pernix avx2
uv run pernix-bench run cp2k
uv run pernix-bench run pcie avx2
```

The runner configures and builds the selected CMake targets, captures machine metadata, and writes Google Benchmark JSON files under `benchmark-results/<timestamp>/` in the repository root.

Pernix benchmark iterations call the native multi-block kernels with one scale for the complete batch. This avoids per-block framework dispatch overhead and follows the Pernix multi-block API contract. CP2K uses the same 512-bit block, value-type, bit-width, and memory-mode dimensions so its results can be compared with Pernix.

The PCIe suite uses two pinned 1 MiB staging buffers. H2D runs overlap compression of the next chunk with the current DMA transfer; D2H runs overlap the next DMA transfer with decompression of the current chunk. The reported pipeline runtime is therefore wall-clock end-to-end time, while `codec_seconds` and `dma_seconds` are component-work measurements and are not expected to add up to it.

See the [benchmark tools guide](benchmark-tools/README.md) for CLI options, supported implementations, instruction modeling, and CUDA requirements. Otus setup and submission are covered by the separate [Slurm guide](slurm/README.md).

## Licensing

The original benchmark framework code is MIT-licensed under [LICENSE](LICENSE).

The repository also contains a CP2K submodule under `external/cp2k`. CP2K is GPL-2.0-or-later. The `cp2k_compression` static library is built from selected CP2K source files, and `bench_cp2k` links against that library. Those CP2K-derived targets, the CP2K patch files under `external/cp2k-patches/`, and benchmark executables linked with `cp2k_compression` must be treated as GPL-2.0-or-later.

See [LICENSES.md](LICENSES.md) for the intended license boundary. This documentation is not legal advice.
