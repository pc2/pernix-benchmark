# Pernix Benchmark Tools

`pernix-bench` configures, builds, and runs the repository's Pernix, CP2K, libxs, and optional CUDA PCIe Google Benchmark targets.

## Setup

From this directory:

```console
uv sync
uv run pernix-bench --help
```

The CMake build requires the Pernix and libxs submodules, a C/C++ toolchain, `pkg-config`, and Google Benchmark. CP2K runs also require the CP2K submodule and a Fortran compiler. PCIe runs additionally require the CUDA Toolkit and a visible NVIDIA GPU.

## Run benchmarks

Run every Pernix implementation supported by the host, followed by CP2K and libxs:

```console
uv run pernix-bench run
```

Select a benchmark family or implementation:

```console
uv run pernix-bench run pernix
uv run pernix-bench run pernix avx2
uv run pernix-bench run cp2k
uv run pernix-bench run libxs
uv run pernix-bench run hfx
uv run pernix-bench run pcie
uv run pernix-bench run pcie avx2
```

Pernix implementations are `fallback`, `avx2`, `bmi2`, `avx512vbmi`, `neon`, and `sve2`. The runner rejects explicitly selected implementations that are incompatible with the host. ARM NEON and SVE2 targets currently provide decompression benchmarks only.

The libxs suite times `libxs_vpack` and `libxs_vunpack` for signed and unsigned 8-, 16-, 32-, and 64-bit values, group sizes 1–32, and broad or sparse synthetic inputs. Inputs are deterministic, and results include packed bytes, bits per value, compression ratio, and throughput.

The HFX suite compares CP2K's original fixed-width codec, `libxs_vpack` with group size 8, and Pernix on the same deterministic synthetic 64×32×16 tensor in three separately compiled and run ISA variants: scalar, AVX2, and AVX-512-VBMI. The tensor has mostly zero or small values and a sparse wide-magnitude tail. It quantizes at widths 2–24 using `--scale` (default: `1.0`) and stores 1024-value cache batches in 1024-word in-memory container nodes. Each codec retains its native payload framing: CP2K's fixed-width stream, LIBXS's length-prefixed group record, or Pernix's fixed-size blocks. Results are written to `benchmark_hfx_scalar_results.json`, `benchmark_hfx_avx2_results.json`, and `benchmark_hfx_avx512_results.json`; each file records its ISA and includes the implementation name, synthetic profile, dimensions, scale, logical packed size, allocated container size, compression ratio, and throughput. The runner requires an x86 host with AVX2 and AVX-512 support, including VNNI for LIBXS and VBMI for Pernix. With `--build-dir DIR`, it creates separate sibling build directories named `DIR_hfx_scalar`, `DIR_hfx_avx2`, and `DIR_hfx_avx512`. The LIBXS path follows [CP2K PR #6352](https://github.com/cp2k/cp2k/pull/6352).

Run commands accept `--compiler`, `--build-type`, `--jobs`, `--build-dir`, `--output-dir`, `--clean`, `--benchmark-min-time`, `--benchmark-repetitions`, and `--benchmark-min-warmup-time`. Use the relevant subcommand's `--help` for defaults and details.

Results are written as Google Benchmark JSON under `../benchmark-results/<timestamp>/` by default. The same directory receives `machinestate.json`; collection failures are warnings and do not stop benchmark execution.

## Instruction model

On x86 hosts supporting AVX2 and AVX-512-VBMI, a complete `pernix-bench run` also generates an LLVM-MCA instruction-throughput model. LLVM-MCA is required; LIKWID hardware validation is optional and falls back to the live Linux CPU frequency when counters are unavailable.

Generate or refresh the model without rerunning Google Benchmark:

```console
uv run pernix-bench model \
  --results-dir ../benchmark-results/<run> \
  --llvm-cpu znver5
```

Set `LLVM_MCA` and `LIKWID_PERFCTR` to override tool discovery. The generated `incore_models.csv`, diagnostics CSV, and metadata JSON are placed in the selected result directory. Model-specific header transformations use a build-directory overlay and do not modify the Pernix submodule.

Cluster setup and submission are documented separately in the [Slurm guide](../slurm/README.md).
