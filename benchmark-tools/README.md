# Pernix Benchmark Tools

`pernix-bench` configures, builds, and runs the repository's Pernix, CP2K, and optional CUDA PCIe Google Benchmark targets.

## Setup

From this directory:

```console
uv sync
uv run pernix-bench --help
```

The CMake build requires the recursively initialized Pernix and CP2K submodules, a C/C++/Fortran toolchain, `pkg-config`, and Google Benchmark. PCIe runs additionally require the CUDA Toolkit and a visible NVIDIA GPU.

## Run benchmarks

Run every Pernix implementation supported by the host, followed by CP2K:

```console
uv run pernix-bench run
```

Select a benchmark family or implementation:

```console
uv run pernix-bench run pernix
uv run pernix-bench run pernix avx2
uv run pernix-bench run cp2k
uv run pernix-bench run pcie
uv run pernix-bench run pcie avx2
```

Pernix implementations are `fallback`, `avx2`, `bmi2`, `avx512vbmi`, `neon`, and `sve2`. The runner rejects explicitly selected implementations that are incompatible with the host. ARM NEON and SVE2 targets currently provide decompression benchmarks only.

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
