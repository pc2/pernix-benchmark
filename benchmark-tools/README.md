# Pernix Benchmark Tools

This uv project replaces the repository's benchmark shell scripts
incrementally. The current CLI builds and runs the Pernix and CP2K Google
Benchmark targets locally. The existing scripts remain available in
`../scripts/` for workflows that have not been migrated yet.

## Setup

```console
python -m uv sync
python -m uv run pernix-bench --help
```

For result exploration and visualization, install the optional notebook
environment and start JupyterLab from this directory:

```console
python -m uv sync --group notebook
python -m uv run --group notebook jupyter lab notebooks/
```

## Commands

Run every Pernix implementation supported by the current CPU, followed by
CP2K:

```console
python -m uv run pernix-bench run
```

On x86 hosts that support both AVX2 and AVX-512-VBMI, this complete run also
generates the instruction-throughput model used by the poster notebook. It
requires `llvm-mca` and uses LIKWID's `CLOCK` group for hardware validation. The model is compiled at
`-O3`, analyzes a model-only fully unrolled one-block kernel for every width
N=2 through 24, and measures the actual CPU frequency instead of assuming a
nominal clock. The result directory gains:

- `incore_models.csv`, the plot-ready model throughput;
- `incore_model_diagnostics.csv`, static and retired instruction counts, uops,
  IPC, cycles, and toolchain details;
- `incore_model_metadata.json`, the model semantics and LLVM target metadata.

The same files can be generated or regenerated without rerunning Google
Benchmark:

```console
python -m uv run pernix-bench model \
  --results-dir ../benchmark-results/<run> \
  --llvm-cpu znver5
```

Set `LLVM_MCA` or pass `--llvm-mca` if the executable has a versioned name.
The default one-second LIKWID probe duration applies independently to the real
kernel and its no-op control. `model_instructions_per_block` is LLVM-MCA's
static instruction count; `retired_instructions_per_block` is the
control-subtracted LIKWID hardware validation count. Override the executable
with `LIKWID_PERFCTR` or `--likwid-perfctr`. If LIKWID counters cannot be used,
the generator records that error and samples the pinned CPU's live Linux
frequency; the two hardware-counter diagnostic columns remain empty.

Run only the supported Pernix implementations:

```console
python -m uv run pernix-bench run pernix
```

Run one implementation or CP2K:

```console
python -m uv run pernix-bench run pernix avx2
python -m uv run pernix-bench run cp2k
```

Run CUDA PCIe end-to-end benchmarks for every supported x86 Pernix implementation,
or select one implementation:

```console
python -m uv run pernix-bench run pcie
python -m uv run pernix-bench run pcie avx2
```

The PCIe suite requires the CUDA Toolkit at build time and a visible NVIDIA GPU at
run time. It measures the two directions separately: host compression followed by
a synchronized asynchronous H2D transfer, and a synchronized asynchronous D2H
transfer followed by host decompression. All host and device buffers are allocated
before timing; host buffers are CUDA-pinned and the default stream is used without
codec/transfer overlap. Each 1–24 bit width is run for f32 and f64 with 4 KiB and
1 MiB packed payloads plus a width-dependent payload representing approximately
1 GiB of original values. A separately warmed, at-least-250-ms uncompressed copy
measurement supplies the baseline. Google Benchmark throughput is the useful
uncompressed data rate; component timings and pipeline metadata are stored as
counters. ARM variants are not included because Pernix ARM compression is not yet
available.

Pernix implementation names are `fallback`, `avx2`, `bmi2`, `avx512vbmi`,
`neon`, and `sve2`. The portable `fallback` implementation runs on every host.
An ISA-specific implementation requested explicitly must match the host
architecture and CPU features. On ARM, the NEON and SVE2 executables currently
contain decompression benchmarks only because Pernix ARM compression is not yet
implemented.

The CP2K executable benchmarks its f32 and f64 packing kernels in the same 512-bit
matrix as Pernix for bit widths 1 through 24. Both cache-hot core (`true`) and full
memory-throughput (`false`) modes use the common
`BM_[direction]_cp2k[value_type]_[core]_[width]/[blocks]` format so result rows can be
compared directly. Core mode reuses small buffers; the kernels still execute loads and stores.
Full-memory mode targets working sets from 4 KiB through 2 GiB at two samples per
octave and adds samples at 0.75, 1, and 1.25 times the measured L1, L2, and L3
capacities. Core mode retains the power-of-two block-count sweep.

Both commands accept `--compiler`, `--build-type`, `--jobs`, `--build-dir`,
`--output-dir`, `--clean`, `--benchmark-min-time`, `--benchmark-repetitions`, and
`--benchmark-min-warmup-time`. Each stored repetition defaults to at least 0.25
seconds after at least 0.05 seconds of warm-up, and five repetitions are retained
for median-based analysis. Increase the measurement time for lower-noise runs,
for example with `--benchmark-min-time 1.0`. Results are Google Benchmark JSON
files under `../benchmark-results/<YYYYMMDD_HHMMSS>/`. Before benchmark
execution, the CLI captures extended system information once per invocation in
`machinestate.json` in the same directory. Collection failures are reported as
warnings and do not prevent benchmarks from running. Each result's Google
Benchmark context records the selected minimum time.

When `SLURM_JOB_ID` is present, the runner updates the job comment as each
implementation starts, for example
`1/4 (Current: AVX2) [00:12:34<00:37:42]`. The two times are elapsed time and
estimated remaining time, calculated from completed implementations. The first
implementation shows `[00:00:00<?]` because no timing sample exists yet. After the final
implementation, the comment reports the total time, such as
`4/4 (Complete) [00:50:16]`.

## Otus SLURM jobs

The Otus launcher uses one exclusive `normal` node with all 192 CPU cores. Set
up the environment once on an Otus login node using the same modules as the
batch job:

```console
module reset
module load lang/Python/3.13.5-GCCcore-14.3.0
module load devel/CMake/4.0.3-GCCcore-14.3.0
module load tools/googlebenchmark/1.9.4-GCCcore-14.3.0
cd benchmark-tools
python -m uv sync --frozen
cd ..
```

Also load the site LLVM 20 module that supplies `llvm-mca-20` or `llvm-mca`, or
export `LLVM_MCA=/path/to/llvm-mca`. The batch launcher loads
`tools/likwid/5.5.2rc2-GCC-14.3.0`, checks both tools, and defaults
`PERNIX_LLVM_CPU` to `znver5` for Otus.

Submit from the repository root. With no arguments the job runs every
host-supported Pernix benchmark followed by CP2K:

```console
sbatch scripts/runBenchmarksOtus.slurm
```

Arguments after the script select a narrower benchmark run:

```console
sbatch scripts/runBenchmarksOtus.slurm pernix avx512vbmi
sbatch scripts/runBenchmarksOtus.slurm cp2k
```

Submit the CUDA PCIe suite through its dedicated GPU launcher:

```console
sbatch scripts/runPcieBenchmarksOtus.slurm
sbatch scripts/runPcieBenchmarksOtus.slurm avx2
```

The PCIe launcher requests one H100 from the `gpu_h100` partition and does not
request an exclusive node allocation.

The CPU job writes benchmark JSON and `machinestate.json` to
`benchmark-results/otus-<job-id>/`; the PCIe job writes them plus negotiated-link
metadata to `benchmark-results/pcie-<job-id>/`. Compilation uses all allocated cores in a
fresh job-local directory, which is removed when the job exits. SLURM output is
written to `slurm-pernix-benchmarks-<job-id>.out` in the submission directory.
If a listed module version is no longer available, use `find_module` on Otus to
locate its replacement and update both the setup commands and launcher.

CPU pinning, SIMDe, result archives, LIKWID memory benchmarks, and result
transformation are planned for later migrations.

Notebooks belong in `notebooks/`. Shared loading, transformation, and plotting
logic should live in the Python package so notebooks remain small and
reproducible.
