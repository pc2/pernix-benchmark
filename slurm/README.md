# Slurm Benchmark Jobs

These launchers run the benchmark suite on the Otus cluster. Submit them from the repository root so results and logs are written to predictable locations.

## Prepare the environment

Create the frozen `uv` environment once on an Otus login node:

```console
module reset
module load lang/Python/3.13.5-GCCcore-14.3.0
module load devel/CMake/4.0.3-GCCcore-14.3.0
module load tools/googlebenchmark/1.9.4-GCCcore-14.3.0
cd benchmark-tools
uv sync --frozen
cd ..
```

The CPU batch job also loads GCC 14.3, LLVM 20, and LIKWID. If the site module versions change, locate replacements with `find_module` and update both the setup commands and launcher.

## CPU benchmarks and instruction model

```console
slurm/submit-otus-benchmarks.sh
```

The wrapper submits AVX2, AVX2+BMI2, AVX-512-VBMI, and CP2K as a four-task array. Each task receives an exclusive 192-core `normal` node for up to four hours. A one-CPU, 8 GiB instruction-model job is submitted with an `afterok` dependency on the complete array.

Both stages write to `benchmark-results/otus-<array-job-id>/`. Machine-state filenames are task-specific, array logs use `slurm-pernix-benchmarks-<array-job-id>_<task-id>.out`, and temporary build directories are removed when jobs exit.

The following environment variables override defaults:

- `PERNIX_BENCHMARK_MIN_TIME_SECONDS`
- `PERNIX_BENCHMARK_REPETITIONS`
- `PERNIX_BENCHMARK_MIN_WARMUP_TIME_SECONDS`
- `LLVM_MCA`
- `LIKWID_PERFCTR`
- `PERNIX_LLVM_CPU`

The runner updates the Slurm job comment with completed targets, the current target, elapsed time, and estimated remaining time. The submission wrapper prints both job IDs; monitor them with the displayed `squeue` command.

## CUDA PCIe benchmarks

Run every supported x86 implementation or select one:

```console
sbatch slurm/otus-pcie-benchmark.sbatch
sbatch slurm/otus-pcie-benchmark.sbatch avx2
```

The launcher requests one H100 from `gpu_h100`, 16 CPU cores, and a two-hour limit. Results, negotiated-link metadata, and machine state are written under `benchmark-results/pcie-<job-id>/`; the Slurm log is written to the submission directory.

## LIKWID memory benchmarks

```console
sbatch slurm/otus-memory-benchmark.sbatch
```

This Slurm-only job measures the configured LIKWID kernels over the repository's memory-size sweep. It requests one exclusive `normal` node task and writes `memory_benchmarks_<timestamp>.csv` to the submission directory.
