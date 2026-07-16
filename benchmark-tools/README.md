# Pernix Benchmark Tools

This uv project replaces the repository's benchmark shell scripts
incrementally. The current CLI builds and runs the Pernix and CP2K Google
Benchmark targets locally. The existing scripts remain available in
`../scripts/` for workflows that have not been migrated yet.

## Setup

```console
uv sync
uv run pernix-bench --help
```

For result exploration and visualization, install the optional notebook
environment and start JupyterLab from this directory:

```console
uv sync --group notebook
uv run --group notebook jupyter lab notebooks/
```

## Commands

Run every Pernix implementation supported by the current CPU, followed by
CP2K:

```console
uv run pernix-bench run
```

Run only the supported Pernix implementations:

```console
uv run pernix-bench run pernix
```

Run one implementation or CP2K:

```console
uv run pernix-bench run pernix avx2
uv run pernix-bench run cp2k
```

Pernix implementation names are `fallback`, `avx2`, `bmi2`, `avx512vbmi`,
`neon`, and `sve2`. The portable `fallback` implementation runs on every host.
An ISA-specific implementation requested explicitly must match the host
architecture and CPU features. On ARM, the NEON and SVE2 executables currently
contain decompression benchmarks only because Pernix ARM compression is not yet
implemented.

Both commands accept `--compiler`, `--build-type`, `--jobs`, `--build-dir`,
`--output-dir`, and `--clean`. By default, results are Google Benchmark JSON
files under `../benchmark-results/<YYYYMMDD_HHMMSS>/`. Before benchmark
execution, the CLI captures extended system information once per invocation in
`machinestate.json` in the same directory. Collection failures are reported as
warnings and do not prevent benchmarks from running.

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
uv sync --frozen
cd ..
```

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

The job writes benchmark JSON and `machinestate.json` to
`benchmark-results/otus-<job-id>/`. Compilation uses all allocated cores in a
fresh job-local directory, which is removed when the job exits. SLURM output is
written to `slurm-pernix-benchmarks-<job-id>.out` in the submission directory.
If a listed module version is no longer available, use `find_module` on Otus to
locate its replacement and update both the setup commands and launcher.

CPU pinning, SIMDe, result archives, LIKWID memory benchmarks, and result
transformation are planned for later migrations.

Notebooks belong in `notebooks/`. Shared loading, transformation, and plotting
logic should live in the Python package so notebooks remain small and
reproducible.
