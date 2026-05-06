# Licensing Boundary

This repository contains code under more than one license. This file documents the intended license
boundary for the benchmark framework and the CP2K-based benchmark target. It is a repository
organization note, not legal advice.

## MIT-Licensed Benchmark Framework

The repository's original benchmark framework code is licensed under the MIT License in
`LICENSE`, except where that code incorporates, modifies, or links CP2K GPL-covered code.

This MIT-covered framework code includes the independent Pernix benchmark implementation and
benchmark harness code that does not link against `cp2k_compression`.

## CP2K GPL-Covered Components

CP2K is included as the `external/cp2k` Git submodule and is licensed GPL-2.0-or-later. See
`external/cp2k/LICENSE` and the CP2K project notices for the authoritative CP2K license terms.

The following repository components must be treated as GPL-2.0-or-later because they are built from,
modify, or link against CP2K source code:

- The `cp2k_compression` static library target.
- The selected CP2K source files compiled into `cp2k_compression`.
- CP2K patch files under `external/cp2k-patches/`, including patches applied before building the
  CP2K-derived target.
- Benchmark executables linked against `cp2k_compression`, including `bench_cp2k`.
- Source files that are specific to the CP2K benchmark path and are compiled into such executables,
  including files under `src/cp2k/`.

The current CP2K-derived source subset is defined in `external/CMakeLists.txt` as
`CP2K_SOURCE_FILES`. At the time of writing, it includes:

```cmake
cp2k/src/base/machine_cpuid.c
cp2k/src/common/cp_data_dir.c
cp2k/src/base/kinds.F
cp2k/src/hfx_types.F
cp2k/src/base/machine.F
cp2k/src/base/base_hooks.F
cp2k/src/common/cp_files.F
cp2k/src/hfxbase/hfx_compression_core_methods.F
cp2k/src/hfx_compression_methods.F
```

The current repository patch is `external/cp2k-patches/0001-Delete-unneeded-types-hfx_types.F.patch`,
which modifies CP2K source before building the CP2K-derived static library.

## Submodules and Linking

The presence of a Git submodule by itself does not automatically relicense independent benchmark
framework code. The boundary changes when CP2K GPL-covered source is compiled into a library or when
a benchmark executable is linked with that CP2K-derived library.

According to the GNU GPL FAQ, static or dynamic linking with GPL-covered code creates a combined
work. For this repository, that means the `cp2k_compression` target and any benchmark executable
linked against it are intended to be distributed under GPL-2.0-or-later terms, while independent
benchmark targets that do not incorporate or link CP2K code remain under their own applicable
licenses.
