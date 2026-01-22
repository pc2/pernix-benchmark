# Overview
This repository provides a benchamrk framewaork based on google benchmark (https://github.com/google/benchmark) for PERNIX (https://github.com/pc2/pernix).

# Installation
1. install google benchmark https://github.com/google/benchmark?tab=readme-ov-file#installation
2. clone repo: `git clone --recursive https://github.com/pc2/pernix-benchmark`

# Running the Benchmarks
```
Usage: scripts/runBenchmarks.sh [OPTIONS]

Options:
  -t, --target TARGET         Specify target to benchmark (default: all targets)
      --list-targets          List all available benchmark targets
  -c, --compiler COMPILER     Specify compiler to use (e.g. gcc, clang)
      --release-type TYPE     Specify release type (e.g. Debug, Release)
  -o, --output PATH           Specify output archive path (default: benchmark_results.tar.gz)
  -s, --slurm                 Run benchmarks using SLURM job scheduler
  -m, --modules               Load required modules for PC2 environment
      --pin                   Pin benchmark to a specific CPU core
      --clean                 Clean build directory before building
  -h, --help                  Show this help message and exit
```

# Examples:
* AVX512-VMBI (e.g. AMD Zen 5): `bash scripts/runBenchmarks.sh --target pernix_avx512vbmi  --release-type Release`
* AVX2+BMI2: `bash scripts/runBenchmarks.sh --target pernix_bmi2  --release-type Release`

# Meaning of Benchmark Identifiers
`BM_[direction]_[target]_[core-throughput flag]_[bit width]/[number of blocks per iteration]`
* direction: compress, decompress
* target (instruction set): pernix_fallback, pernix_avx2, pernix_bmi2, pernix_avx512vbmi, cp2k
* core-throughput flag: true= no loads and stores (core-throughput scenario), false=including loads and stores (full throughput scenario)
* bit width: width of compressed numbers
* number of blocks per iteration: number of 512-bit blocks processed per iteration
# Example Output (AMD Ryzen 7 9700X, GCC 15.2.0):
* bash scripts/runBenchmarks.sh --target pernix_avx512vbmi --release-type Release
```
----------------------------------------------------------------------------------------------------
Benchmark                                          Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------------------------
BM_compress_avx512vbmi_true_8/1                0.976 ns        0.975 ns    718152018 bytes_per_second=61.1536Gi/s items_per_second=1.02599G/s sum=0
BM_compress_avx512vbmi_true_8/2                 1.86 ns         1.86 ns    374432549 bytes_per_second=64.1302Gi/s items_per_second=1.07593G/s sum=0
BM_compress_avx512vbmi_true_8/4                 3.71 ns         3.70 ns    189712756 bytes_per_second=64.3988Gi/s items_per_second=1.08043G/s sum=0
BM_compress_avx512vbmi_true_8/8                 7.34 ns         7.30 ns     93810270 bytes_per_second=65.324Gi/s items_per_second=1.09596G/s sum=0
BM_compress_avx512vbmi_true_8/16                14.7 ns         14.6 ns     46871103 bytes_per_second=65.108Gi/s items_per_second=1.09233G/s sum=0
BM_compress_avx512vbmi_true_8/32                29.4 ns         29.3 ns     23918928 bytes_per_second=64.9922Gi/s items_per_second=1.09039G/s sum=0
BM_compress_avx512vbmi_true_8/64                58.8 ns         58.4 ns     11509539 bytes_per_second=65.3221Gi/s items_per_second=1.09592G/s sum=0
BM_compress_avx512vbmi_true_8/128                118 ns          117 ns      6030110 bytes_per_second=65.3272Gi/s items_per_second=1.09601G/s sum=0
BM_compress_avx512vbmi_true_8/256                252 ns          251 ns      2810948 bytes_per_second=60.8622Gi/s items_per_second=1.0211G/s sum=0
BM_compress_avx512vbmi_true_8/512                500 ns          500 ns      1350947 bytes_per_second=61.0802Gi/s items_per_second=1.02476G/s sum=0
BM_compress_avx512vbmi_true_8/1024               983 ns          982 ns       712877 bytes_per_second=62.1356Gi/s items_per_second=1.04246G/s sum=0
BM_compress_avx512vbmi_true_8/2048              1975 ns         1970 ns       355349 bytes_per_second=61.961Gi/s items_per_second=1.03953G/s sum=0
BM_compress_avx512vbmi_true_8/4096              3943 ns         3913 ns       178889 bytes_per_second=62.3953Gi/s items_per_second=1.04682G/s sum=0
BM_compress_avx512vbmi_true_8/8192              7878 ns         7857 ns        89061 bytes_per_second=62.1453Gi/s items_per_second=1.04262G/s sum=0
BM_compress_avx512vbmi_true_8/16384            15561 ns        15537 ns        44904 bytes_per_second=62.8535Gi/s items_per_second=1.05451G/s sum=0
BM_compress_avx512vbmi_true_8/32768            30934 ns        30764 ns        22808 bytes_per_second=63.4865Gi/s items_per_second=1.06513G/s sum=0
BM_compress_avx512vbmi_true_8/65536            62174 ns        61922 ns        11510 bytes_per_second=63.0832Gi/s items_per_second=1.05836G/s sum=0
BM_compress_avx512vbmi_true_8/131072          121324 ns       120424 ns         5846 bytes_per_second=64.8749Gi/s items_per_second=1.08842G/s sum=0
BM_compress_avx512vbmi_true_8/262144          242927 ns       242638 ns         2903 bytes_per_second=64.3962Gi/s items_per_second=1.08039G/s sum=0
BM_compress_avx512vbmi_true_8/524288          481676 ns       481255 ns         1383 bytes_per_second=64.9344Gi/s items_per_second=1.08942G/s sum=0
BM_compress_avx512vbmi_true_8/1048576         962669 ns       961290 ns          718 bytes_per_second=65.0168Gi/s items_per_second=1.0908G/s sum=0
BM_compress_avx512vbmi_true_8/2097152        1923946 ns      1923075 ns          362 bytes_per_second=65.0001Gi/s items_per_second=1.09052G/s sum=0
BM_compress_avx512vbmi_true_8/4194304        3872130 ns      3849850 ns          181 bytes_per_second=64.9376Gi/s items_per_second=1.08947G/s sum=0
BM_compress_avx512vbmi_true_9/1                 7.55 ns         7.49 ns     92748076 bytes_per_second=7.82876Gi/s items_per_second=133.43M/s sum=0
BM_compress_avx512vbmi_true_9/2                 15.0 ns         14.9 ns     46971995 bytes_per_second=7.86999Gi/s items_per_second=134.132M/s sum=0
```

