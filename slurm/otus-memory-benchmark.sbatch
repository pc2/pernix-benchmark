#!/usr/bin/env bash

#SBATCH -J "Pernix and CP2K Number De/Compression Benchmarks"
#SBATCH -A hpc-prf-ecderi
#SBATCH -t 05:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -p normal

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch." >&2
  exit 1
fi

VALUES=(128 256 384 512 640 768 896 1024 1280 1536 1792 2048 2560 3072 3584 4096 5120 6144 7168 8192 10240 12288 14336 16384 20480 24576 28672 32768 40960 49152 57344 65536 81920 98304 114688 131072 163840 196608 229376 262144 327680 393216 458752 524288 655360 786432 917504 1048576 1310720 1572864 1835008 2097152 2621440 3145728 3670016 4194304 5242880 6291456 7340032 8388608 10485760 12582912 14680064 16777216 20971520 25165824 29360128 33554432 41943040 50331648 58720256 67108864 83886080 100663296 117440512 134217728 167772160 201326592 234881024 268435456 335544320 402653184 469762048 536870912 671088640 805306368 939524096 1073741824)
BENCHMARKS=(daxpy daxpy_sse daxpy_avx daxpy_avx512 copy copy_sse copy_avx copy_avx512 ddot ddot_sse ddot_avx ddot_avx512 update update_sse update_avx update_avx512 triad triad_sse triad_avx triad_avx512 load load_sse load_avx load_avx512 store store_sse store_avx store_avx512)

module reset
module load tools/likwid/5.4.1_zen5

if ! command -v likwid-bench &> /dev/null; then
  echo "likwid-bench could not be found. Please ensure LIKWID is installed and in your PATH."
  exit 1
fi

run_benchmarks() {
  local output_file="$1"

  header="data_size"
  for benchmark in "${BENCHMARKS[@]}"; do
    header+=",${benchmark}"
  done
  echo "$header" > "$output_file"

  for bytes in "${VALUES[@]}"; do
    row="${bytes}"
    for benchmark in "${BENCHMARKS[@]}"; do
      output=$(srun --exclusive --ntasks=1 --cpus-per-task=1 \
        likwid-bench -t "${benchmark}" -w S0:"${bytes}"B:1 2>/dev/null)
      bandwidth=$(awk '/MByte\/s:/ {print $2; exit}' <<< "${output}")
      if [[ -n "$bandwidth" ]]; then
        echo "Found bandwidth: $bandwidth MByte/s for benchmark: $benchmark with data size: ${bytes} bytes"
        bandwidth_bps=$(awk "BEGIN {printf \"%.0f\", $bandwidth * 1024 * 1024}")
        row+=",${bandwidth_bps}"
      else
        row+=","
      fi
    done
    echo "$row" >> "$output_file"
  done
}

OUTPUT_FILE="${SLURM_SUBMIT_DIR:-$PWD}/memory_benchmarks_$(date +%Y%m%d_%H%M%S).csv"
run_benchmarks "$OUTPUT_FILE"
