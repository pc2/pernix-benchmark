#!/usr/bin/env bash

#SBATCH -J "Pernix and CP2K Number De/Compression Benchmarks"
#SBATCH -A hpc-prf-ecderi
#SBATCH -t 05:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH -p normal

VALUES=(128 256 384 512 640 768 896 1024 1280 1536 1792 2048 2560 3072 3584 4096 5120 6144 7168 8192 10240 12288 14336 16384 20480 24576 28672 32768 40960 49152 57344 65536 81920 98304 114688 131072 163840 196608 229376 262144 327680 393216 458752 524288 655360 786432 917504 1048576 1310720 1572864 1835008 2097152 2621440 3145728 3670016 4194304 5242880 6291456 7340032 8388608 10485760 12582912 14680064 16777216 20971520 25165824 29360128 33554432 41943040 50331648 58720256 67108864 83886080 100663296 117440512 134217728 167772160 201326592 234881024 268435456 335544320 402653184 469762048 536870912 671088640 805306368 939524096 1073741824)
BENCHMARKS=(daxpy daxpy_sse daxpy_avx daxpy_avx512 copy copy_sse copy_avx copy_avx512 ddot ddot_sse ddot_avx ddot_avx512 update update_sse update_avx update_avx512 triad triad_sse triad_avx triad_avx512 load load_sse load_avx load_avx512 store store_sse store_avx store_avx512)

# Navigate to project root if script is run from scripts directory
if [ "$(basename "$PWD")" == "scripts" ]; then
  cd ..
fi
if [ ! -f "scripts/runBenchmarks.sh" ]; then
  echo "Please run this script from the project root directory."
  exit 1
fi

print_help() {
  cat <<'USAGE'
Usage: scripts/runMemoryBenchmarks.sh [OPTIONS]

Options:
  -s, --slurm                 Run benchmarks using SLURM job scheduler
  -m, --modules               Load required modules for PC2 environment
  -h, --help                  Show this help message and exit
USAGE
}

PARSED=$(getopt -o shm -l slurm,modules,help -n runMemoryBenchmarks.sh -- "$@") || { echo "Invalid options"; exit 2; }
eval set -- "$PARSED"
INSTALL_PC2_MODULES=0
SLURM_EXECUTION=0
if [ -n "$SLURM_JOB_ID" ]; then
  SLURM_EXECUTION=1
  echo "Detected SLURM environment. SLURM execution mode enabled."
fi
while true; do
  case "$1" in
    -s|--slurm) SLURM_EXECUTION=1; shift ;;
    -m|--modules) INSTALL_PC2_MODULES=1; shift ;;
    -h|--help) print_help; exit 0 ;;
    --) shift; break ;;
    *) echo "Internal error while parsing options"; exit 3 ;;
  esac
done

if [ "$INSTALL_PC2_MODULES" -eq 1 ] || [ "$SLURM_EXECUTION" -eq 1 ]; then
  module reset
  module load tools/likwid/5.4.1_zen5
fi

if [ "$SLURM_EXECUTION" -eq 1 ]; then
  if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script is set to run with SLURM, but no SLURM job ID is found. Please submit the script via sbatch or srun."
    exit 1
  fi
  echo "Running in SLURM execution mode."
else
  echo "Running in local execution mode."
fi

if ! command -v likwid-pin &> /dev/null; then
  echo "likwid-pin could not be found. Please ensure LIKWID is installed and in your PATH."
  exit 1
fi

if ! command -v likwid-bench &> /dev/null; then
  echo "likwid-bench could not be found. Please ensure LIKWID is installed and in your PATH."
  exit 1
fi

runBenchmarks() {
  local slurm_execution="$1"
  local output_file="$2"
  local bench_command="likwid-bench"

  if [ "$slurm_execution" -eq 1 ]; then
    bench_command="srun --exclusive likwid-bench"
  fi

  header="data_size"
  for benchmark in "${BENCHMARKS[@]}"; do
    header+=",${benchmark}"
  done
  echo "$header" > "$output_file"

  for bytes in "${VALUES[@]}"; do
    row="${bytes}"
    for benchmark in "${BENCHMARKS[@]}"; do
      # echo "Running benchmark: ${benchmark} with data size: ${bytes} bytes"
      output=$($bench_command -t "${benchmark}" -w S0:"${bytes}"B:1 2>/dev/null)
      bandwidth=$(echo "$output" | grep "MByte/s:" | awk '{print $2}')
      if [[ -n "$bandwidth"  ]]; then
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

OUTPUT_FILE="memory_benchmarks_$(date +%Y%m%d_%H%M%S).csv"
runBenchmarks "$SLURM_EXECUTION" "$OUTPUT_FILE"