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
BENCHMARKS=(daxpy_scalar daxpy_sse daxpy_avx2 daxpy_avx512 copy_scalar copy_sse copy_sse_nt copy_avx2 copy_avx2_nt copy_avx512 copy_avx512_nt ddot_scalar ddot_sse ddot_avx2 ddot_avx512 update_scalar update_sse update_avx2 update_avx512 triad_scalar triad_sse triad_avx2 triad_avx512 load_scalar load_sse load_sse_nt load_avx2 load_avx512 store_scalar store_sse store_avx2 store_avx512)

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
      --membench-bin PATH     Path to the membench executable
      --build-dir DIR         CMake build directory for membench
      --build-jobs N          Parallel build jobs passed to cmake --build
      --target-seconds VALUE  Auto-calibration target seconds per repetition
      --repetitions VALUE     Measured timing samples per kernel
      --warmup-iterations N   Warmup iterations before measuring
      --cpu-id ID             CPU id for membench affinity, negative disables affinity
  -h, --help                  Show this help message and exit
USAGE
}

PARSED=$(getopt -o shm -l slurm,modules,membench-bin:,build-dir:,build-jobs:,target-seconds:,repetitions:,warmup-iterations:,cpu-id:,help -n runMemoryBenchmarks.sh -- "$@") || { echo "Invalid options"; exit 2; }
eval set -- "$PARSED"
INSTALL_PC2_MODULES=0
SLURM_EXECUTION=0
MEMBENCH_BIN="${MEMBENCH_BIN:-}"
BUILD_DIR="${BUILD_DIR:-memory/build}"
BUILD_JOBS="${BUILD_JOBS:-}"
TARGET_SECONDS="${TARGET_SECONDS:-0.1}"
REPETITIONS="${REPETITIONS:-1}"
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-100}"
CPU_ID="${CPU_ID:--1}"
if [ -n "$SLURM_JOB_ID" ]; then
  SLURM_EXECUTION=1
  echo "Detected SLURM environment. SLURM execution mode enabled."
fi
while true; do
  case "$1" in
    -s|--slurm) SLURM_EXECUTION=1; shift ;;
    -m|--modules) INSTALL_PC2_MODULES=1; shift ;;
    --membench-bin) MEMBENCH_BIN="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --build-jobs) BUILD_JOBS="$2"; shift 2 ;;
    --target-seconds) TARGET_SECONDS="$2"; shift 2 ;;
    --repetitions) REPETITIONS="$2"; shift 2 ;;
    --warmup-iterations) WARMUP_ITERATIONS="$2"; shift 2 ;;
    --cpu-id) CPU_ID="$2"; shift 2 ;;
    -h|--help) print_help; exit 0 ;;
    --) shift; break ;;
    *) echo "Internal error while parsing options"; exit 3 ;;
  esac
done

if [ "$INSTALL_PC2_MODULES" -eq 1 ] || [ "$SLURM_EXECUTION" -eq 1 ]; then
  module reset
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

build_membench() {
  if [ -n "$MEMBENCH_BIN" ]; then
    echo "Using explicit membench binary: $MEMBENCH_BIN"
    return
  fi

  if ! command -v cmake &> /dev/null; then
    echo "cmake could not be found. Please install CMake or pass --membench-bin PATH."
    exit 1
  fi

  echo "Configuring membench in $BUILD_DIR"
  cmake -S memory -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release || {
    echo "Failed to configure membench."
    exit 1
  }

  local build_args=(--build "$BUILD_DIR" --target membench --config Release)
  if [ -n "$BUILD_JOBS" ]; then
    build_args+=(--parallel "$BUILD_JOBS")
  else
    build_args+=(--parallel)
  fi

  echo "Building membench"
  cmake "${build_args[@]}" || {
    echo "Failed to build membench."
    exit 1
  }

  MEMBENCH_BIN="$BUILD_DIR/membench"
  if [ ! -x "$MEMBENCH_BIN" ]; then
    MEMBENCH_BIN="$BUILD_DIR/Release/membench"
  fi
}

build_membench
if [ -z "$MEMBENCH_BIN" ] || [ ! -x "$MEMBENCH_BIN" ]; then
  echo "membench executable could not be found after build. Pass --membench-bin PATH if your generator placed it elsewhere."
  exit 1
fi

if ! command -v python3 &> /dev/null; then
  echo "python3 is required to read membench JSON output."
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

declare -A AVAILABLE_KERNELS=()
while IFS= read -r kernel; do
  AVAILABLE_KERNELS["$kernel"]=1
done < <("$MEMBENCH_BIN" --list-kernels)

run_membench() {
  local kernel="$1"
  local bytes="$2"
  local json_file="$3"

  if [ "$SLURM_EXECUTION" -eq 1 ]; then
    srun --exclusive "$MEMBENCH_BIN" \
      --kernel "$kernel" \
      --bytes "${bytes}B" \
      --iterations 0 \
      --target-seconds "$TARGET_SECONDS" \
      --repetitions "$REPETITIONS" \
      --warmup-iterations "$WARMUP_ITERATIONS" \
      --cpu-id "$CPU_ID" \
      --output "$json_file"
  else
    "$MEMBENCH_BIN" \
      --kernel "$kernel" \
      --bytes "${bytes}B" \
      --iterations 0 \
      --target-seconds "$TARGET_SECONDS" \
      --repetitions "$REPETITIONS" \
      --warmup-iterations "$WARMUP_ITERATIONS" \
      --cpu-id "$CPU_ID" \
      --output "$json_file"
  fi
}

read_bandwidth_bps() {
  local json_file="$1"
  python3 - "$json_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)

results = data.get("results", [])
if not results:
    sys.exit(1)

value = results[0]["bandwidth_bytes_per_second"]["mean"]
print(f"{value:.0f}")
PY
}

runBenchmarks() {
  local output_file="$1"

  header="data_size"
  for benchmark in "${BENCHMARKS[@]}"; do
    header+=",${benchmark}"
  done
  echo "$header" > "$output_file"

  for bytes in "${VALUES[@]}"; do
    row="${bytes}"
    for benchmark in "${BENCHMARKS[@]}"; do
      if [ -z "${AVAILABLE_KERNELS[$benchmark]+x}" ]; then
        row+=","
        continue
      fi

      json_file="$TMP_DIR/${benchmark}_${bytes}.json"
      echo "Running benchmark: ${benchmark} with data size: ${bytes} bytes"
      if run_membench "$benchmark" "$bytes" "$json_file" > "$TMP_DIR/${benchmark}_${bytes}.log" 2>&1; then
        bandwidth_bps="$(read_bandwidth_bps "$json_file" || true)"
        if [[ -n "$bandwidth_bps" ]]; then
          echo "Found bandwidth: $bandwidth_bps B/s for benchmark: $benchmark with data size: ${bytes} bytes"
          row+=",${bandwidth_bps}"
        else
          row+=","
        fi
      else
        echo "Benchmark failed: ${benchmark} with data size: ${bytes} bytes"
        cat "$TMP_DIR/${benchmark}_${bytes}.log"
        row+=","
      fi
    done
    echo "$row" >> "$output_file"
  done
}

OUTPUT_FILE="memory_benchmarks_$(date +%Y%m%d_%H%M%S).csv"
runBenchmarks "$OUTPUT_FILE"
