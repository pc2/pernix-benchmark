#!/usr/bin/env bash

#SBATCH -J "Pernix and CP2K Number De/Compression Benchmarks"
#SBATCH -A hpc-prf-ecderi
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH -p normal

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
Usage: scripts/runBenchmarks.sh [OPTIONS]

Options:
  -t, --target TARGET         Specify target to benchmark (default: all targets)
      --list-targets          List all available benchmark targets
  -c, --compiler COMPILER     Specify compiler to use (e.g. gcc, clang)
      --release-type TYPE     Specify release type (e.g. Debug, Release)
  -o, --output PATH           Specify output archive path (default: benchmark_results.tar.gz)
  -s, --slurm                 Run benchmarks using SLURM job scheduler
  -m, --modules               Load required modules for PC2 environment
      --simde                 Build and run Pernix benchmark targets with SIMDe
      --pin                   Pin benchmark to a specific CPU core
      --clean                 Delete the selected build directory and .venv before building
  -h, --help                  Show this help message and exit
USAGE
}

AVAILABLE_TARGETS=("cp2k" "pernix_fallback" "pernix_avx2" "pernix_bmi2" "pernix_avx512vbmi")

print_available_targets() {
  # print list  AVAILABLE_TARGETS
  echo "Available benchmark targets:"
  for target in "${AVAILABLE_TARGETS[@]}"; do
    echo "  - $target"
  done
}

validate_targets() {
  local -a targets=("$@")
  local t at found trimmed

  for t in "${targets[@]}"; do
    trimmed="$(echo "$t" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    found=0
    for at in "${AVAILABLE_TARGETS[@]}"; do
      if [ "$trimmed" = "$at" ]; then
        found=1
        break
      fi
    done
    if [ "$found" -eq 0 ]; then
      echo "Unknown target: $trimmed"
      print_available_targets
      exit 1
    fi
  done
}

parse_target_list() {
  local value="$1"
  local oldIFS="$IFS"
  IFS=','
  read -r -a TARGETS <<< "$value"
  IFS="$oldIFS"
  validate_targets "${TARGETS[@]}"
}

get_cpu_count() {
  if command -v nproc &> /dev/null; then
    nproc
  elif command -v sysctl &> /dev/null; then
    sysctl -n hw.ncpu
  else
    echo 1
  fi
}

build_benchmarks() {
  local compiler="$1"
  local release_type="$2"
  local -a targets=("${!3}")
  local slurm_execution="$4"
  local use_simde="$5"
  local cmake_args=()
  local -a build_targets=()

  cmake_args+=("-DCMAKE_CXX_COMPILER=$compiler")
  cmake_args+=("-DCMAKE_BUILD_TYPE=$release_type")
  if [ "$use_simde" -eq 1 ]; then
    cmake_args+=("-DPERNIX_BENCHMARK_USE_SIMDE=ON")
  else
    cmake_args+=("-DPERNIX_BENCHMARK_USE_SIMDE=OFF")
  fi

  for t in "${targets[@]}"; do
    build_targets+=("bench_$t")
  done

  echo "Configuring benchmarks with CMake..."
  echo "  CMake arguments: ${cmake_args[*]}"
  cmake "${cmake_args[@]}" ../.. || { echo "CMake configuration failed"; exit 1; }

  echo "Building benchmarks..."

  cmake --build . --target "${build_targets[@]}" --parallel "$(get_cpu_count)" || { echo "Build failed"; exit 1; }
}

run_benchmarks() {
  local -a targets=("${!1}")
  local slurm_execution="$2"
  local output_dir="$3"
  local use_simde="$4"

  for t in "${targets[@]}"; do
    local benchmark_executable="src/bench_$t"
    if [ ! -x "$benchmark_executable" ]; then
      echo "Benchmark executable $benchmark_executable not found or not executable."
      exit 1
    fi

    local output_target="$t"
    if [ "$use_simde" -eq 1 ] && [[ "$t" == pernix_* ]]; then
      output_target="${t}_simde"
    fi
    local output_file="${output_dir}/benchmark_${output_target}_results.json"

    echo "Running benchmark target: $t"
    if [ "$slurm_execution" -eq 1 ]; then
      if [ -n "$PIN" ]; then
        srun --exclusive likwid-pin -c "$PIN" ./"$benchmark_executable" --benchmark_out="$output_file" --benchmark_out_format=json || { echo "Benchmark $t failed"; exit 1; }
      else
        srun ./"$benchmark_executable" --benchmark_out="$output_file" --benchmark_out_format=json || { echo "Benchmark $t failed"; exit 1; }
      fi
    else
      if [ -n "$PIN" ]; then
        likwid-pin -c "$PIN" ./"$benchmark_executable" --benchmark_out="$output_file" --benchmark_out_format=json || { echo "Benchmark $t failed"; exit 1; }
      else
        ./"$benchmark_executable" --benchmark_out="$output_file" --benchmark_out_format=json || { echo "Benchmark $t failed"; exit 1; }
      fi
    fi
    echo "Benchmark for target $t completed. Results saved to $output_file"
  done
}

setup_python_environment() {
  if ! command -v python3 &> /dev/null; then
    echo "python3 could not be found, please install it."
    exit 1
  fi

  if ! python3 -m venv .venv; then
    echo "Failed to create Python virtual environment."
    exit 1
  fi

  # shellcheck disable=SC1091
  source .venv/bin/activate

  if ! pip install --upgrade pip; then
    echo "Failed to upgrade pip."
    exit 1
  fi

  if ! pip install machinestate; then
    echo "Failed to install required Python packages."
    exit 1
  fi
}

collect_machine_information() {
  local tmp_output="$1"

  if command -v machinestate &> /dev/null; then
    echo "Collecting machine state information..."
    machinestate -e -o "$tmp_output/machinestate.json" || { echo "Failed to collect machine state information"; exit 1; }
  else
    echo "machinestate command not found, skipping machine information collection."
  fi

  if command -v likwid-topology &> /dev/null; then
    echo "Collecting CPU topology information with likwid-topology..."
    likwid-topology -g > "$tmp_output/likwid_topology.txt" || { echo "Failed to collect CPU topology information"; exit 1; }
  else
    echo "likwid-topology command not found, skipping CPU topology collection."
  fi
}

COMPILER="g++"
RELEASE_TYPE="RELEASE"
TARGETS=("${AVAILABLE_TARGETS[@]}")
FINAL_OUTPUT="benchmark_results.tar.gz"
INSTALL_PC2_MODULES=0
SLURM_EXECUTION=0
USE_SIMDE=0
PIN=""
CLEAN_BUILD=0
if [ -n "$SLURM_JOB_ID" ]; then
  SLURM_EXECUTION=1
  echo "Detected SLURM environment. SLURM execution mode enabled."
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    -c|--compiler)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument"; exit 2; fi
      COMPILER="$2"
      shift 2
      ;;
    --compiler=*)
      COMPILER="${1#*=}"
      shift
      ;;
    -t|--target)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument"; exit 2; fi
      parse_target_list "$2"
      shift 2
      ;;
    --target=*)
      parse_target_list "${1#*=}"
      shift
      ;;
    --release-type)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument"; exit 2; fi
      RELEASE_TYPE="$2"
      shift 2
      ;;
    --release-type=*)
      RELEASE_TYPE="${1#*=}"
      shift
      ;;
    -o|--output)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument"; exit 2; fi
      FINAL_OUTPUT="$2"
      shift 2
      ;;
    --output=*)
      FINAL_OUTPUT="${1#*=}"
      shift
      ;;
    --list-targets) print_available_targets; exit 0 ;;
    -s|--slurm) SLURM_EXECUTION=1; shift ;;
    -m|--modules) INSTALL_PC2_MODULES=1; shift ;;
    --simde) USE_SIMDE=1; shift ;;
    --pin)
      if [ "$#" -lt 2 ]; then echo "$1 requires an argument"; exit 2; fi
      PIN="$2"
      shift 2
      ;;
    --pin=*)
      PIN="${1#*=}"
      shift
      ;;
    --clean) CLEAN_BUILD=1; shift ;;
    -h|--help) print_help; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1"; print_help; exit 2 ;;
  esac
done

# Load modules if requested or if running under SLURM
if [ "$INSTALL_PC2_MODULES" -eq 1 ] || [ "$SLURM_EXECUTION" -eq 1 ]; then
  module reset
  module load tools/likwid/5.4.1_zen5
  module load lang/Python/3.13.5-GCCcore-14.3.0
  module load devel/CMake/4.0.3-GCCcore-14.3.0
  module load tools/googlebenchmark/1.9.4-GCCcore-14.3.0
  module load tools/googletest/1.17.0-GCCcore-14.3.0
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

# Check for required tools: cmake and the specified compiler
if ! command -v cmake &> /dev/null; then
  echo "cmake could not be found, please install it."
  exit 1
fi
if ! command -v "$COMPILER" &> /dev/null; then
  echo "$COMPILER could not be found, please install it."
  exit 1
fi

WORK_DIR=$(mktemp -d)
echo "Using temporary working directory: $WORK_DIR"

if [ -n "$FINAL_OUTPUT" ]; then
  # Ensure the directory for the final output exists
  mkdir -p "$(dirname "$FINAL_OUTPUT")"
  # Resolve absolute path for final output
  if [[ "$FINAL_OUTPUT" != /* ]]; then
    FINAL_OUTPUT="$(pwd)/$FINAL_OUTPUT"
  fi
  echo "Final results will be archived to: $FINAL_OUTPUT"
fi
trap 'rm -rf "$WORK_DIR"' EXIT

# Print configuration
echo "Benchmark configuration:"
echo "  Compiler: $COMPILER"
echo "  Release Type: $RELEASE_TYPE"
echo "  Targets: ${TARGETS[*]}"
echo "  Working Directory: $WORK_DIR"
if [ -n "$FINAL_OUTPUT" ]; then
  echo "  Final Output: $FINAL_OUTPUT"
fi
echo "  SLURM Execution: $SLURM_EXECUTION"
echo "  SIMDe: $USE_SIMDE"
echo "  Clean: $CLEAN_BUILD"

BUILD_DIR="build/benchmarks_${COMPILER}"
if [ -n "$RELEASE_TYPE" ]; then
  BUILD_DIR="${BUILD_DIR}_${RELEASE_TYPE}"
fi
if [ "$USE_SIMDE" -eq 1 ]; then
  BUILD_DIR="${BUILD_DIR}_simde"
fi

if [ "$CLEAN_BUILD" -eq 1 ]; then
  echo "Cleaning build directory and Python virtual environment..."
  rm -rf "$BUILD_DIR" .venv
fi

setup_python_environment
collect_machine_information "$WORK_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || { echo "Failed to enter build directory"; exit 1; }

build_benchmarks "$COMPILER" "$RELEASE_TYPE" TARGETS[@] "$SLURM_EXECUTION" "$USE_SIMDE"
run_benchmarks TARGETS[@] "$SLURM_EXECUTION" "$WORK_DIR" "$USE_SIMDE"

if [ -n "$FINAL_OUTPUT" ]; then
  echo "Archiving results..."
  tar -czf "$FINAL_OUTPUT" -C "$WORK_DIR" . || { echo "Failed to create archive"; exit 1; }
  echo "Results archived to $FINAL_OUTPUT"
fi
