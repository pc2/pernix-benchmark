#!/usr/bin/env bash

#SBATCH -J "Pernix and CP2K Number De/Compression Benchmarks"
#SBATCH -A hpc-prf-ecderi
#SBATCH -t 01:00:00
#SBATCH -N 1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclusive

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
      --list_targets          List all available benchmark targets
  -c, --compiler COMPILER     Specify compiler to use (e.g. gcc, clang)
      --release_type TYPE     Specify release type (e.g. Debug, Release)
  -s, --slurm                 Run benchmarks using SLURM job scheduler
  -m, --modules               Load required modules for PC2 environment
  -h, --help                  Show this help message and exit
USAGE
}

AVAILABLE_TARGETS=("cp2k") # "pernix_fallback" "pernix_avx2" "pernix_bmi2" "pernix_avx512vbmi")

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

patch_cp2k() {
  local cp2k_source_dir="external/cp2k"
  local patch_file="external/cp2k-patches/0001-Delete-unneeded-types-hfx_types.F.patch"
  local patch_abs
  patch_abs="$(cd "$(dirname "$patch_file")" && pwd -P)/$(basename "$patch_file")"

  echo "Patching CP2K source code..."

  if [ ! -d "$cp2k_source_dir" ]; then
    echo "CP2K source directory not found at $cp2k_source_dir"
    exit 1
  fi

  if [ ! -f "$patch_file" ]; then
    echo "Patch file not found at $patch_file"
    exit 1
  fi

  pushd "$cp2k_source_dir" > /dev/null || { echo "Failed to enter CP2K source directory"; exit 1; }
  git checkout support/v2025.2 || { echo "Failed to checkout CP2K support/v2025.2 branch"; popd > /dev/null || exit; exit 1; }
  if git apply --check "$patch_abs" 2>/dev/null; then
    git apply "$patch_abs" || { echo "Failed to apply patch $patch_file"; popd > /dev/null || exit; exit 1; }
    echo "Applied patch $patch_file successfully."
  else
    echo "Patch $patch_file has already been applied or cannot be applied cleanly."
  fi
  popd > /dev/null || { echo "Failed to return to previous directory"; exit 1; }
}

build_benchmarks() {
  local compiler="$1"
  local release_type="$2"
  local -a targets=("${!3}")
  local slurm_execution="$4"
  local cmake_args=()
  local cmake_target=""

  cmake_args+=("-DCMAKE_CXX_COMPILER=$compiler")
  cmake_args+=("-DCMAKE_BUILD_TYPE=$release_type")

  for t in "${targets[@]}"; do
    if [ -n "$cmake_target" ]; then
      cmake_target+=" "
    fi
    cmake_target+="bench_$t"
  done

  echo "Configuring benchmarks with CMake..."
  echo "  CMake arguments: ${cmake_args[*]}"
  cmake "${cmake_args[@]}" ../.. || { echo "CMake configuration failed"; exit 1; }

  echo "Building benchmarks..."

  make -j"$(nproc)" "${cmake_target}" || { echo "Build failed"; exit 1; }
}

run_benchmarks() {
  local -a targets=("${!1}")
  local slurm_execution="$2"

  for t in "${targets[@]}"; do
    local benchmark_executable="src/bench_$t"
    if [ ! -x "$benchmark_executable" ]; then
      echo "Benchmark executable $benchmark_executable not found or not executable."
      exit 1
    fi

    echo "Running benchmark target: $t"
    if [ "$slurm_execution" -eq 1 ]; then
      srun --exclusive ./"$benchmark_executable" --benchmark_out="benchmark_${t}_results.json" --benchmark_out_format=json || { echo "Benchmark $t failed"; exit 1; }
    else
      ./"$benchmark_executable" --benchmark_out="benchmark_${t}_results.json" --benchmark_out_format=json || { echo "Benchmark $t failed"; exit 1; }
    fi
    echo "Benchmark for target $t completed. Results saved to benchmark_${t}_results.json"
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

collect_machinestate() {
  if ! command -v machinestate &> /dev/null; then
    echo "machinestate could not be found, please ensure it is installed in the Python environment."
    exit 1
  fi

  # machinestate collect --output machinestate.json || { echo "Failed to collect machine state."; exit 1; }
  echo "Machine state collected and saved to machinestate.json"
}

PARSED=$(getopt -o c,t:shm -l compiler:,target:,list_targets,release_type:,slurm,modules:help -n runBenchmarks.sh -- "$@") || { echo "Invalid options"; exit 2; }
eval set -- "$PARSED"
COMPILER="g++"
RELEASE_TYPE="RELEASE"
TARGETS=("${AVAILABLE_TARGETS[@]}")
INSTALL_PC2_MODULES=0
SLURM_EXECUTION=0
if [ -n "$SLURM_JOB_ID" ]; then
  SLURM_EXECUTION=1
  echo "Detected SLURM environment. SLURM execution mode enabled."
fi
while true; do
  case "$1" in
    -c|--compiler) COMPILER="$2"; shift 2 ;;
    -t|--target)
    oldIFS="$IFS"
    IFS=','
    read -r -a TARGETS <<< "$2"
    IFS="$oldIFS"
    validate_targets "${TARGETS[@]}"
    shift 2 ;;
    --release_type) RELEASE_TYPE="$2"; shift 2 ;;
    --list_targets) print_available_targets; exit 0 ;;
    -s|--slurm) SLURM_EXECUTION=1; shift ;;
    -m|--modules) INSTALL_PC2_MODULES=1; shift ;;
    -h|--help) print_help; exit 0 ;;
    --) shift; break ;;
    *) echo "Internal error while parsing options"; exit 3 ;;
  esac
done

# Load modules if requested or if running under SLURM
if [ "$INSTALL_PC2_MODULES" -eq 1 ] || [ "$SLURM_EXECUTION" -eq 1 ]; then
  if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script is set to run with SLURM, but no SLURM job ID is found. Please submit the script via sbatch or srun."
    exit 1
  fi

  module reset
  module load lang/Python/3.13.5-GCCcore-14.3.0
  module load devel/CMake/4.0.3-GCCcore-14.3.0
  module load tools/googlebenchmark/1.9.4-GCCcore-14.3.0
  module load tools/googletest/1.17.0-GCCcore-14.3.0
fi

# Check for required tools: cmake, make, and the specified compiler
if ! command -v cmake &> /dev/null; then
  echo "cmake could not be found, please install it."
  exit 1
fi
if ! command -v "$COMPILER" &> /dev/null; then
  echo "$COMPILER could not be found, please install it."
  exit 1
fi
if ! command -v make &> /dev/null; then
  echo "make could not be found, please install it."
  exit 1
fi

# Print configuration
echo "Benchmark configuration:"
echo "  Compiler: $COMPILER"
echo "  Release Type: $RELEASE_TYPE"
echo "  Targets: ${TARGETS[*]}"
echo "  SLURM Execution: $SLURM_EXECUTION"

setup_python_environment
collect_machinestate

patch_cp2k

BUILD_DIR="build/benchmarks_${COMPILER}"
if [ -n "$RELEASE_TYPE" ]; then
  BUILD_DIR="${BUILD_DIR}_${RELEASE_TYPE}"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || { echo "Failed to enter build directory"; exit 1; }

build_benchmarks "$COMPILER" "$RELEASE_TYPE" TARGETS[@] "$SLURM_EXECUTION"
run_benchmarks TARGETS[@] "$SLURM_EXECUTION"