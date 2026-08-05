#!/usr/bin/env bash

#SBATCH --job-name=pernix-benchmarks
#SBATCH --account=hpc-prf-tenstorrent
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "This script must be submitted with sbatch." >&2
    exit 1
fi

REPOSITORY_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_PROJECT="${REPOSITORY_ROOT}/benchmark-tools"

if [[ ! -f "${REPOSITORY_ROOT}/CMakeLists.txt" || ! -f "${PYTHON_PROJECT}/pyproject.toml" ]]; then
    echo "Submit this script from the pernix-benchmark repository root." >&2
    exit 1
fi

module reset
module load compiler/GCC/14.3.0
module load compiler/LLVM/20.1.8-GCCcore-14.3.0
module load tools/googlebenchmark/1.9.4-GCCcore-14.3.0
module load tools/likwid/5.5.2rc2-GCC-14.3.0
module load devel/CMake/4.0.3-GCCcore-14.3.0
module load lang/Python/3.13.5-GCCcore-14.3.0


export LLVM_MCA="${LLVM_MCA:-llvm-mca}"
export LIKWID_PERFCTR="${LIKWID_PERFCTR:-likwid-perfctr}"
export PERNIX_LLVM_CPU="${PERNIX_LLVM_CPU:-znver5}"
if ! command -v "${LLVM_MCA}" >/dev/null 2>&1; then
    echo "LLVM-MCA 20 was not found. Load the installed LLVM 20 module or set LLVM_MCA." >&2
    exit 1
fi
if ! command -v "${LIKWID_PERFCTR}" >/dev/null 2>&1; then
    echo "likwid-perfctr was not found after loading tools/likwid/5.5.2rc2-GCC-14.3.0." >&2
    exit 1
fi

if ! python -m pip show uv >/dev/null 2>&1; then
    echo "Installing uv in the user site-packages directory."
    python -m pip install --user uv
fi

if ! python -m uv --version >/dev/null 2>&1; then
    echo "The loaded Python cannot import uv. Install uv before submitting." >&2
    exit 1
fi

if [[ ! -x "${PYTHON_PROJECT}/.venv/bin/python" ]]; then
    echo "The uv environment is missing. Run 'python -m uv sync --frozen' in benchmark-tools on an Otus login node first." >&2
    exit 1
fi

BUILD_BASE="${TMPDIR:-/tmp}"
BUILD_DIR="${BUILD_BASE}/pernix-benchmark-build-${SLURM_JOB_ID}"
RESULT_DIR="${REPOSITORY_ROOT}/benchmark-results/otus-${SLURM_JOB_ID}"
BENCHMARK_MIN_TIME_SECONDS="${PERNIX_BENCHMARK_MIN_TIME_SECONDS:-0.25}"
BENCHMARK_REPETITIONS="${PERNIX_BENCHMARK_REPETITIONS:-5}"
BENCHMARK_MIN_WARMUP_TIME_SECONDS="${PERNIX_BENCHMARK_MIN_WARMUP_TIME_SECONDS:-0.05}"

cleanup() {
    rm -rf -- "${BUILD_DIR}"
}
trap cleanup EXIT

echo "Otus benchmark job ${SLURM_JOB_ID}"
echo "Repository: ${REPOSITORY_ROOT}"
echo "Build directory: ${BUILD_DIR}"
echo "Result directory: ${RESULT_DIR}"
echo "Build CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Benchmark minimum time: ${BENCHMARK_MIN_TIME_SECONDS} s per repetition"
echo "Benchmark repetitions: ${BENCHMARK_REPETITIONS}"
echo "Benchmark minimum warm-up: ${BENCHMARK_MIN_WARMUP_TIME_SECONDS} s"

cd "${PYTHON_PROJECT}"

command=(
    python -m uv run --frozen --no-sync pernix-bench run
    --jobs "${SLURM_CPUS_PER_TASK}"
    --build-dir "${BUILD_DIR}"
    --output-dir "${RESULT_DIR}"
    --benchmark-min-time "${BENCHMARK_MIN_TIME_SECONDS}"
    --benchmark-repetitions "${BENCHMARK_REPETITIONS}"
    --benchmark-min-warmup-time "${BENCHMARK_MIN_WARMUP_TIME_SECONDS}"
    --clean
)
command+=("$@")

srun \
    --exclusive \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --cpu-bind=cores \
    "${command[@]}"
