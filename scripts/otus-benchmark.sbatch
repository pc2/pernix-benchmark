#!/usr/bin/env bash

#SBATCH --job-name=pernix-benchmarks
#SBATCH --account=hpc-prf-tenstorrent
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "This script must be submitted with sbatch." >&2
    exit 1
fi

JOB_MODE="${1:-}"

case "${JOB_MODE}" in
    benchmark | model) ;;
    *)
        echo "Usage: sbatch [options] $0 {benchmark|model}" >&2
        exit 2
        ;;
esac

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
    echo "LLVM-MCA 20 was not found. Load LLVM 20 or set LLVM_MCA." >&2
    exit 1
fi

if ! command -v "${LIKWID_PERFCTR}" >/dev/null 2>&1; then
    echo "likwid-perfctr was not found after loading the LIKWID module." >&2
    exit 1
fi

if ! python -m uv --version >/dev/null 2>&1; then
    echo "The loaded Python cannot import uv." >&2
    echo "Install it before submitting with:" >&2
    echo "  python -m pip install --user uv" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_PROJECT}/.venv/bin/python" ]]; then
    echo "The uv environment is missing." >&2
    echo "Run this on an Otus login node first:" >&2
    echo "  cd benchmark-tools && python -m uv sync --frozen" >&2
    exit 1
fi

BUILD_BASE="${TMPDIR:-/tmp}"
BENCHMARK_MIN_TIME_SECONDS="${PERNIX_BENCHMARK_MIN_TIME_SECONDS:-0.25}"
BENCHMARK_REPETITIONS="${PERNIX_BENCHMARK_REPETITIONS:-5}"
BENCHMARK_MIN_WARMUP_TIME_SECONDS="${PERNIX_BENCHMARK_MIN_WARMUP_TIME_SECONDS:-0.05}"

if [[ "${JOB_MODE}" == "benchmark" ]]; then
    if [[ -z "${SLURM_ARRAY_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        echo "Benchmark mode must be submitted as a job array." >&2
        exit 1
    fi

    RESULT_SET_ID="${SLURM_ARRAY_JOB_ID}"
    BUILD_DIR="${BUILD_BASE}/pernix-benchmark-build-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

    case "${SLURM_ARRAY_TASK_ID}" in
        0)
            TASK_LABEL="AVX2"
            TASK_NAME="pernix_avx2"
            TASK_ARGUMENTS=(pernix avx2)
            ;;
        1)
            TASK_LABEL="AVX2+BMI2"
            TASK_NAME="pernix_bmi2"
            TASK_ARGUMENTS=(pernix bmi2)
            ;;
        2)
            TASK_LABEL="AVX-512-VBMI"
            TASK_NAME="pernix_avx512vbmi"
            TASK_ARGUMENTS=(pernix avx512vbmi)
            ;;
        3)
            TASK_LABEL="CP2K"
            TASK_NAME="cp2k"
            TASK_ARGUMENTS=(cp2k)
            ;;
        *)
            echo "Unsupported array task: ${SLURM_ARRAY_TASK_ID}" >&2
            exit 2
            ;;
    esac
else
    if [[ -z "${PERNIX_BENCHMARK_ARRAY_JOB_ID:-}" ]]; then
        echo "PERNIX_BENCHMARK_ARRAY_JOB_ID is missing for model mode." >&2
        exit 1
    fi

    RESULT_SET_ID="${PERNIX_BENCHMARK_ARRAY_JOB_ID}"
    BUILD_DIR="${BUILD_BASE}/pernix-benchmark-model-build-${SLURM_JOB_ID}"
    TASK_LABEL="instruction model"
    TASK_NAME="model"
    TASK_ARGUMENTS=()
fi

RESULT_DIR="${REPOSITORY_ROOT}/benchmark-results/otus-${RESULT_SET_ID}"
export PERNIX_MACHINE_STATE_FILENAME="machinestate_${TASK_NAME}.json"

cleanup() {
    rm -rf -- "${BUILD_DIR}"
}
trap cleanup EXIT

mkdir -p -- "${RESULT_DIR}"

echo "Mode:                  ${JOB_MODE}"
echo "Task:                  ${TASK_LABEL}"
echo "Slurm job:             ${SLURM_JOB_ID}"
echo "Result-set job ID:     ${RESULT_SET_ID}"
echo "Repository:            ${REPOSITORY_ROOT}"
echo "Build directory:       ${BUILD_DIR}"
echo "Result directory:      ${RESULT_DIR}"
echo "Allocated CPUs:        ${SLURM_CPUS_PER_TASK}"

if [[ "${JOB_MODE}" == "benchmark" ]]; then
    echo "Array task:            ${SLURM_ARRAY_TASK_ID}/3"
    echo "Benchmark min time:    ${BENCHMARK_MIN_TIME_SECONDS} s"
    echo "Benchmark repetitions: ${BENCHMARK_REPETITIONS}"
    echo "Benchmark warm-up:     ${BENCHMARK_MIN_WARMUP_TIME_SECONDS} s"
fi

cd -- "${PYTHON_PROJECT}"

if [[ "${JOB_MODE}" == "model" ]]; then
    command=(
        python -m uv run
        --frozen
        --no-sync
        pernix-bench model
        --results-dir "${RESULT_DIR}"
        --build-dir "${BUILD_DIR}"
        --llvm-cpu "${PERNIX_LLVM_CPU}"
    )
else
    command=(
        python -m uv run
        --frozen
        --no-sync
        pernix-bench run
        --jobs "${SLURM_CPUS_PER_TASK}"
        --build-dir "${BUILD_DIR}"
        --output-dir "${RESULT_DIR}"
        --benchmark-min-time "${BENCHMARK_MIN_TIME_SECONDS}"
        --benchmark-repetitions "${BENCHMARK_REPETITIONS}"
        --benchmark-min-warmup-time "${BENCHMARK_MIN_WARMUP_TIME_SECONDS}"
        --clean
        "${TASK_ARGUMENTS[@]}"
    )
fi

srun \
    --exclusive \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --cpu-bind=cores \
    "${command[@]}"
