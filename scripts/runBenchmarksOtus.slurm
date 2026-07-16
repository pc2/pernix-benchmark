#!/usr/bin/env bash

#SBATCH --job-name=pernix-benchmarks
#SBATCH --account=hpc-prf-tenstorrent
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=08:00:00
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
module load lang/Python/3.13.5-GCCcore-14.3.0
module load devel/CMake/4.0.3-GCCcore-14.3.0
module load tools/googlebenchmark/1.9.4-GCCcore-14.3.0

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not available. Install it in your user environment before submitting." >&2
    exit 1
fi

if [[ ! -x "${PYTHON_PROJECT}/.venv/bin/python" ]]; then
    echo "The uv environment is missing. Run 'uv sync --frozen' in benchmark-tools on an Otus login node first." >&2
    exit 1
fi

BUILD_BASE="${TMPDIR:-/tmp}"
BUILD_DIR="${BUILD_BASE}/pernix-benchmark-build-${SLURM_JOB_ID}"
RESULT_DIR="${REPOSITORY_ROOT}/benchmark-results/otus-${SLURM_JOB_ID}"

cleanup() {
    rm -rf -- "${BUILD_DIR}"
}
trap cleanup EXIT

echo "Otus benchmark job ${SLURM_JOB_ID}"
echo "Repository: ${REPOSITORY_ROOT}"
echo "Build directory: ${BUILD_DIR}"
echo "Result directory: ${RESULT_DIR}"
echo "Build CPUs: ${SLURM_CPUS_PER_TASK}"

cd "${PYTHON_PROJECT}"

command=(
    uv run --frozen --no-sync pernix-bench run
    --jobs "${SLURM_CPUS_PER_TASK}"
    --build-dir "${BUILD_DIR}"
    --output-dir "${RESULT_DIR}"
    --clean
)
command+=("$@")

srun \
    --exclusive \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --cpu-bind=cores \
    "${command[@]}"
