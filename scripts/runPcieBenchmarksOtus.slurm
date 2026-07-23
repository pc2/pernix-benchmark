#!/usr/bin/env bash

#SBATCH --job-name=pernix-pcie-benchmarks
#SBATCH --account=hpc-prf-tenstorrent
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "This script must be submitted with sbatch." >&2
    exit 1
fi

REPOSITORY_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_PROJECT="${REPOSITORY_ROOT}/benchmark-tools"
BUILD_DIR="${TMPDIR:-/tmp}/pernix-pcie-build-${SLURM_JOB_ID}"
RESULT_DIR="${REPOSITORY_ROOT}/benchmark-results/pcie-${SLURM_JOB_ID}"

if [[ ! -f "${REPOSITORY_ROOT}/CMakeLists.txt" || ! -d "${PYTHON_PROJECT}" ]]; then
    echo "Submit this script from the pernix-benchmark repository root." >&2
    exit 1
fi

module reset
module load tools/googlebenchmark/1.9.4-GCCcore-14.3.0
module load devel/CMake/4.2.1-GCCcore-15.2.0
module load lang/Python/3.14.2-GCCcore-15.2.0
module load compiler/GCC/15.2.0
module load CUDA

cleanup() { rm -rf -- "${BUILD_DIR}"; }
trap cleanup EXIT

cd "${PYTHON_PROJECT}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
    python -m uv run --frozen --no-sync pernix-bench run \
    --jobs "${SLURM_CPUS_PER_TASK}" --build-dir "${BUILD_DIR}" \
    --output-dir "${RESULT_DIR}" --clean pcie "$@"
