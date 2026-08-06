#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/otus-benchmark.sbatch"

if [[ ! -f "${JOB_SCRIPT}" ]]; then
    echo "Job script not found: ${JOB_SCRIPT}" >&2
    exit 1
fi

benchmark_job_id="$(
    sbatch \
        --parsable \
        --array=0-3%4 \
        --exclusive \
        "${JOB_SCRIPT}" \
        benchmark
)"

# Some Slurm configurations return "<job-id>;<cluster-name>".
benchmark_job_id="${benchmark_job_id%%;*}"

if [[ ! "${benchmark_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Invalid benchmark job ID returned by sbatch: ${benchmark_job_id}" >&2
    exit 1
fi

model_job_id="$(
    sbatch \
        --parsable \
        --dependency="afterok:${benchmark_job_id}" \
        --export="ALL,PERNIX_BENCHMARK_ARRAY_JOB_ID=${benchmark_job_id}" \
        --cpus-per-task=1 \
        --mem=8G \
        --time=01:00:00 \
        --output="slurm-%x-%j.out" \
        "${JOB_SCRIPT}" \
        model
)"

model_job_id="${model_job_id%%;*}"

if [[ ! "${model_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Invalid model job ID returned by sbatch: ${model_job_id}" >&2
    exit 1
fi

echo "Benchmark array: ${benchmark_job_id}"
echo "Model job:       ${model_job_id}"
echo
echo "Monitor with:"
echo "  squeue --jobs=${benchmark_job_id},${model_job_id}"
echo
echo "Results:"
echo "  benchmark-results/otus-${benchmark_job_id}"
