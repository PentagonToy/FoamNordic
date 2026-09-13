#!/usr/bin/env bash

set -Eeuo pipefail

repo=${FOAMNORDIC_REPO:-/scratch/project_2015384/Hanseul/Source/FoamNordic}
output=${FOAMNORDIC_BENCHMARK_OUTPUT:-/scratch/project_2015384/Hanseul/Tests/openfoam-benchmarks}
account=${FOAMNORDIC_SLURM_ACCOUNT:-project_2015384}

mkdir -p "$output/slurm"

job_id=$(
    sbatch \
        --parsable \
        --account="$account" \
        --partition=interactive \
        --time="${FOAMNORDIC_SLURM_TIME:-02:00:00}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="${FOAMNORDIC_BENCHMARK_RANKS:-16}" \
        --mem="${FOAMNORDIC_SLURM_MEMORY:-32G}" \
        --job-name=fn-openfoam-benchmarks \
        --output="$output/slurm/openfoam-benchmarks-%j.log" \
        --export=ALL,FOAMNORDIC_REPO="$repo",FOAMNORDIC_BENCHMARK_OUTPUT="$output" \
        "$repo/tools/hpc/runOpenfoamBenchmarks.sh"
)
job_id=${job_id%%;*}

echo "[FoamNordic] Submitted OpenFOAM benchmarks: $job_id"
echo "[FoamNordic] Slurm log: $output/slurm/openfoam-benchmarks-$job_id.log"
