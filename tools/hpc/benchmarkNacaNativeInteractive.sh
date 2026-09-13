#!/usr/bin/env bash

set -Eeuo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "Run this script inside a Roihu CPU interactive allocation" >&2
    exit 2
}

repo=${FOAMNORDIC_REPO:-/scratch/project_2015384/Hanseul/Source/FoamNordic}

echo "=== NACA4412 paired performance ==="
bash "$repo/tools/hpc/benchmarkNacaInteractive.sh"

echo "=== NACA4412 numerical trajectory ==="
bash "$repo/tools/hpc/validateNacaClosure.sh"
