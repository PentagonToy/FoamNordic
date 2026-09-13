#!/usr/bin/env bash

set -Eeuo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "Run this script through Slurm" >&2
    exit 2
}

repo=${FOAMNORDIC_REPO:-/scratch/project_2015384/Hanseul/Source/FoamNordic}
venv=${FOAMNORDIC_VENV:-/scratch/project_2015384/Hanseul/Utilities/Python/x86_64/envs/PentagonToy-3.12}
ranks=${FOAMNORDIC_BENCHMARK_RANKS:-16}

module load openfoam/2512
export PYTHONPATH="$repo/packages/foamnordic${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Build current FoamNordic checkout ==="
"$venv/bin/foamnordic" build --jobs "$ranks"

echo "=== nutFjord Smagorinsky performance ==="
bash "$repo/tools/hpc/benchmarkSmagorinskyInteractive.sh"

echo "=== nutFjord WALE performance ==="
bash "$repo/tools/hpc/benchmarkWaleInteractive.sh"

echo "=== nutFjord numerical trajectories ==="
bash "$repo/tools/hpc/validateNutClosure.sh"

echo "=== kEqnFjord native performance and trajectory ==="
bash "$repo/tools/hpc/benchmarkNacaNativeInteractive.sh"

echo "[FoamNordic] OpenFOAM benchmark suite: PASS"
