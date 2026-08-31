#!/usr/bin/env bash
set -Eeuo pipefail

: "${FOAMNORDIC_SLURM_ACCOUNT:?Set FOAMNORDIC_SLURM_ACCOUNT to your allocation account}"
: "${FOAMNORDIC_REPO:?Set FOAMNORDIC_REPO to the FoamNordic checkout}"
: "${FOAMNORDIC_VENV:?Set FOAMNORDIC_VENV to the FoamNordic virtual environment}"
: "${FOAMNORDIC_HPC_GATE_ROOT:?Set FOAMNORDIC_HPC_GATE_ROOT to the gate bundle}"

BUNDLE=$(cd "$FOAMNORDIC_HPC_GATE_ROOT" && pwd)
mkdir -p "$BUNDLE/slurm"

JOB_ID=$(
    sbatch \
        --parsable \
        --account="$FOAMNORDIC_SLURM_ACCOUNT" \
        --partition="${FOAMNORDIC_SLURM_PARTITION:-small}" \
        --time="${FOAMNORDIC_SLURM_TIME:-00:20:00}" \
        --nodes=1 \
        --ntasks="${FOAMNORDIC_MPI_RANKS:-2}" \
        --cpus-per-task=1 \
        --mem-per-cpu="${FOAMNORDIC_MEM_PER_CPU:-4G}" \
        --job-name=fn-combustion \
        --output="$BUNDLE/slurm/combustion-%j.out" \
        --export=ALL \
        "$FOAMNORDIC_REPO/tools/hpc/runCombustionGates.sh"
)
JOB_ID=${JOB_ID%%;*}

echo "[FoamNordic] Submitted combustion gates: $JOB_ID"
echo "[FoamNordic] Slurm output: $BUNDLE/slurm/combustion-$JOB_ID.out"
squeue --job="$JOB_ID" --format='%.18i %.12P %.24j %.10T %.10M %.20R'
