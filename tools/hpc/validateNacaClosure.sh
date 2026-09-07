#!/usr/bin/env bash

set -Eeuo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "Run this script inside a Roihu CPU interactive allocation" >&2
    exit 2
}

repo=${FOAMNORDIC_REPO:-/scratch/project_2015384/Hanseul/Source/FoamNordic}
venv=${FOAMNORDIC_VENV:-/scratch/project_2015384/Hanseul/Utilities/Python/x86_64/envs/PentagonToy-3.12}
source_case=${FOAMNORDIC_NACA_CASE:-/scratch/project_2015384/Hanseul/Codes/FoamNordic/openfoam_tutorials/incompressible/les/NACA4412}
output=${FOAMNORDIC_BENCHMARK_OUTPUT:-/scratch/project_2015384/Hanseul/Tests/openfoam-benchmarks}
ranks=${FOAMNORDIC_BENCHMARK_RANKS:-16}

[[ -x "$venv/bin/python" ]] || {
    echo "Python environment not found: $venv" >&2
    exit 2
}
[[ -f "$source_case/system/controlDict" ]] || {
    echo "NACA4412 source case not found: $source_case" >&2
    exit 2
}

module load openfoam/2512

allocated_cpus=${SLURM_CPUS_PER_TASK:-0}
[[ "$allocated_cpus" =~ ^[1-9][0-9]*$ ]] || {
    echo "SLURM_CPUS_PER_TASK is unavailable or invalid: ${allocated_cpus:-unset}" >&2
    exit 2
}
(( ranks <= allocated_cpus )) || {
    echo "Requested $ranks MPI ranks but this allocation provides $allocated_cpus CPUs" >&2
    exit 2
}

export FOAMNORDIC_MPIRUN="$repo/tools/hpc/mpirunSlurmLocal.sh"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Host: $(hostname)"
echo "Allocation: $SLURM_JOB_ID"
echo "MPI ranks: $ranks of $allocated_cpus allocated CPUs"
echo "FoamNordic commit: $(git -C "$repo" rev-parse HEAD)"
echo "OpenFOAM: ${WM_PROJECT_VERSION:-unknown}"
echo "Checkpoints: 1, 10, 100, 400 time steps"
echo "Output root: $output"

"$venv/bin/python" "$repo/tools/hpc/validateNacaClosure.py" \
    --source "$source_case" \
    --output "$output" \
    --ranks "$ranks" \
    --delta-t "${FOAMNORDIC_NACA_DELTA_T:-0.0025}" \
    --steps 1 10 100 400 \
    --timeout "${FOAMNORDIC_BENCHMARK_TIMEOUT:-3600}"
