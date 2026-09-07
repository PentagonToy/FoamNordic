#!/usr/bin/env bash

set -Eeuo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Run inside a Roihu CPU interactive allocation" >&2; exit 2; }

repo=${FOAMNORDIC_REPO:-/scratch/project_2015384/Hanseul/Source/FoamNordic}
venv=${FOAMNORDIC_VENV:-/scratch/project_2015384/Hanseul/Utilities/Python/x86_64/envs/PentagonToy-3.12}
cases=${FOAMNORDIC_TUTORIALS:-/scratch/project_2015384/Hanseul/Codes/FoamNordic/openfoam_tutorials/incompressible/les}
output=${FOAMNORDIC_BENCHMARK_OUTPUT:-/scratch/project_2015384/Hanseul/Tests/openfoam-benchmarks}
ranks=${FOAMNORDIC_BENCHMARK_RANKS:-16}
allocated=${SLURM_CPUS_PER_TASK:-0}

[[ -x "$venv/bin/python" ]] || { echo "Python environment not found: $venv" >&2; exit 2; }
[[ "$allocated" =~ ^[1-9][0-9]*$ ]] && (( ranks <= allocated )) || { echo "Requested $ranks ranks with $allocated allocated CPUs" >&2; exit 2; }

module load openfoam/2512
export FOAMNORDIC_MPIRUN="$repo/tools/hpc/mpirunSlurmLocal.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "Host: $(hostname)"
echo "Allocation: $SLURM_JOB_ID"
echo "MPI ranks: $ranks of $allocated allocated CPUs"
echo "Models: Smagorinsky, WALE"
echo "Checkpoints: 1, 10, 100, 400 time steps"

"$venv/bin/python" "$repo/tools/hpc/validateNutClosure.py" \
  --smagorinsky-source "$cases/pitzDailySmag" \
  --wale-source "$cases/pitzDailyWALE" \
  --output "$output" --ranks "$ranks" --delta-t 1e-5 \
  --steps 1 10 100 400 --timeout "${FOAMNORDIC_BENCHMARK_TIMEOUT:-3600}"
