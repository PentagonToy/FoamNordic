#!/usr/bin/env bash

set -Eeuo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Run inside a Roihu CPU interactive allocation" >&2; exit 2; }

repo=${FOAMNORDIC_REPO:-/scratch/project_2015384/Hanseul/Source/FoamNordic}
venv=${FOAMNORDIC_VENV:-/scratch/project_2015384/Hanseul/Utilities/Python/x86_64/envs/PentagonToy-3.12}
source_case=${FOAMNORDIC_SMAGORINSKY_CASE:-/scratch/project_2015384/Hanseul/Codes/FoamNordic/openfoam_tutorials/incompressible/les/pitzDailySmag}
output=${FOAMNORDIC_BENCHMARK_OUTPUT:-/scratch/project_2015384/Hanseul/Tests/openfoam-benchmarks}
ranks=${FOAMNORDIC_BENCHMARK_RANKS:-16}
allocated=${SLURM_CPUS_PER_TASK:-0}

[[ -x "$venv/bin/python" ]] || { echo "Python environment not found: $venv" >&2; exit 2; }
[[ -f "$source_case/system/controlDict" ]] || { echo "Smagorinsky source case not found: $source_case" >&2; exit 2; }
[[ "$allocated" =~ ^[1-9][0-9]*$ ]] && (( ranks <= allocated )) || { echo "Requested $ranks ranks with $allocated allocated CPUs" >&2; exit 2; }

module load openfoam/2512
export FOAMNORDIC_MPIRUN="$repo/tools/hpc/mpirunSlurmLocal.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "Host: $(hostname)"
echo "Allocation: $SLURM_JOB_ID"
echo "MPI ranks: $ranks of $allocated allocated CPUs"
echo "FoamNordic commit: $(git -C "$repo" rev-parse HEAD)"
echo "OpenFOAM: ${WM_PROJECT_VERSION:-unknown}"
echo "Source case: $source_case"
echo "Output root: $output"

"$venv/bin/python" "$repo/tools/hpc/benchmarkSmagorinskyInteractive.py" \
  --source "$source_case" --output "$output" --ranks "$ranks" \
  --repetitions "${FOAMNORDIC_BENCHMARK_REPETITIONS:-7}" \
  --end-time "${FOAMNORDIC_BENCHMARK_END_TIME:-4e-3}" \
  --timeout "${FOAMNORDIC_BENCHMARK_TIMEOUT:-3600}"
