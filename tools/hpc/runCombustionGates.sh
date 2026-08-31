#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" && "${FOAMNORDIC_ALLOW_LOGIN:-false}" != true ]]; then
    echo "[FoamNordic] Run this script inside an interactive or sbatch allocation." >&2
    exit 2
fi

: "${FOAMNORDIC_REPO:?Set FOAMNORDIC_REPO to the FoamNordic checkout}"
: "${FOAMNORDIC_VENV:?Set FOAMNORDIC_VENV to the FoamNordic virtual environment}"
: "${FOAMNORDIC_HPC_GATE_ROOT:?Set FOAMNORDIC_HPC_GATE_ROOT to the gate bundle}"

module load openfoam/2512
source "$FOAMNORDIC_VENV/bin/activate"

REPO=$(cd "$FOAMNORDIC_REPO" && pwd)
BUNDLE=$(cd "$FOAMNORDIC_HPC_GATE_ROOT" && pwd)
CASE=${FOAMNORDIC_COMBUSTION_CASE:-$BUNDLE/cases/counterFlowFlame2DLTS_GRI_TDAC}
RANKS=${FOAMNORDIC_MPI_RANKS:-2}

if [[ ! -f "$CASE/system/controlDict" ]]; then
    echo "[FoamNordic] Combustion case is unavailable: $CASE" >&2
    exit 2
fi
if ! [[ "$RANKS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[FoamNordic] FOAMNORDIC_MPI_RANKS must be a positive integer." >&2
    exit 2
fi
if [[ -n "${SLURM_NTASKS:-}" && "$RANKS" -gt "$SLURM_NTASKS" ]]; then
    echo "[FoamNordic] Requested $RANKS ranks but allocation has $SLURM_NTASKS tasks." >&2
    exit 2
fi

cd "$REPO"
echo "[FoamNordic] Commit: $(git rev-parse --short HEAD)"
echo "[FoamNordic] OpenFOAM: ${WM_PROJECT_VERSION:-unknown}"
echo "[FoamNordic] Allocation: ${SLURM_JOB_ID:-local} on $(hostname)"

if [[ "${FOAMNORDIC_SKIP_BUILD:-false}" != true ]]; then
    foamnordic build --source "$REPO"
fi
RUNTIME=$(
    python -c 'from foamnordic.execution.runtime_paths import profile; print(profile().runtime_dir)'
)
if [[ ! -x "$RUNTIME/bin/foamnordicProgressVariableFoam" ]]; then
    echo "[FoamNordic] Progress-variable runtime is unavailable: $RUNTIME" >&2
    exit 2
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="$BUNDLE/results/${SLURM_JOB_ID:-interactive}-$STAMP"
mkdir -p "$RUN_ROOT"
GATE_SET=${FOAMNORDIC_GATE_SET:-all}
if [[ "$GATE_SET" != all && "$GATE_SET" != mpi ]]; then
    echo "[FoamNordic] FOAMNORDIC_GATE_SET must be 'all' or 'mpi'." >&2
    exit 2
fi

run_gate() {
    local name=$1
    shift
    echo "[FoamNordic] Gate: $name"
    "$@" 2>&1 | tee "$RUN_ROOT/$name.log"
}

if [[ "$GATE_SET" == all ]]; then
    run_gate stock-volumetric \
        python tools/openfoam/reactionRateReactingFoam.py \
        "$CASE" \
        --workspace "$RUN_ROOT/stock-volumetric" \
        --end-time 1 \
        --reaction-rate-basis volumetric_mass \
        --timeout 300

    run_gate stock-specific \
        python tools/openfoam/reactionRateReactingFoam.py \
        "$CASE" \
        --workspace "$RUN_ROOT/stock-specific" \
        --end-time 1 \
        --reaction-rate-basis specific \
        --timeout 300

    run_gate progress-serial-volumetric \
        python tools/openfoam/testProgressVariableCombustion.py \
        --source "$CASE" \
        --workspace "$RUN_ROOT/progress-serial-volumetric" \
        --runtime "$RUNTIME" \
        --ranks 1 \
        --reaction-rate-basis volumetric_mass

    run_gate progress-serial-specific \
        python tools/openfoam/testProgressVariableCombustion.py \
        --source "$CASE" \
        --workspace "$RUN_ROOT/progress-serial-specific" \
        --runtime "$RUNTIME" \
        --ranks 1 \
        --reaction-rate-basis specific
fi

run_gate progress-mpi-specific \
    python tools/openfoam/testProgressVariableCombustion.py \
    --source "$CASE" \
    --workspace "$RUN_ROOT/progress-mpi-specific" \
    --runtime "$RUNTIME" \
    --ranks "$RANKS" \
    --reaction-rate-basis specific

cat > "$RUN_ROOT/PASS" <<EOF
FoamNordic combustion HPC gates passed.
commit=$(git rev-parse HEAD)
openfoam=${WM_PROJECT_VERSION:-unknown}
job=${SLURM_JOB_ID:-interactive}
ranks=$RANKS
gate_set=$GATE_SET
completed_utc=$STAMP
EOF

echo "[FoamNordic] Combustion HPC gates: PASS"
echo "[FoamNordic] Results: $RUN_ROOT"
