#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" && "${FOAMNORDIC_ALLOW_LOGIN:-false}" != true ]]; then
    echo "[FoamNordic] Run this baseline inside an interactive or sbatch allocation." >&2
    exit 2
fi

: "${FOAMNORDIC_HPC_GATE_ROOT:?Set FOAMNORDIC_HPC_GATE_ROOT to the gate bundle}"

module load openfoam/2512

BUNDLE=$(cd "$FOAMNORDIC_HPC_GATE_ROOT" && pwd)
SOURCE=${FOAMNORDIC_COMBUSTION_CASE:-$BUNDLE/cases/counterFlowFlame2DLTS_GRI_TDAC}
END_TIME=${FOAMNORDIC_PURE_END_TIME:-1}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="$BUNDLE/results/${SLURM_JOB_ID:-interactive}-$STAMP-pure-openfoam"
CASE="$RUN_ROOT/case"

if [[ ! -f "$SOURCE/system/controlDict" ]]; then
    echo "[FoamNordic] OpenFOAM source case is unavailable: $SOURCE" >&2
    exit 2
fi
if ! command -v reactingFoam >/dev/null 2>&1; then
    echo "[FoamNordic] reactingFoam is unavailable after loading OpenFOAM." >&2
    exit 2
fi

mkdir -p "$RUN_ROOT"
cp -a "$SOURCE" "$CASE"

CONTROL="$CASE/system/controlDict"
foamDictionary "$CONTROL" -entry application -set reactingFoam
foamDictionary "$CONTROL" -entry startFrom -set startTime
foamDictionary "$CONTROL" -entry startTime -set 0
foamDictionary "$CONTROL" -entry endTime -set "$END_TIME"
foamDictionary "$CONTROL" -entry writeControl -set timeStep
foamDictionary "$CONTROL" -entry writeInterval -set 1

blockMesh -case "$CASE" >"$RUN_ROOT/blockMesh.log" 2>&1
checkMesh -case "$CASE" >"$RUN_ROOT/checkMesh.log" 2>&1

echo "[FoamNordic] Pure OpenFOAM baseline: reactingFoam"
echo "[FoamNordic] OpenFOAM: ${WM_PROJECT_VERSION:-unknown}"
echo "[FoamNordic] Allocation: ${SLURM_JOB_ID:-interactive} on $(hostname)"
echo "[FoamNordic] End time: $END_TIME"

START_NS=$(date +%s%N)
set +e
reactingFoam -case "$CASE" >"$RUN_ROOT/reactingFoam.log" 2>&1
status=$?
set -e
if [[ "$status" -ne 0 ]]; then
    tail -80 "$RUN_ROOT/reactingFoam.log" >&2
    echo "[FoamNordic] Pure OpenFOAM baseline: FAIL ($status)" >&2
    exit "$status"
fi
END_NS=$(date +%s%N)
ELAPSED=$(awk -v start="$START_NS" -v end="$END_NS" \
    'BEGIN { printf "%.6f", (end - start) / 1000000000 }')

: >"$RUN_ROOT/fieldMinMax.log"
for field in U p T CH4 O2 CO2 H2O N2; do
    {
        echo "=== $field ==="
        postProcess -case "$CASE" -latestTime -func "fieldMinMax($field)"
    } >>"$RUN_ROOT/fieldMinMax.log" 2>&1 || {
        echo "[FoamNordic] fieldMinMax unavailable for $field" \
            >>"$RUN_ROOT/fieldMinMax.log"
    }
done

LATEST_TIME=$(foamListTimes -case "$CASE" -latestTime 2>/dev/null | tail -1)
cat >"$RUN_ROOT/baseline.env" <<EOF
schema=foamnordic.pure-openfoam-baseline/v1
application=reactingFoam
openfoam=${WM_PROJECT_VERSION:-unknown}
job=${SLURM_JOB_ID:-interactive}
node=$(hostname)
end_time=$END_TIME
latest_time=$LATEST_TIME
elapsed_seconds=$ELAPSED
source=$SOURCE
case=$CASE
passed=true
completed_utc=$STAMP
EOF
touch "$RUN_ROOT/PASS"

echo "[FoamNordic] Pure OpenFOAM baseline: PASS"
echo "[FoamNordic] Elapsed: ${ELAPSED}s"
echo "[FoamNordic] Results: $RUN_ROOT"
