#!/usr/bin/env bash

set -Eeuo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "mpirunSlurmLocal.sh requires an active Slurm allocation" >&2
    exit 2
}

slots=${SLURM_CPUS_PER_TASK:-}
[[ "$slots" =~ ^[1-9][0-9]*$ ]] || {
    echo "SLURM_CPUS_PER_TASK is unavailable or invalid: ${slots:-unset}" >&2
    exit 2
}

launcher=$(command -v mpirun) || {
    echo "mpirun is unavailable in the active OpenFOAM environment" >&2
    exit 2
}

exec "$launcher" --map-by "slot:OVERSUBSCRIBE" --bind-to none "$@"
