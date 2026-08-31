#!/usr/bin/env bash
set -Eeuo pipefail

: "${FOAMNORDIC_REPO:?Set FOAMNORDIC_REPO to the FoamNordic checkout}"
: "${FOAMNORDIC_VENV:?Set FOAMNORDIC_VENV to the FoamNordic virtual environment}"

cd "$FOAMNORDIC_REPO"
git fetch origin dev
git switch dev
git pull --ff-only origin dev

source "$FOAMNORDIC_VENV/bin/activate"
python -m pip install \
    --no-cache-dir \
    --force-reinstall \
    --no-deps \
    "$FOAMNORDIC_REPO/python"

python - <<'PY'
import foamnordic as fno
import foamnordic._native as native

print(f"[FoamNordic] Python package: {fno.__version__}")
print(f"[FoamNordic] Native module: {native.__file__}")
PY
