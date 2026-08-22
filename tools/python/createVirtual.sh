#!/usr/bin/env bash

set -Eeuo pipefail

REPOSITORY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SOURCE_ROOT=$(dirname "$REPOSITORY")
VIRTUAL_ROOT=${FOAMNORDIC_VIRTUAL_ROOT:-"$SOURCE_ROOT/Virtual"}
ENVIRONMENT=${1:-"$VIRTUAL_ROOT/FoamNordic"}

python_is_supported() {
    local candidate=$1
    env -u PYTHONHOME -u PYTHONPATH "$candidate" - <<'PY' >/dev/null 2>&1
import os
from pathlib import Path
import sys

version_ok = sys.version_info[:2] in {(3, 11), (3, 12)}
prefixes = " ".join(
    str(value) for value in (sys.executable, sys.prefix, sys.base_prefix)
).upper()
modules = os.environ.get("LOADEDMODULES", "").split(":")
python_module = any(
    module.split("/", 1)[0].lower().startswith("python")
    for module in modules
)
non_relocatable = "ROIHU_TYKKY" in prefixes and not python_module
allowed = version_ok and not non_relocatable
raise SystemExit(0 if allowed and Path(sys.executable).is_file() else 1)
PY
}

python_uses_module_base() {
    local candidate=$1
    env -u PYTHONHOME -u PYTHONPATH "$candidate" - <<'PY' >/dev/null 2>&1
import os
import sys

modules = os.environ.get("LOADEDMODULES", "").split(":")
python_module = any(
    module.split("/", 1)[0].lower().startswith("python")
    for module in modules
)
raise SystemExit(0 if python_module and sys.prefix == sys.base_prefix else 1)
PY
}

select_python() {
    local candidate
    if [[ -n "${FOAMNORDIC_SEED_PYTHON:-}" ]]; then
        if python_is_supported "$FOAMNORDIC_SEED_PYTHON"; then
            printf '%s\n' "$FOAMNORDIC_SEED_PYTHON"
            return 0
        fi
        echo "[FoamNordic] FOAMNORDIC_SEED_PYTHON is not a supported Python 3.11/3.12." >&2
        return 1
    fi
    while IFS= read -r candidate; do
        [[ -n "$candidate" ]] || continue
        if python_is_supported "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done < <(type -a -p python3.12 python3.11 python3 python 2>/dev/null | awk '!seen[$0]++')
    return 1
}

SEED_PYTHON=$(select_python || true)

if [[ -z "$SEED_PYTHON" ]]; then
    echo "[FoamNordic] A supported Python 3.11 or 3.12 base was not found." >&2
    echo "[FoamNordic] Load an HPC Python module or select another local/virtual Python." >&2
    echo "[FoamNordic] Read-only Tykky-style container interpreters cannot seed this environment." >&2
    echo "[FoamNordic] Set FOAMNORDIC_SEED_PYTHON=/path/to/python when needed." >&2
    exit 1
fi
if [[ -e "$ENVIRONMENT" && ! -f "$ENVIRONMENT/pyvenv.cfg" ]]; then
    echo "[FoamNordic] Refusing non-virtual-environment path: $ENVIRONMENT" >&2
    exit 1
fi

mkdir -p "$VIRTUAL_ROOT"
if [[ ! -f "$ENVIRONMENT/pyvenv.cfg" ]]; then
    if python_uses_module_base "$SEED_PYTHON"; then
        echo "[FoamNordic] Using the loaded HPC Python module as the base."
        env -u PYTHONHOME -u PYTHONPATH \
            "$SEED_PYTHON" -m venv --system-site-packages "$ENVIRONMENT"
    else
        env -u PYTHONHOME -u PYTHONPATH \
            "$SEED_PYTHON" -m venv "$ENVIRONMENT"
    fi
fi

if ! env -u PYTHONHOME -u PYTHONPATH "$ENVIRONMENT/bin/python" -c \
    'import encodings; import sys; assert sys.prefix != sys.base_prefix' 2>/dev/null
then
    echo "[FoamNordic] Existing virtual environment is unusable: $ENVIRONMENT" >&2
    echo "[FoamNordic] Move it aside, then run this command again." >&2
    exit 1
fi

env -u PYTHONHOME -u PYTHONPATH "$ENVIRONMENT/bin/python" -m pip install --upgrade pip
env -u PYTHONHOME -u PYTHONPATH "$ENVIRONMENT/bin/python" -m pip install \
    "scikit-build-core>=0.10" \
    "nanobind>=2.4" \
    "onsaemiro>=1.0.5"
env -u PYTHONHOME -u PYTHONPATH "$ENVIRONMENT/bin/python" -m pip install \
    --no-build-isolation \
    "$REPOSITORY/python"

env -u PYTHONHOME -u PYTHONPATH "$ENVIRONMENT/bin/python" - "$ENVIRONMENT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
(root / ".foamnordic-generated.json").write_text(
    json.dumps(
        {"schema_version": 1, "kind": "virtual-environment", "root": str(root)},
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY

echo "[FoamNordic] Virtual environment: $ENVIRONMENT"
echo "[FoamNordic] Activate: source $ENVIRONMENT/bin/activate"
env -u PYTHONHOME -u PYTHONPATH "$ENVIRONMENT/bin/foamnordic" --version
