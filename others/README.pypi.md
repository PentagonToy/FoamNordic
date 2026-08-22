<p align="center">
  <img src="https://raw.githubusercontent.com/PentagonToy/FoamNordic/main/others/icon.png" alt="FoamNordic" width="220">
</p>

# FoamNordic

FoamNordic is a native C++ and Python framework for ordinary OpenFOAM and
machine-learning closure workloads. It keeps atomic field exchange and model
evaluation outside Python while exposing a compact declarative API for cases,
placement, launch, observations, and results.

FoamNordic is active research software. This development line is
`1.0.3.dev3`.

## Install

```console
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install foamnordic

foamnordic --version
foamnordic --help
foamnordic dir
foamnordic build
```

Binary wheels carry the native Python control runtime. OpenFOAM integration is
prepared from the `of_cmd` and `shell` declared by each case.

## First case

```python
from pathlib import Path
import foamnordic as fno

case = fno.OpenFOAM.Case(
    name="cavity",
    case_dir=Path("cases/cavity"),
    run_dir="output/cavity",
    of_cmd="openfoam/2512",
    shell="bash",
)

print(case.fields.keys())

run = fno.Longship(case=case).launch()
run.summary()
```

Paths may be strings, `pathlib.Path`, or text `PathLike` objects. FoamNordic
uses `0/` when available and otherwise copies `0.orig/` to `0/` inside the
isolated run without changing the source case.

## Platforms

- Linux `x86_64`
- Linux `aarch64`
- macOS Apple Silicon (`arm64`)

Native macOS development uses
[OpenFOAM.app](https://github.com/gerlero/openfoam-app). It provides current
OpenFOAM releases as Apple Silicon applications and Homebrew packages.

## Learn more

- [Project repository](https://github.com/PentagonToy/FoamNordic)
- [Installation and build guide](https://github.com/PentagonToy/FoamNordic#install)
- [Python run-control API](https://github.com/PentagonToy/FoamNordic/blob/main/docs/python/run-control-api.md)
- [Native C++ documentation](https://github.com/PentagonToy/FoamNordic/tree/main/docs/native)
- [Architecture](https://github.com/PentagonToy/FoamNordic/tree/main/docs/architecture)
- [License](https://github.com/PentagonToy/FoamNordic/blob/main/LICENSE)
