<p align="center">
  <img src="https://raw.githubusercontent.com/PentagonToy/FoamNordic/main/others/icon.png" alt="FoamNordic" width="220">
</p>

# FoamNordic

FoamNordic is a native C++ and Python framework for ordinary OpenFOAM and
machine-learning closure workloads. It keeps atomic field exchange, packing,
scaling, and lifecycle in the native runtime while exposing a compact
declarative API for cases, placement, launch, observations, and results.

FoamNordic is active research software. This development line is
`1.0.3.dev6`.

## Install

```console
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install foamnordic

# Load or enter the desired OpenFOAM environment once, then:
foamnordic build

foamnordic --version
foamnordic --help
foamnordic dir
```

The same installation includes ONNX packaging, Joblib/scikit-learn models,
and JAX/Equinox resident models. No backend-specific extra is required.

Binary wheels carry the native Python control runtime and compact OpenFOAM
source build kit. `foamnordic build` compiles it for the currently loaded ABI;
each case later selects that runtime from its declared `of_cmd` and `shell`.

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

result = run.stop()
post = result.postprocess
statistics = post.statistics(["U", "p"], time_idx=-1, verbose=True)
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
- [Postprocess API](https://github.com/PentagonToy/FoamNordic/blob/main/docs/python/postprocess-api.md)
- [Native C++ documentation](https://github.com/PentagonToy/FoamNordic/tree/main/docs/native)
- [Architecture](https://github.com/PentagonToy/FoamNordic/tree/main/docs/architecture)
- [License](https://github.com/PentagonToy/FoamNordic/blob/main/LICENSE)
