<p align="center">
  <img src="others/icon.png" alt="FoamNordic" width="220">
</p>

# FoamNordic

FoamNordic is a native C++ and Python framework for running ordinary OpenFOAM
cases and machine-learning closure cases through the same reproducible
pipeline. Closure exchange stays outside Python: Fjord carries atomic fields
over SHM, UDS, UCX, or TCP, while Longship owns placement, launch, lifecycle,
logs, observations, and results.

FoamNordic is active research software. The current package version is
`1.0.3.dev4`.

## Install

The eventual PyPI installation is intentionally simple:

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

The same installation includes ONNX packaging, Joblib/scikit-learn models,
and JAX/Equinox resident models. No backend-specific install command is needed.

Binary wheels include the native Python control runtime, so ordinary users do
not rebuild the C++ core after `pip install`. The selected OpenFOAM integration
is prepared from the case's `of_cmd` and `shell` because OpenFOAM has its own
platform and version ABI.

### Install from this repository

Use Python 3.11 or 3.12. A standalone interpreter, an ordinary writable virtual
environment, Conda, or an HPC Python module can seed the dedicated FoamNordic
environment. Non-relocatable read-only container interpreters cannot.

```console
git clone https://github.com/PentagonToy/FoamNordic.git
cd FoamNordic
bash tools/python/createVirtual.sh
source ../Virtual/FoamNordic/bin/activate
python -m unittest discover -s python/tests -v
```

The script selects a usable interpreter and rejects personal non-relocatable
Tykky-style paths. Site-managed Python modules remain supported even when
their implementation uses a read-only container internally. Any compatible
local or virtual interpreter may be selected explicitly with
`FOAMNORDIC_SEED_PYTHON=/path/to/python3.12`.

On an HPC system, first load a site-provided Python module. Module names vary;
for example, CSC systems provide `python-data`:

```console
module load python-data
bash tools/python/createVirtual.sh

# Repeat the module load before every later activation.
module load python-data
source ../Virtual/FoamNordic/bin/activate
```

When a site Python module is loaded, the installer uses
`venv --system-site-packages` so the supported module stack remains available
while FoamNordic additions stay in their own marked environment.

## Python API

Grouped namespaces use PascalCase, classes use PascalCase, and functions use
snake_case. All filesystem inputs accept strings, `pathlib.Path`, and other
text `PathLike` objects.

```python
from pathlib import Path
import foamnordic as fno

case = fno.OpenFOAM.Case(
    name="NACA4412",
    case_dir=Path("cases/NACA4412"),
    run_dir="output/NACA4412",
    of_cmd="openfoam/2512",
    shell="bash",
    ranks=16,
)

print(case.fields.keys())
print(case.field("U").internal_value)

longship = fno.Longship(case=case)
run = longship.launch()
run.summary()

# Later, when a terminal result is needed:
result = run.stop(force=False, timeout=900)
result.summary(style="compact")

post = result.postprocess
statistics = post.statistics(["U", "p"], time_idx=-1, verbose=True)
```

`0/` is used when present. If a source case contains only `0.orig/`, FoamNordic
reads it as the initial state and copies it to `0/` only inside the isolated run
directory. The source case is never modified. OpenFOAM directives such as
`#calc`, `#include`, and variable references are resolved with
`foamDictionary -expand` through the declared OpenFOAM environment.
`of_cmd="openfoam/2512"` is shorthand for loading that environment module;
provide a complete command such as `source /opt/OpenFOAM/bashrc` for another
installation, or use `None` when OpenFOAM is already available on `PATH`.

On Apple Silicon with OpenFOAM.app, its command is used as a wrapper. Local MPI
therefore needs no separate launcher configuration:

```python
case = fno.OpenFOAM.Case(
    name="lidDrivenCavity",
    case_dir=case_dir,
    run_dir=output_dir,
    of_cmd="openfoam",
    shell="zsh",
    application="pimpleFoam",
    ranks=6,
)
run = fno.Longship(case=case).launch()
```

With no `scheduler`, rank one runs directly and multiple ranks use the MPI
launcher supplied by the declared OpenFOAM environment.

See the [run-control API](docs/python/run-control-api.md) for closure,
observation, Slurm, and pure-OpenFOAM examples.
Stored fields and baseline/ML comparisons are covered by the
[postprocess API](docs/python/postprocess-api.md).

## Native C++ build

Developers can build the native libraries and tests directly:

```console
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFOAMNORDIC_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Optional backends are selected explicitly, for example with
`FOAMNORDIC_ONNX_RUNTIME=ON`, `FOAMNORDIC_ONNX_RUNTIME_ROOT=...`, or
`FOAMNORDIC_UCX=ON`. `foamnordic build --source .` provides the corresponding
compact SDK build frontend; use `foamnordic build --help` for its options.

## Target platforms

| Target | Architecture | OpenFOAM |
| --- | --- | --- |
| Linux | `x86_64` | OpenFOAM v2512; primary HPC validation target |
| Linux | `aarch64` | Supported wheel and native build target |
| macOS | Apple Silicon (`arm64`) | Native OpenFOAM v2512/v2606 development target |

The Apple Silicon setup uses
[OpenFOAM.app](https://github.com/gerlero/openfoam-app), which provides native
OpenFOAM for macOS. For example:

```console
brew install gerlero/openfoam/openfoam@2512
openfoam
```

OpenFOAM.app documents macOS 14 or later and Apple Silicon as its current
native prerequisites.

## Documentation

### Overview

- [Documentation index](docs/README.md)
- [Python package guide](python/README.md)
- [Backend-neutral mathematics](docs/python/math-api.md)
- [PyPI README](others/README.pypi.md)

### Architecture

- [Architecture index](docs/architecture/README.md)
- [Control and data planes](docs/architecture/control-and-data-planes.md)
- [Declarative plans](docs/architecture/declarative-plans.md)
- [Execution topologies](docs/architecture/execution-topologies.md)
- [Model execution](docs/architecture/model-execution.md)
- [Observations and retention](docs/architecture/observations-and-retention.md)

### Native C++ and OpenFOAM

- [Native documentation index](docs/native/README.md)
- [Benchmarks](docs/native/benchmarks.md)
- [Closure engine](docs/native/closure-engine.md)
- [Data plane](docs/native/data-plane.md)
- [Field pipeline](docs/native/field-pipeline.md)
- [HPC transport status](docs/native/hpc-transport-status.md)
- [Model artifacts](docs/native/model-artifacts.md)
- [OpenFOAM adapter](docs/native/openfoam-adapter.md)
- [Placement and lifecycle](docs/native/placement-and-lifecycle.md)

### Python

- [Python documentation index](docs/python/README.md)
- [Python design](docs/python/design.md)
- [Run-control API](docs/python/run-control-api.md)
- [Postprocess API](docs/python/postprocess-api.md)

## License

FoamNordic is distributed under the
[GNU General Public License v3.0](LICENSE).
