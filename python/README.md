# FoamNordic Python package

The Python package is FoamNordic's declarative control plane. It describes an
OpenFOAM case, field programs, observations, placement, and scheduler
resources, then compiles them into an immutable native plan. Solver equations
and repeated field exchange remain outside Python.

## Public API

Grouped namespaces use PascalCase, declarations use PascalCase, and functions
use snake_case:

```python
import foamnordic as fno

case = fno.OpenFOAM.Case(
    name="lidDrivenCavity",
    case_dir="cases/lidDrivenCavity",
    run_dir="output",
    application="pimpleFoam",
)

run = fno.Longship(case=case).launch()
result = run.stop(progress=True)
post = result.postprocess
```

`fno.OpenFOAM`, `fno.Export`, `fno.Models`,
`fno.Postprocess`, and `fno.Runtime` are the canonical grouped namespaces.
Existing lowercase aliases remain part of the supported API. Filesystem
arguments accept strings, `pathlib.Path`, and text `PathLike` objects.

See the [API index](../docs/api/README.md) for closures, transforms, Slurm,
observations, model export, and postprocessing.

## Package layout

| Path | Responsibility |
| --- | --- |
| `core/` | declarations, expressions, field metadata, plans, and validation |
| `execution/` | case preparation, launch, MPI, Slurm, lifecycle, and resident workers |
| `models/` | FNOM inspection and compiled-estimator generation |
| `postprocess/` | stored-field reading, statistics, and comparisons |
| `contracts/` | shipped OpenFOAM adapter contracts |
| `build/` | native compiler and ONNX Runtime acquisition helpers |
| `export.py` | public FNOM exporters |
| `openfoam.py` | public OpenFOAM case declaration |
| `math.py`, `random.py` | backend-neutral numerical APIs |
| `_cli.py` | installed `foamnordic` command entry point |

Implementation modules may move between these categories; the exports in
`foamnordic.__init__` and the documented grouped namespaces define the public
surface.

## Install from a checkout

Python 3.11 and 3.12 are supported.

```console
python -m pip install -e ./python
module load openfoam/2512  # omit when OpenFOAM is already loaded
foamnordic build
foamnordic dir
```

The same `foamnordic build` command works after PyPI, GitHub, or editable
installation. It builds the OpenFOAM ABI-specific integration and reference
solver below:

```text
~/.local/share/foamnordic/runtime/<platform>/<openfoam-abi>/
```

On Apple Silicon, enter the OpenFOAM.app shell before building. On offline
systems, provide an unpacked ONNX Runtime through
`FOAMNORDIC_ONNX_RUNTIME_ROOT`, or use `foamnordic build --without-onnx` when
ONNX execution is not required.

The repository helper creates a marked development environment when desired:

```console
bash tools/python/createVirtual.sh
source ../Virtual/FoamNordic/bin/activate
```

Site-managed HPC Python modules should be loaded before environment creation
and each later activation.

## Runtime invariants

- A pure OpenFOAM baseline starts no ModelHost.
- A model-driven workload loads each FNOM artifact once per resident worker.
- Node-local Slurm placement starts one ModelHost per solver node.
- `Case.of_cmd` selects the matching installed OpenFOAM ABI at launch.
- Generated cases and caches carry ownership markers; `clobber` removes only
  matching marked paths.
- `Run.stop()` returns one immutable `Result`; `run.detach()` deliberately
  releases lifecycle ownership.
- Completed fields remain on disk and are independent from live observations.

## Tests

Run the Python suite from the repository root in an installed development
environment:

```console
python -m unittest discover -s python/tests -v
```

Native-extension tests skip when `_native` is not installed. Backend tests may
also require their optional runtime libraries. The native C++ suite is built
from the repository root with `FOAMNORDIC_TESTS=ON`; see the main README.

Architecture and ownership details are in
[the architecture guide](../docs/architecture/README.md). Measured behavior is
kept separately under [benchmarks](../docs/benchmarks/README.md).
