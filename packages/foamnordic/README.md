# FoamNordic Python package

The Python package is FoamNordic's declarative control plane. It describes an OpenFOAM case, field programs, observations, placement, and scheduler resources, then compiles them into an immutable native plan. Solver equations and repeated field exchange remain outside Python.

The authoritative [package boundary](../README.md) states the invariant shared with BindNordic: BindNordic understands the model, while FoamNordic understands how that model is used in a simulation.

## API surface

Grouped namespaces use PascalCase, declarations use PascalCase, and functions use snake_case:

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

| Group | Names |
| --- | --- |
| Workload | `OpenFOAM`, `Field`, `Context`, `Math`, `Operator`, `Closure`, `Transform`, `Observe` |
| Execution | `Longship`, `Slurm`, `Runtime` |
| Model artefacts | `Tensor`, `Export`, `Models` |
| Results | `Postprocess` |
| Reproducibility | `Random` |

Compatibility aliases remain importable but are omitted from interactive discovery. Filesystem arguments accept strings, `pathlib.Path`, and text `PathLike` objects.

See the [API reference](../../docs/api/README.md) for closures, transforms, Slurm, observations, model export and post-processing.

## Package layout

| Path | Responsibility |
| --- | --- |
| `core/` | declarations, expressions, field metadata, plans, and validation |
| `execution/` | case preparation, launch, MPI, Slurm, lifecycle, and resident workers |
| `models/` | public compatibility facades over BindNordic model services |
| `postprocess/` | stored-field reading, statistics, and comparisons |
| `contracts/` | shipped OpenFOAM adapter contracts |
| `build/` | native compiler and ONNX Runtime acquisition helpers |
| `export.py` | public FNOM exporters backed by BindNordic |
| `openfoam.py` | public OpenFOAM case declaration |
| `math.py`, `random.py` | backend-neutral numerical APIs |
| `_cli.py` | installed `foamnordic` command entry point |

Implementation modules may move between these categories; the exports in `foamnordic.__init__` and the documented grouped namespaces define the public surface.

## Develop from source

Python 3.11 and 3.12 are supported.

```console
uv sync --all-packages --all-extras
source .venv/bin/activate
module load openfoam/2512  # omit when OpenFOAM is already loaded
foamnordic build
foamnordic dir
```

The same `foamnordic build` command works after PyPI, GitHub, or editable installation. It builds the OpenFOAM ABI-specific integration and reference solver below:

```text
~/.local/share/foamnordic/runtime/<platform>/<openfoam-abi>/
```

On Apple Silicon, enter the OpenFOAM.app shell before building. ONNX Runtime downloads try GitHub first and fall back to Microsoft's official NuGet package after network or HTTP errors. The NuGet fallback requires macOS 14 or later on Apple Silicon. On offline systems, provide an unpacked ONNX Runtime 1.28.0 SDK through `FOAMNORDIC_ONNX_RUNTIME_ROOT`, or use `foamnordic build --without-onnx` when ONNX execution is not required. See the [installation guide](../../docs/installation.md) for acquisition checks and supported platforms.

For an isolated environment outside the uv workspace, the repository helper creates a marked virtual environment:

```console
bash tools/python/createVirtual.sh
source ../Virtual/FoamNordic/bin/activate
```

Site-managed HPC Python modules should be loaded before environment creation and each later activation.

The `ml` extra uses CPU JAX on Linux and macOS. Replace it with `ml-cuda12` on a Linux CUDA 12 host.

## Runtime invariants

- An OpenFOAM baseline starts no ModelHost.
- A model-driven workload loads each FNOM artefact once per resident worker.
- Node-local Slurm placement starts one ModelHost per solver node.
- `Case.of_cmd` selects the matching installed OpenFOAM ABI at launch.
- Generated cases and caches carry ownership markers; `clobber` removes only matching marked paths.
- `Run.stop()` returns one immutable `Result`; `run.detach()` deliberately releases lifecycle ownership.
- Slurm runs cancel after a 30-second ownership grace if their Python or Jupyter process disappears abnormally.
- Completed fields remain on disk and are independent from live observations.

## Verify

Run the Python suite from the repository root in an installed development environment:

```console
python -m pytest
```

This command runs the BindNordic and FoamNordic Python suites together. Native-extension tests skip when `_native` is not installed. Backend tests may also require their optional runtime libraries. The native C++ suite is built from the repository root with `FOAMNORDIC_TESTS=ON`; see the main README.

Architecture and ownership details are in [the architecture guide](../../docs/architecture/README.md). Measured behaviour is kept separately under [benchmarks](../../docs/benchmarks/README.md).
