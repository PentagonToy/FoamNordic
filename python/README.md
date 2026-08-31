# FoamNordic Python API

This directory contains the declarative Python control plane for FoamNordic.
It describes a case, closure bindings, observations, placement, and scheduler
resources, then compiles those declarations into a deterministic immutable
plan. Solver fields and closure evaluation remain in the native C++ path.

A small nanobind `_native` module connects the stable native placement and
Longship resource-plan facades. `Longship.launch()` compiles that plan, copies
the source case into a marker-owned run directory, prepares OpenFOAM through
the case environment, and returns a non-blocking `Run`.

Public naming follows one rule: grouped namespaces use PascalCase
(`fno.OpenFOAM.Case`, `fno.Export.onnx`, `fno.Models`, `fno.Runtime`), classes
use PascalCase, and functions use snake_case. Primary workflow declarations
such as `fno.Longship` remain available directly. The original lowercase
module names remain compatibility aliases, but new examples use the canonical
PascalCase namespaces.

All public filesystem inputs accept strings, `pathlib.Path`, and other
`os.PathLike[str]` objects. A case lazily discovers valid initial fields through
`case.fields`, `case.field("U")`, and `case.boundary_names`. Ordinary files are
read locally with foamlib; directives such as `#calc`, `#include`, and variable
references automatically fall back to `foamDictionary -expand` through the
case's declared OpenFOAM environment.

An empty closure tuple is a first-class pure OpenFOAM baseline. It uses the
same case, resource planner, Slurm submission, logs, summary, and result API
without allocating or starting a ClosureHost. This keeps OpenFOAM and ML runs
comparable without maintaining two orchestration stacks.

Launch probes the case toolchain and resolves its ABI-specific OpenFOAM runtime
automatically. Prepared profiles and `FOAMNORDIC_CLOSURE_WORKER` or
`FOAMNORDIC_OPENFOAM_LIB` remain advanced overrides. Local launch and one-node
attached Slurm launch support `auto`, `shm`, and `uds`; multi-node rank-to-host
rendering and central UCX remain explicit validation topologies.

From the repository root, create a clean local environment and run the API
tests with:

```console
bash tools/python/createVirtual.sh
source ../Virtual/FoamNordic/bin/activate
python -m unittest discover -s python/tests -v
```

The installer accepts standalone Python, ordinary writable virtual
environments, Conda environments, and site-provided HPC Python modules. It
rejects personal non-relocatable read-only Tykky-style paths while allowing
site modules that use a supported container wrapper internally. An explicit
compatible interpreter can be supplied without activating it:

```console
FOAMNORDIC_SEED_PYTHON=/path/to/python3.12 bash tools/python/createVirtual.sh
source /path/to/Source/Virtual/FoamNordic/bin/activate
```

On HPC systems, load the site's Python module before both creation and every
later activation. FoamNordic automatically uses `venv --system-site-packages`
for a module-provided base. For example, on CSC:

```console
module load python-data
bash tools/python/createVirtual.sh
module load python-data
source /path/to/Source/Virtual/FoamNordic/bin/activate
```

Every installed package goes under `Source/Virtual/FoamNordic`. The environment
carries a FoamNordic ownership marker. It is intentionally excluded from bare
`clobber`; remove it only with the explicit, previewable form:

```console
foamnordic clobber --virtual /path/to/Source/Virtual/FoamNordic --dry-run
```

Supported Python versions are 3.11 and 3.12. The Python wheel is driven by the
repository's top-level CMake project so the nanobind extension links the
exported native facade targets rather than compiling private C++ sources a
second time.

A binary wheel contains the native Python control runtime and compact OpenFOAM
build kit:

```console
pip install foamnordic
module load openfoam/2512
foamnordic build
foamnordic dir
```

Joblib/scikit-learn and JAX/Equinox resident runtimes are included in the same
installation; model backends do not require separate extras.

`foamnordic build` uses the kit bundled by PyPI and GitHub installations, or
the live sources of an editable checkout. It installs the C++ SDK plus
OpenFOAM integration and `foamnordicProgressVariableFoam` reference solver below
`~/.local/share/foamnordic/runtime/<platform>/<openfoam-abi>/`. Its cache uses
the same ABI partition below `~/.cache/foamnordic/build/`. This prevents a
macOS build, a Linux build, or two OpenFOAM compiler ABIs from overwriting one
another. macOS adapters also receive a content-addressed install name, so an
older `FOAM_USER_LIBBIN` copy cannot silently shadow the selected runtime.
A source checkout can be used without a wheel:

```console
git clone https://github.com/PentagonToy/FoamNordic.git
cd FoamNordic
python -m pip install -e ./python
module load openfoam/2512  # site-specific; omit inside an OpenFOAM.app shell
foamnordic build
```

`Case.of_cmd` is probed before launch, so a case declaring
`of_cmd="module load openfoam/2512"` automatically finds that ABI's installed
runtime even when the parent Jupyter kernel did not inherit `WM_*` variables.
`FOAMNORDIC_OPENFOAM_LIB` remains an explicit diagnostic override only.

`fno.Math` supplies backend-neutral scalar, array, and physical tensor
operations for NumPy and JAX closure functions. See the
[Math API](../docs/python/math-api.md).

The binary wheel also owns the native Longship supervisor used by `Run`.
For Slurm, `launch()` waits for `RUNNING`, reports the Job ID and actual Slurm
start timestamp, and then returns the background handle. If `start_timeout`
expires while pending, an available Slurm estimated start is shown without
cancelling the job. Local launch returns after the process starts. Slurm
submission filters inherited `SLURM_*`, `SBATCH_*`, and `SRUN_*` variables in a
private environment copy without mutating the Python process. Omitting
`scheduler` runs on the current machine or current allocation.
`stop(force=False)` waits for normal completion indefinitely when no timeout is
supplied and returns one immutable `Result`. A timeout leaves the workload
running. `stop(force=True)` immediately kills the complete locally owned
process group or issues Slurm `scancel KILL`, then reaps and reports the result.
There is no second `wait()` or `raise_for_status()` ceremony.

An orderly Python or Jupyter shutdown makes a best-effort attempt to terminate
workloads owned by that process. Call `run.detach()` when a submitted workload
must deliberately outlive the kernel. Abrupt `SIGKILL`, node loss, and network
failure cannot provide the same cleanup guarantee.

`Case.name` is inherited by `Longship` unless explicitly overridden. Each
isolated run writes `Sailing_<name>_<jobid>.out` for OpenFOAM output and
`Sailing_<name>_<jobid>.log` for the FoamNordic lifecycle; the latter begins
with the standard large banner. Detailed ClosureHost output remains available
in `Harbor_<name>_<jobid>.log`, which begins with the same banner. These three
files live under `logs/`, generated
scheduler scripts live under `slurm/`, and native observation shards use an
`observations/` directory only when requested. After completion, the complete
run directory is named `<name>-slurm-<jobid>` for batch work or
`<name>-local-<timestamp>-<short-hash>` locally. The same identity is appended
to the three log names. Sailing logs finish with a compact start, finish,
total, OpenFOAM, and orchestration timing line. The hidden ownership manifest
also carries the immutable
compiled plan and digest. Preparation, submission, and job-identity state stay
under hidden `.foamnordic/` diagnostics so they remain available after a
failure without cluttering the public run layout.
`foamnordic clobber` removes only directories carrying a
matching FoamNordic ownership marker; `--workspace PATH` includes generated
runs without ever treating the source case as a cleanup target. Preview with:

```console
foamnordic clobber --workspace /path/to/output --dry-run
```

`Run.summary()` and `Result.summary()` use the required Onsaemiro dependency
for compact Job ID, Name, Status, Partition, Node, and Elapsed tables.
`run.observe()` is directly iterable and does not require `with`.

Completed solver fields remain on disk and are opened independently from live
observations:

```python
post = result.postprocess
velocity = post.field("U", time_idx=-1)
metrics = fno.Postprocess.compare(
    baseline_result,
    result,
    fields=["U", "p"],
    physical_time=1.0,
    verbose=True,
)
```

`time_idx` and `physical_time` are mutually exclusive. Statistics and
comparison return plain numerical dictionaries; `verbose=True` adds compact
Onsaemiro tables. See the
[Postprocess API](../docs/python/postprocess-api.md).

`fno.Export.onnx(...)` accepts a path without loading the entire ONNX payload
into Python memory, which is important for large ensembles. FNOM v1 remains a
small uncompressed native manifest beside that payload.
`fno.Export.joblib(...)` keeps an uncompressed sibling payload so large
NumPy-backed estimators can be memory-mapped at worker startup.
`fno.Export.equinox(...)` records the PyTree leaves in FNOM and reconstructs
and JIT-compiles the trusted model once. Joblib and Equinox are selected by
the artifact rather than by separate installation profiles.

All exporters accept fitted scikit-learn preprocessing without embedding the
Python scaler in the runtime model:

```python
artifact = fno.Export.joblib(
    model,
    path="reactionRate.fnom",
    inputs={"features": fno.Tensor.vector(components=3)},
    outputs={"omega": fno.Tensor.scalar()},
    x_scaler=fitted_x_scaler,
    y_scaler=fitted_y_scaler,
    runtime="sklearnex",  # optional; default is "sklearn"
)
```

`StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, and affine
`FunctionTransformer` are converted to FNOM coefficients once; C++ applies
them for every backend. Both scaler arguments default to `None`.
The Joblib runtime is an export-time artifact contract, so
`fno.Operator.model(...)` remains backend-neutral.

Field mutation and physical closure remain separate declarations while sharing
the same native worker and Fjord transport. Use
`fno.Operator.model("model.fnom")` inside `fno.Transform(...)` for a general
field mapping, or use `fno.Operator.function(...)` without first exporting a
model:

```python
key = fno.Random.key(42, scope="global")

def perturb_velocity(velocity, *, key):
    scale_key, noise_key = fno.Random.split(key, 2)
    scale = fno.Random.uniform(scale_key, low=0.995, high=1.005)
    noise = fno.Random.normal(noise_key, shape=velocity.shape, std=1.0e-6)
    return {"updated_velocity": velocity * scale + noise}

transform = fno.Transform(
    name="perturbVelocity",
    operator=fno.Operator.function(perturb_velocity),
    inputs={"velocity": fno.field("U")},
    outputs={"updated_velocity": fno.field("U")},
    at="time_step_start",
    key=key,
)

observations = fno.Observe(
    summaries={"U": ("min", "max", "mean", "l2")},
    interval=100,
)

longship = fno.Longship(
    case=case,
    transforms=(transform,),
    observations=(observations,),
)
```

Function arguments and returned mapping keys follow the logical port names.
Functions may request `key`, `exchange_index`, `physical_time`, or `rank` as
optional keyword arguments. The older `rng` and `seed` injections remain
available for source compatibility. `U.x`, `U.y`, and `U.z` select
one mutable vector component, for example `fno.field("U.x")` on both sides of
a transform.

Stock OpenFOAM applications support exact `time_step_start` and
`time_step_end` boundaries. `outer_corrector` and `pressure_corrected` are
valid plan stages for solver integrations, but launch is rejected unless the
application provides the corresponding native FoamNordic hook.
Both `Transform` and `Closure` accept an immutable `fno.Random.Key` and default
to `fno.Random.key(42)`. Every invocation folds in the program identity and
exchange index. `scope="global"` shares that key across ranks, while
`scope="rank"` additionally folds in the actual solver rank. Equinox
materializes the same public key as a JAX key inside its resident backend.

The Python implementation is grouped by responsibility: `core/` owns public
declarations, expressions, plans, and validation; `execution/` owns launch,
Slurm, lifecycle, observation, and resident workers; `models/` and
`postprocess/` contain their corresponding user-facing subsystems.

One Longship may carry multiple transforms. They use separate sockets,
readiness markers, and resident workers under one fail-together host group.
The same output field can therefore be transformed at `time_step_start` and
again at `time_step_end`; duplicate writers at one stage are rejected.
Live summaries work for Transform workloads as well as Closure workloads. For
Transform workloads, each due record observes fields after the latest declared
solver stage has committed its output. `interval` controls the exchange
cadence; queue bounds and stale-record eviction remain internal non-blocking
safety policies.
