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

Launch resolves the ABI-specific ClosureHost and OpenFOAM library from a
prepared profile (`FOAMNORDIC_PREPARED_WORK_DIR`) or the explicit
`FOAMNORDIC_CLOSURE_WORKER` and `FOAMNORDIC_OPENFOAM_LIB` paths. Local launch
and one-node attached Slurm launch support `auto`, `shm`, and `uds`; multi-node
rank-to-host rendering and central UCX remain explicit validation topologies.

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

A binary wheel already contains the native control runtime:

```console
pip install foamnordic
foamnordic dir
foamnordic build
```

Outside a source tree, `foamnordic build` verifies that runtime and exits
without compiling. Inside a FoamNordic source tree, the same command builds and
installs the C++ development SDK with compact step, timing, and failure-log
output. OpenFOAM-specific compilation uses the case's `of_cmd` and `shell`
declaration because its ABI belongs to the selected OpenFOAM environment.

The binary wheel also owns the native Longship supervisor used by `Run`.
For Slurm, `launch()` waits for `RUNNING`, reports the Job ID, and then returns
the background handle; `start_timeout` can bound that pending wait without
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
in `Harbor_<name>_<jobid>.log`. These three files live under `logs/`, generated
scheduler scripts live under `slurm/`, and native observation shards use an
`observations/` directory only when requested. Local identities use
`local-<pid>`. The hidden ownership manifest also carries the immutable
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

`fno.Export.onnx(...)` accepts a path without loading the entire ONNX payload
into Python memory, which is important for large ensembles. FNOM v1 remains a
small uncompressed native manifest beside that payload. Joblib and Equinox are
separate execution backends rather than mandatory dependencies of every
FoamNordic installation.
