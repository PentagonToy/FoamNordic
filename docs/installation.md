# Installation

Install the Python package, activate OpenFOAM, then build the integration for that OpenFOAM environment. Finish with `foamnordic doctor` before running a case.

## Before starting

Use CPython 3.11 or 3.12 on Linux (`x86_64` or `aarch64`) or Apple Silicon. The commands below use a bash or zsh terminal. OpenFOAM must already be installed with its compiler and MPI environment; installing FoamNordic does not install OpenFOAM.

See the [OpenFOAM compatibility table](../README.md#openfoam-compatibility) for the tested releases, platforms and workflows. Check your site's compiler, MPI and module settings before building.

## 1. Install the Python package

Install the PyPI release with either pip or uv in the Python environment used for the project. uv requires a virtual environment; an existing one is sufficient.

<!-- tabs -->
### pip

```console
pip install foamnordic
```

### uv

```console
uv pip install foamnordic
```

<!-- /tabs -->

<details>
<summary>Create a virtual environment if needed</summary>

Use an existing Python 3.11 or 3.12 environment, or create and activate one before installing.

<!-- tabs -->
### pip

```console
python3.12 -m venv .venv
source .venv/bin/activate
```

### uv

```console
uv venv --python 3.12
source .venv/bin/activate
```

<!-- /tabs -->

For uv setup, see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

</details>

Check that the package and terminal use the intended environment:

```console
python -c "import sys, foamnordic; print(sys.executable); print(foamnordic.__file__)"
foamnordic --version
```

The plain package supports ordinary OpenFOAM runs, mathematical functions and existing model artefacts. Add [model-training dependencies](#optional-model-training-dependencies) only when needed.

## 2. Activate OpenFOAM and build

Wheels include the Python API and native control runtime. The integration build supplies the libraries and runtime executables for the active OpenFOAM compiler/ABI combination. It downloads ONNX Runtime 1.28.0 from GitHub and tries Microsoft's official NuGet package after a network or HTTP failure. Both archives use pinned SHA-256 checks; a checksum mismatch stops the build.

The NuGet fallback supports Linux x86_64, Linux ARM64 and macOS ARM64; its macOS library requires macOS 14 or later. Network access is required unless a compatible SDK is already available locally.

<!-- tabs -->
### Linux / HPC

Use the same Python environment as in step 1 and load the site's OpenFOAM module:

```console
module load openfoam/2512
command -v wmake
foamnordic build
foamnordic doctor
```

For a standalone OpenFOAM installation, source its `etc/bashrc` instead of loading a module. Follow the installation's compiler and MPI setup instructions.

### Apple Silicon

Enter the OpenFOAM.app environment, then build and check FoamNordic:

```console
openfoam
foamnordic build
foamnordic doctor
```

Use the Python environment containing FoamNordic inside this shell. If `foamnordic` is unavailable, activate that environment before building. `foamnordic build` calls `wmake` internally; no separate `wmake` command is required. Use `exit` to leave the OpenFOAM shell.

<details>
<summary>Run a command without entering the shell</summary>

The launcher also accepts a command. Resolve the executable in the current Python environment to preserve its selection:

```console
openfoam "$(command -v foamnordic)" build
openfoam "$(command -v foamnordic)" doctor
```

</details>

<!-- /tabs -->

The runtime is installed below `~/.local/share/foamnordic/runtime/<platform>/<openfoam-abi>/`. Repeat the build when selecting a different OpenFOAM compiler/ABI; separate runtimes coexist. See the [command-line reference](api/command-line.md#native-build) for build directories, prefixes and other options.

## 3. Check the environment and run a case

`foamnordic doctor` checks the environment without starting a solver. Resolve `FAIL` entries first. Solver runs require OpenFOAM, runtime profile, Longship and adapter checks to pass; parallel runs also need MPI. ModelHost or ONNX warnings can be expected from a build without ONNX.

Continue with the [first OpenFOAM run](../README.md#quick-start). Set `Case.of_cmd` to the same OpenFOAM activation used for the build: `"module load openfoam/2512"` with `shell="bash"` on the example cluster, or `"openfoam"` with `shell="zsh"` on macOS. Supply an existing case directory containing the solver dictionaries and fields.

For Foundation 14, activate `openfoam-org/14` before building and set `of_cmd="module load openfoam-org/14"` and `application="foamRun"` when declaring a case on Roihu. The case's `system/controlDict` selects the solver, such as `incompressibleFluid`. The field, closure and workload declarations use the same Python API across these releases.

For notebooks, select a kernel from the Python environment containing FoamNordic. Importing the package in the terminal does not establish that a different notebook kernel can import it.

After the first run, follow the [local cavity example](https://pentagontoy.github.io/FoamNordic/foamnordic/examples/lid-driven-cavity/) to inspect fields and run results. The [tutorials](../tutorials/README.md) introduce field exchange and closures on Slurm.

## Optional model-training dependencies

The `ml` extra adds scikit-learn, ONNX, Equinox, Optax and CPU JAX. Use `ml-cuda12` for training on a compatible Linux CUDA 12 system. These Python extras do not replace the OpenFOAM integration build.

<!-- tabs -->
### pip

```console
pip install "foamnordic[ml]"
```

For Linux CUDA 12, replace `ml` with `ml-cuda12`.

### uv

```console
uv pip install "foamnordic[ml]"
```

For Linux CUDA 12, replace `ml` with `ml-cuda12`.

<!-- /tabs -->

## If installation stops

| Symptom | Next step |
| --- | --- |
| Python version rejected | Select Python 3.11 or 3.12, then recreate the virtual environment with that interpreter. |
| `foamnordic: command not found` | Activate the environment used for installation. Inside the OpenFOAM shell, activate the same Python environment. |
| `wmake is unavailable` | Activate OpenFOAM with its compiler environment before building. |
| Runtime profile or adapter missing | Run `foamnordic build` in the OpenFOAM environment selected for the case, then run `doctor` there. |
| ONNX Runtime download fails | GitHub network or HTTP failures trigger the official NuGet fallback. If both downloads fail, the error includes both URLs. Supply an unpacked ONNX Runtime 1.28.0 SDK through `FOAMNORDIC_ONNX_RUNTIME_ROOT`, or use `build --without-onnx` for workloads without native ONNX. |
| Import works in the terminal but fails in a notebook | Select the matching Python kernel and restart it after changing the environment. |

For an unresolved failure, retain the build error and `doctor` output, including the Python path and OpenFOAM version. See the [command-line reference](api/command-line.md) for diagnostic commands.

## Install from main

The PyPI release is sufficient for normal use. To install the current `main` branch instead, repository access is required while FoamNordic remains private.

<!-- tabs -->
### pip

```console
pip install "git+https://github.com/PentagonToy/FoamNordic.git@main#subdirectory=packages/foamnordic"
```

### uv

```console
uv pip install "git+https://github.com/PentagonToy/FoamNordic.git@main#subdirectory=packages/foamnordic"
```

<!-- /tabs -->

For reproducible work, replace `main` with a tag or commit. Then repeat the OpenFOAM integration build and environment check above.

## Editable installation

Use an editable installation when modifying the Python source. Clone the repository, then install the library from the checkout:

```console
git clone https://github.com/PentagonToy/FoamNordic.git
cd FoamNordic
```

<!-- tabs -->
### pip

```console
pip install -e ./packages/foamnordic
```

### uv

```console
uv pip install -e ./packages/foamnordic
```

<!-- /tabs -->

Build the OpenFOAM integration and check the environment after installation. Editable installs reflect Python changes; native C++ changes require a rebuild. See [development](../README.md#develop-from-source) and [verification](developer/verification.md).
