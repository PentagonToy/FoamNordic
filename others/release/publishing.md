# Publishing FoamNordic wheels

FoamNordic publishes native wheels, not an sdist: the monorepo keeps the shared C++ project above `packages/foamnordic/`. Source remains available through the release Git tag. Move the pending changelog entries into a dated section for the candidate version and synchronise the version sources listed below, CLI assertions and `uv.lock`.

When local and remote `main` and `dev` already identify the same commit, `others/release/prepare-release VERSION --push` can prepare a `## [VERSION] - Unreleased` candidate. It dates the section, updates versions and the lockfile, checks metadata, amends the current commit and updates both branches with lease protection. Prepare the release on `main` directly when `dev` has a separate history; preserve that branch.

## Build

Commit and push the FoamNordic release source, then run `./others/release/wheel-maker` from a clean checkout. It builds and tests six wheels containing the `foamnordic` package and its internal `bindnordic` namespace: CPython 3.11 and 3.12 on Linux x86_64, Linux aarch64, and macOS arm64. It checks metadata, records SHA256 sums, and extracts the release body from `CHANGELOG.md`. It does not publish anything.

Release CI runs the complete Python suite once on each supported platform with Python 3.12. Each CPython wheel is then installed separately, imports its native packages, reports its CLI version and compiles the installed build kit, including Longship, without OpenFOAM or ONNX. Validate the ONNX-enabled OpenFOAM build separately on the release target environment; `--without-onnx` does not check native ONNX acquisition or execution.

## Clean-install gate

Use the designated validation environment outside the source checkout, or create a fresh environment when authorised. Keep the regular user environment unchanged. Select the exact wheel for that interpreter and platform from the newly prepared wheelhouse:

```console
VERSION="<release-version>"
python3 -m venv /absolute/path/to/authorised-test-install
source /absolute/path/to/authorised-test-install/bin/activate
python -m pip install "foamnordic[ml] @ file:///absolute/path/to/wheelhouse/foamnordic-${VERSION}-<python>-<abi>-<platform>.whl"
python -m pip check
foamnordic --version
foamnordic dir
module load openfoam/2512  # use the target site's command; macOS: openfoam
foamnordic build --build-dir /absolute/path/to/validation/build
foamnordic doctor
```

Replace the wheel placeholders with its actual filename. Use `ml-cuda12` instead of `ml` for Linux CUDA 12 validation. Confirm the candidate version, verify that `foamnordic` and `bindnordic` both come from the fresh installation, and build from the bundled kit without `--source`. Run a small OpenFOAM case to successful completion and record the platform, Python and OpenFOAM versions and result. A source-tree smoke test using an existing native runtime does not replace this installed-wheel gate. Record the exact wheel hash, interpreter, installed package paths and runtime generation. Remove only test-owned artefacts after validation.

If source or package metadata changes, rebuild and validate the resulting wheels again. Never rename a wheel to change its version.

## Publish

After the local release gates pass, upload the same checked wheels to TestPyPI and validate a clean installation:

```console
./wheelhouse-${VERSION}-<run-id>/upload-test-pypi
```

TestPyPI does not mirror every runtime dependency. When installing FoamNordic from it, keep production PyPI available for dependencies and request the exact FoamNordic version. After that gate passes, tag the exact validated commit and publish the same wheel files:

```console
./wheelhouse-${VERSION}-<run-id>/upload-pypi
```

This helper uploads to **production PyPI** and prompts for the API token through Twine. Do not store credentials in the repository. PyPI files are immutable; check that the intended version is available before publication.

Tag pushes also trigger wheel CI. Do not replace already validated artefacts with newly rebuilt ones merely because their filenames match.

## GitHub release notes

`CHANGELOG.md` is the single source of release bodies. Add one section per release and synchronize `packages/foamnordic/pyproject.toml`, `packages/foamnordic/foamnordic/__init__.py`, `CMakeLists.txt`, `packages/foamnordic/buildkit/CMakeLists.txt`, and CLI version assertions.

```console
python3 others/release/release_notes.py --tag "v${VERSION}"
python3 -m pytest others/release/tests
```

Use the generated wheelhouse `release-notes.md` verbatim for the GitHub release associated with the validated tag. Publish the GitHub release after confirming the PyPI upload. Neither wheel preparation nor changelog extraction creates a tag or a release automatically.
