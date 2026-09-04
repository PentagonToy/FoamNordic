# Publishing FoamNordic wheels

Version 1.0.5 establishes the documented distribution baseline. FoamNordic
publishes native wheels, not an sdist: the current source layout keeps the C++
project above `python/`. Source remains available through the release Git tag.

## Prepare

Commit and push the release source, then run `./others/wheel-maker` from a
clean checkout. It builds and tests six wheels: CPython 3.11 and 3.12 on Linux
x86_64, Linux aarch64, and macOS arm64. It checks metadata, records SHA256 sums,
and extracts the release body from `CHANGELOG.md`. It does not publish anything.

Wheel CI tests the installed Python/native package and compiles the installed
standalone build kit, including Longship. This is not a substitute for running
OpenFOAM on the release target environments.

## Clean-install gate

Use a fresh virtual environment outside the source checkout. Select the exact
wheel for that interpreter and platform from the newly prepared wheelhouse:

```console
python3 -m venv test-install
source test-install/bin/activate
python -m pip install --upgrade pip
python -m pip install /absolute/path/to/wheelhouse/foamnordic-1.0.5-<python>-<abi>-<platform>.whl
python -m pip check
foamnordic --version
foamnordic dir
module load openfoam/2512  # use the target site's command; macOS: openfoam
foamnordic build
foamnordic doctor
```

Replace the wheel placeholders with its actual filename. Confirm version
`1.0.5` and that Python/native module paths belong to the fresh environment.
Build from the bundled kit without `--source`. Run a small OpenFOAM case to
successful completion and record the platform, Python/OpenFOAM versions and
results. Do not overwrite an existing user's runtime to test: use an isolated
build/runtime prefix and verify that runtime, not an old default installation.
Remove only test-owned artifacts after validation.

If source or package metadata changes, rebuild and validate the resulting
wheels again. Never rename a wheel to change its version.

## Publish

After the release gates pass, tag the exact validated commit as `v1.0.5` and
publish the same checked wheel files:

```console
./wheelhouse-1.0.5-<run-id>/upload-pypi
```

This helper uploads to **production PyPI** and prompts for the API token through
Twine. Do not store credentials in the repository. PyPI files are immutable;
check that the intended version is available before publication.

Tag pushes also trigger wheel CI. Do not replace already validated artifacts
with newly rebuilt ones merely because their filenames match.

## GitHub release notes

`CHANGELOG.md` is the single source of release bodies. Add one section per
release and synchronize `python/pyproject.toml`, `python/foamnordic/__init__.py`,
`CMakeLists.txt`, `python/buildkit/CMakeLists.txt`, and CLI version assertions.

```console
python3 others/release_notes.py --tag v1.0.5
python3 -m unittest discover -s others/tests -v
```

Use the generated wheelhouse `release-notes.md` verbatim for the GitHub release
associated with the validated tag. Publish the GitHub release after confirming
the PyPI upload. Neither wheel preparation nor changelog extraction creates a
tag or a release automatically.
