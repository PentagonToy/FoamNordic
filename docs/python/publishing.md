# Publishing FoamNordic wheels

FoamNordic publishes native wheels rather than an sdist. The current source
layout keeps the C++ project above `python/`, so an isolated sdist would not
contain a valid CMake source root. The source corresponding to a release stays
available through its signed Git tag.

The `Build release wheels` GitHub workflow builds and tests this matrix:

| Python | Linux x86_64 | Linux aarch64 | macOS arm64 |
| --- | --- | --- | --- |
| CPython 3.11 | wheel | wheel | wheel |
| CPython 3.12 | wheel | wheel | wheel |

It never uploads to a package index automatically. Download all workflow
artifacts into one empty `wheelhouse/` directory, then verify them:

```console
python -m pip install --upgrade twine
python -m twine check wheelhouse/*.whl
```

TestPyPI and PyPI use separate accounts and API tokens. At each password
prompt, use a token and the username `__token__`; do not store a token in the
repository.

```console
python -m twine upload --repository testpypi wheelhouse/*.whl
```

Verify the development release in a fresh environment. The extra index is
needed because runtime dependencies normally come from PyPI, not TestPyPI:

```console
python3 -m venv test-install
source test-install/bin/activate
python -m pip install --upgrade pip
python -m pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    foamnordic==1.0.3.dev6
foamnordic --version
foamnordic dir
module load openfoam/2512  # use the target site's command
foamnordic build
```

The release-wheel gate verifies both the packaged Python/native control runtime
and that its bundled build kit can produce an OpenFOAM ABI integration without
a source checkout.

After that clean installation passes, publish the same six immutable wheel
files to PyPI:

```console
python -m twine upload wheelhouse/*.whl
```

Use the exact `1.0.3.dev6` pin while validating the development release.
Publish `1.0.4` after that gate passes to establish the rebuilt project as a
stable default installation.
