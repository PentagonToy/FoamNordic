# Maintainer assets

This directory keeps publication-facing assets and small maintainer tools out of the runtime package.

End users do not need to run these tools. The PyPI README becomes package metadata, and the icon is referenced remotely; the maintainer scripts and tests are not part of the configured wheel payload. Keep them tracked so releases can be reproduced. Ignore generated wheelhouses, not this directory.

| Path | Purpose |
| --- | --- |
| [`release/`](release/README.md) | PyPI metadata, wheel preparation, release validation and tests |
| `assets/` | PyPI and repository artwork |

## Prepare release wheels

From a clean, pushed `main` branch:

```console
./others/release/wheel-maker
```

The command:

1. verifies that local `HEAD` exactly matches `origin/main`;
2. starts `.github/workflows/wheels.yml` and waits in compact mode;
3. verifies that the completed run built the expected commit;
4. downloads and flattens all six Python 3.11/3.12 platform wheels;
5. checks wheel metadata with Twine and writes `SHA256SUMS`;
6. includes `release-notes.md` from the version's CHANGELOG section;
7. generates `upload-pypi` inside the wheelhouse.

The destination is `wheelhouse-<version>-<run-id>/` by default and is ignored by Git. Publishing remains a deliberate second command:

```console
./wheelhouse-<version>-<run-id>/upload-pypi
```

The upload helper publishes to production PyPI, prompts for a PyPI token through Twine, and does not store it. Both commands use an isolated temporary Twine environment when the active Python environment does not already provide Twine.

Run `./others/release/wheel-maker --help` to select another pushed ref or output path.
