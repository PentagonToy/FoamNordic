# Maintainer assets

This directory keeps publication-facing assets and small maintainer tools out
of the runtime package.

End users do not need to run these tools. The PyPI README becomes package
metadata, and the icon is referenced remotely; the maintainer scripts and tests
are not part of the configured wheel payload. Keep them tracked so releases
can be reproduced. Ignore generated wheelhouses, not this directory.

| Path | Purpose |
| --- | --- |
| `README.pypi.md` | PyPI project description |
| `icon.png` | PyPI and repository artwork |
| `wheel-maker` | GitHub Actions wheel preparation frontend |
| [`publishing.md`](publishing.md) | Clean-install validation and PyPI release gate |
| `release_notes.py` | Version checks and exact CHANGELOG release-body extraction |
| `check_wheel_buildkit.py` | Compile the installed wheel's standalone build kit and test Longship |
| `tests/` | Offline release-metadata regression tests |

## Prepare release wheels

From a clean, pushed `main` branch:

```console
./others/wheel-maker
```

The command:

1. verifies that local `HEAD` exactly matches `origin/main`;
2. starts `.github/workflows/wheels.yml` and waits in compact mode;
3. verifies that the completed run built the expected commit;
4. downloads and flattens all six Python 3.11/3.12 platform wheels;
5. checks wheel metadata with Twine and writes `SHA256SUMS`;
6. includes `release-notes.md` from the version's CHANGELOG section;
7. generates `upload-pypi` inside the wheelhouse.

The destination is `wheelhouse-<version>-<run-id>/` by default and is ignored
by Git. Publishing remains a deliberate second command:

```console
./wheelhouse-<version>-<run-id>/upload-pypi
```

The upload helper publishes to production PyPI, prompts for a PyPI token through Twine, and does not
store it. Both commands use an isolated temporary Twine environment when the
active Python environment does not already provide Twine.

Run `./others/wheel-maker --help` to select another pushed ref or output path.
