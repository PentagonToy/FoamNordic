# Maintenance

This directory keeps publication assets and release tooling outside the runtime package.

Generated wheelhouses remain untracked.

| Path | Purpose |
| --- | --- |
| [`release/`](release/README.md) | PyPI metadata, wheel preparation, release validation and tests |
| `assets/` | Package and repository artwork |
| `figures/` | Repository diagrams |

## Logo variants

| Asset | Use |
| --- | --- |
| [`assets/icon.svg`](assets/icon.svg) | Transparent symbol used in the documentation homepage, header and browser favicon |
| [`assets/banner_light.svg`](assets/banner_light.svg) | README banner for light themes |
| [`assets/banner_dark.svg`](assets/banner_dark.svg) | README banner for dark themes |

Keep the canvas and internal cut-outs transparent. Check light and dark backgrounds after changing the symbol; the documentation build versions SVG URLs by content hash.

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
6. includes `release-notes.md` from the version's changelog section;
7. generates `upload-pypi` inside the wheelhouse.

The destination is `wheelhouse-<version>-<run-id>/` by default and is ignored by Git. Publishing remains a deliberate second command:

```console
./wheelhouse-<version>-<run-id>/upload-pypi
```

The upload helper publishes to production PyPI, prompts for a PyPI token through Twine, and does not store it. Both commands use an isolated temporary Twine environment when the active Python environment does not already provide Twine.

Run `./others/release/wheel-maker --help` to select another pushed ref or output path.
