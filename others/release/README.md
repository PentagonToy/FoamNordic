# Release workflow

These files define FoamNordic's wheel preparation, validation and publication workflow. They are maintainer-only inputs and are not included in the runtime package.

| Path | Purpose |
| --- | --- |
| `README.pypi.md` | PyPI project description |
| `wheel-maker` | GitHub Actions wheel preparation and upload-helper frontend |
| `publishing.md` | Clean-install validation and PyPI release gate |
| `release_notes.py` | Version checks and exact changelog release-body extraction |
| `check_wheel_buildkit.py` | Installed-wheel build-kit and Longship smoke test |
| `tests/` | Offline release-metadata regression tests |

## Use

Run `./others/release/wheel-maker` from a clean, pushed `main` branch to prepare release wheels. Publishing remains a separate deliberate step documented in [`publishing.md`](publishing.md). After a PyPI upload, create the matching Git tag and GitHub release if they have not already been published.
