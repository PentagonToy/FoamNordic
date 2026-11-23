# Verification

Select checks by the boundaries changed. Portable correctness runs locally first; Roihu is reserved for behaviour that requires Linux, OpenFOAM v2512, MPI, UCX, Slurm, multiple nodes or GH200 hardware.

## Verification ladder

| Level | Run when | Evidence |
| --- | --- | --- |
| Documentation | Prose, links or examples | Diff, links, signatures, units and source/site rendering |
| Static | Source change | Repository formatting and relevant import or compile checks |
| Focused unit | Behaviour change | Tests for the owning Python or C++ component |
| Native integration | Runtime, FNOM, connector or exchange change | CMake build and the matching `tests/backend/` or BindNordic native target |
| OpenFOAM integration | Field, expression, closure, transform or installed SDK change | `foamnordic build` under the target OpenFOAM environment |
| Solver smoke | Commit timing, observation, equation or boundary change | Small case completion with expected fields, logs and records |
| Roihu single-node | Linux ABI, MPI, SHM or Slurm change | Saved batch script and retained logs under the configured test directories |
| Roihu multi-node or GPU | UCX, split placement or accelerator change | Explicit allocation, readiness, exchange and teardown evidence |

Documentation-only changes need no runtime tests or solver execution. Escalate functional checks only when the changed boundary requires them.

Release CI runs Python 3.12 source tests on Linux x86_64, Linux ARM64 and the supported `macos-15` ARM64 runner, then checks Python 3.11 and 3.12 wheels. Its installed build-kit check excludes OpenFOAM and ONNX; complete the separate installed-wheel gate in the [publishing guide](../../others/release/publishing.md) before publication.

## Test locations

| Area | Tests |
| --- | --- |
| Python API, plans and rendering | [`packages/foamnordic/tests/`](../../packages/foamnordic/tests/) |
| FNOM and model kernels | [`packages/bindnordic/tests/`](../../packages/bindnordic/tests/) |
| Rune, Harbor and channels | [`tests/backend/communication/`](../../tests/backend/communication/) |
| Field exchange and observations | [`tests/backend/adapter/`](../../tests/backend/adapter/) |
| ModelHost, Smedja and admission | [`tests/backend/inference/`](../../tests/backend/inference/) |
| Longship native lifecycle | [`tests/runtime/`](../../tests/runtime/) |
| OpenFOAM expressions and dictionary contracts | [`tests/openfoam/`](../../tests/openfoam/) |
| Installed OpenFOAM SDK | [`examples/openfoam/externalNutFjord/testInstalledSdk.sh`](../../examples/openfoam/externalNutFjord/testInstalledSdk.sh) |

## Acceptance rules

A passing build is not sufficient for a numerical change. Compare the relevant fields against an accepted baseline and report tolerances, precision, mesh, rank count and physical interval. A performance result must separate packing, inference, exchange and end-to-end solver time and must not generalise a focused microbenchmark into a solver speed-up.

Keep generated cases, build trees and temporary diagnostics outside tracked source paths. Preserve solver logs, scheduler output and numerical reports when they are substantive validation evidence; remove disposable local smoke artefacts after the result has been recorded.
