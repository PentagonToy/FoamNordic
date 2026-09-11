"""Exercise mixed field-program backends in one local ModelHost process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time

import numpy as np

import foamnordic as fno


def identity_vector(value):
    return value


def identity_scalar(value):
    return value


def run_toolchain(command: str, of_cmd: str, shell: str) -> None:
    invocation = (
        ["openfoam", "-c", command]
        if of_cmd == "openfoam"
        else [shell, "-lc", f"{of_cmd}; {command}"]
    )
    subprocess.run(
        invocation,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def prepare_case(
    source: Path, root: Path, of_cmd: str, shell: str
) -> Path:
    case = root / "source"
    shutil.copytree(source, case)
    control = case / "system/controlDict"
    for entry, value in (("endTime", "0.02"), ("writeInterval", "0.02")):
        run_toolchain(
            f"foamDictionary {shlex.quote(str(control))} "
            f"-entry {shlex.quote(entry)} -set {shlex.quote(value)}",
            of_cmd,
            shell,
        )
    return case


def export_artifacts(root: Path) -> dict[str, Path]:
    from sklearn.linear_model import LinearRegression

    features = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    model = LinearRegression().fit(features, features)
    artifacts = {
        "joblib": fno.Export.sklearn(
            model,
            path=root / "identity-joblib.fnom",
            inputs={"value": fno.Tensor.vector()},
            outputs={"value": fno.Tensor.vector()},
            backend="joblib",
        ),
        "table": fno.Export.table(
            {"value": np.asarray([-10.0, 10.0])},
            np.asarray([-10.0, 10.0]),
            path=root / "identity-table.fnom",
            outputs={"value": fno.Tensor.scalar()},
        ),
    }

    import onnx
    from onnx import TensorProto, helper

    node = helper.make_node("Identity", ["value"], ["prediction"])
    graph = helper.make_graph(
        [node],
        "identity",
        [helper.make_tensor_value_info("value", TensorProto.DOUBLE, [None, 3])],
        [helper.make_tensor_value_info("prediction", TensorProto.DOUBLE, [None, 3])],
    )
    artifacts["onnx"] = fno.Export.onnx(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
        path=root / "identity-onnx.fnom",
        inputs={"value": fno.Tensor.vector()},
        outputs={"prediction": fno.Tensor.vector()},
    )

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    identity = eqx.nn.Linear(3, 3, key=jax.random.PRNGKey(0))
    identity = eqx.tree_at(lambda layer: layer.weight, identity, jnp.eye(3))
    identity = eqx.tree_at(lambda layer: layer.bias, identity, jnp.zeros(3))

    artifacts["equinox"] = fno.Export.equinox(
        identity,
        path=root / "identity-equinox.fnom",
        inputs={"value": fno.Tensor.vector()},
        outputs={"value": fno.Tensor.vector()},
    )
    return artifacts


def vector_transform(kind: str, artifacts: dict[str, Path]) -> fno.Transform:
    operator = (
        fno.Operator.function(identity_vector)
        if kind == "function"
        else fno.Operator.model(artifacts[kind])
    )
    output = "prediction" if kind == "onnx" else "value"
    return fno.Transform(
        name=f"{kind}VectorIdentity",
        operator=operator,
        inputs={"value": fno.Field("U")},
        outputs={output: fno.Field("U")},
        at="time_step_start",
    )


def scalar_transform(
    kind: str,
    artifacts: dict[str, Path],
    *,
    at: str = "time_step_end",
    suffix: str = "",
) -> fno.Transform:
    operator = (
        fno.Operator.function(identity_scalar)
        if kind == "function"
        else fno.Operator.table(artifacts["table"])
    )
    return fno.Transform(
        name=f"{kind}ScalarIdentity{suffix}",
        operator=operator,
        inputs={"value": fno.Field("p")},
        outputs={"value": fno.Field("p")},
        at=at,
    )


def latest_time(case: Path) -> Path:
    values = []
    for path in case.iterdir():
        if not path.is_dir():
            continue
        try:
            values.append((float(path.name), path))
        except ValueError:
            pass
    if not values:
        raise RuntimeError(f"no OpenFOAM time directory found in {case}")
    return max(values)[1]


def digest_fields(case: Path) -> dict[str, str]:
    time_directory = latest_time(case)
    return {
        name: hashlib.sha256((time_directory / name).read_bytes()).hexdigest()
        for name in ("U", "p")
    }


def run_case(
    label: str,
    case: Path,
    output: Path,
    transforms: tuple[fno.Transform, ...],
    of_cmd: str,
    shell: str,
) -> dict[str, object]:
    openfoam_case = fno.OpenFOAM.Case(
        name=label,
        case_dir=case,
        run_dir=output,
        of_cmd=of_cmd,
        shell=shell,
        application="pimpleFoam",
        ranks=1,
    ).initialize(mesh="blockMesh", validate_mesh=True)
    longship = fno.Longship(
        name=label,
        case=openfoam_case,
        transforms=transforms,
        placement=fno.Attached(model_cpus_per_node=4),
    )
    started = time.perf_counter()
    result = longship.launch(start_timeout=120, verbose=False).stop(timeout=180)
    if not result.success:
        raise RuntimeError(f"{label} failed; see {result.solver_log}")
    return {
        "wall_s": time.perf_counter() - started,
        "fields": digest_fields(result.case),
        "case": str(result.case),
        "run": str(result.work_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--of-cmd", default="openfoam")
    parser.add_argument("--shell", default="zsh")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="foamnordic-combinations-") as directory:
        root = Path(directory)
        case = prepare_case(
            arguments.source.resolve(), root, arguments.of_cmd, arguments.shell
        )
        artifacts = export_artifacts(root)
        combinations = {
            "function-function": (
                vector_transform("function", artifacts),
                scalar_transform("function", artifacts),
            ),
            "function-table": (
                vector_transform("function", artifacts),
                scalar_transform("table", artifacts),
            ),
            "table-table": (
                scalar_transform("table", artifacts, at="time_step_start", suffix="Start"),
                scalar_transform("table", artifacts, suffix="End"),
            ),
            "joblib-table": (
                vector_transform("joblib", artifacts),
                scalar_transform("table", artifacts),
            ),
            "equinox-table": (
                vector_transform("equinox", artifacts),
                scalar_transform("table", artifacts),
            ),
            "onnx-table": (
                vector_transform("onnx", artifacts),
                scalar_transform("table", artifacts),
            ),
        }
        report: dict[str, object] = {}
        reference = None
        for label, transforms in combinations.items():
            print(f"[FoamNordic] Combination: {label}", flush=True)
            try:
                sample = run_case(
                    label,
                    case,
                    output,
                    transforms,
                    arguments.of_cmd,
                    arguments.shell,
                )
            except Exception as error:
                report[label] = {"error": str(error)}
                continue
            if reference is None:
                reference = sample
                sample["comparison"] = {
                    name: {"max_abs": 0.0, "relative_l2": 0.0}
                    for name in ("U", "p")
                }
            else:
                comparison = fno.Postprocess.compare(
                    reference["case"],
                    sample["case"],
                    fields=("U", "p"),
                    time_idx=-1,
                    mesh="strict",
                    verbose=False,
                )
                sample["comparison"] = {
                    name: {
                        "max_abs": values["max_abs"],
                        "relative_l2": values["relative_l2"],
                    }
                    for name, values in comparison.items()
                }
            sample["matches_reference"] = all(
                values["max_abs"] <= 1.0e-12
                and values["relative_l2"] <= 1.0e-12
                for values in sample["comparison"].values()
            )
            if not sample["matches_reference"]:
                raise RuntimeError(f"{label} did not reproduce the reference fields")
            report[label] = sample
    destination = output / "model-host-combinations.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[FoamNordic] Report: {destination}")
    return 1 if any("error" in sample for sample in report.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
