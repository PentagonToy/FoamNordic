"""Validate grouped ModelHost execution and failure teardown."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import foamnordic as fno


def export_onnx(path: Path) -> Path:
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["value"], ["prediction"])],
        "closure-update-identity",
        [helper.make_tensor_value_info("value", TensorProto.DOUBLE, [None, 3])],
        [helper.make_tensor_value_info("prediction", TensorProto.DOUBLE, [None, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    return fno.Export.onnx(
        model,
        path=path,
        inputs={"value": fno.Tensor.vector()},
        outputs={"prediction": fno.Tensor.vector()},
    )


def failing_vector(value):
    raise RuntimeError("intentional grouped ModelHost failure")


def export_invalid_onnx(path: Path) -> Path:
    return fno.Export.onnx(
        b"not-an-onnx-model",
        path=path,
        inputs={"value": fno.Tensor.vector()},
        outputs={"prediction": fno.Tensor.vector()},
    )


def identity_table(path: Path) -> Path:
    return fno.Export.table(
        {"value": np.asarray([-10.0, 10.0])},
        np.asarray([-10.0, 10.0]),
        path=path,
        outputs={"value": fno.Tensor.scalar()},
    )


def closure(primary: fno.Operator, table: Path) -> fno.Closure:
    return fno.Closure(
        name="vectorIdentity",
        operator=primary,
        inputs={"value": fno.Field("U")},
        outputs={"prediction": fno.Field("U")},
    ).update(
        name="scalarIdentity",
        operator=fno.Operator.table(table),
        inputs={"value": fno.Field("p")},
        outputs={"value": fno.Field("p")},
    )


def launch(
    name: str,
    source: Path,
    output: Path,
    template: Path,
    program: fno.Closure,
    of_cmd: str,
    shell: str,
):
    case = fno.OpenFOAM.Case(
        name=name,
        case_dir=source,
        run_dir=output,
        of_cmd=of_cmd,
        shell=shell,
        application="foamnordicClosureUpdateProbe",
        ranks=1,
        integration=fno.OpenFOAM.DictionaryTemplate(
            source=template,
            destination="system/closureUpdateProperties",
        ),
    ).initialize(mesh="blockMesh", validate_mesh=True)
    return fno.Longship(
        case=case,
        closures=(program,),
        placement=fno.Runtime.Attached(model_cpus_per_node=4),
    ).launch(start_timeout=120, verbose=False).stop(timeout=180)


def configuration(result) -> dict:
    path = result.work_dir / ".foamnordic/host-programs.json"
    return json.loads(path.read_text(encoding="utf-8"))


def lifecycle_paths(config: dict) -> tuple[Path, ...]:
    sockets = (
        Path(argument.removeprefix("unix://"))
        for program in config["programs"]
        for argument in program
        if argument.startswith("unix://")
    )
    return (
        Path(config["aggregate_ready"]),
        *(Path(value) for value in config["ready_files"]),
        *sockets,
    )


def assert_clean(config: dict) -> None:
    remaining = [str(path) for path in lifecycle_paths(config) if path.exists()]
    if remaining:
        raise RuntimeError(f"grouped ModelHost left lifecycle paths: {remaining}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--of-cmd", default="openfoam")
    parser.add_argument("--shell", default="zsh")
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    probe = Path(__file__).with_name("closureUpdateProbe")
    template = probe / "closureUpdateProperties.in"
    environment = Path(sys.executable).parent
    build_command = (
        f"export PATH={shlex.quote(str(environment))}:$PATH; "
        f"export FOAM_USER_APPBIN={shlex.quote(str(environment))}; "
        f"cd {shlex.quote(str(probe))} && wmake"
    )
    invocation = (
        ["openfoam", "-c", build_command]
        if arguments.of_cmd == "openfoam"
        else [arguments.shell, "-lc", f"{arguments.of_cmd}; {build_command}"]
    )
    subprocess.run(
        invocation,
        check=True,
        env=os.environ.copy(),
    )

    with tempfile.TemporaryDirectory(prefix="foamnordic-closure-update-") as directory:
        root = Path(directory)
        onnx = export_onnx(root / "identity-onnx.fnom")
        table = identity_table(root / "identity-table.fnom")
        source = arguments.source.expanduser().resolve()
        success = launch(
            "closure-update-success",
            source,
            output,
            template,
            closure(fno.Operator.model(onnx), table),
            arguments.of_cmd,
            arguments.shell,
        )
        if not success.success:
            raise RuntimeError(f"closure update failed; see {success.solver_log}")

        reader = fno.Postprocess.Case(success.case)
        velocity = np.asarray(reader.field("U", time_idx=-1), dtype=np.float64)
        pressure = np.asarray(reader.field("p", time_idx=-1), dtype=np.float64)
        success_config = configuration(success)
        assert_clean(success_config)
        runtime_failure = launch(
            "closure-update-runtime-failure",
            source,
            output,
            template,
            closure(fno.Operator.function(failing_vector), table),
            arguments.of_cmd,
            arguments.shell,
        )
        runtime_config = configuration(runtime_failure)
        assert_clean(runtime_config)
        if runtime_failure.success:
            raise RuntimeError("intentional runtime failure unexpectedly succeeded")
        runtime_host_log = runtime_failure.host_log.read_text(
            encoding="utf-8", errors="replace"
        )
        if "intentional grouped ModelHost failure" not in runtime_host_log:
            raise RuntimeError("runtime failure was not propagated by ModelHost")

        invalid = export_invalid_onnx(root / "invalid-onnx.fnom")
        readiness_failure = launch(
            "closure-update-readiness-failure",
            source,
            output,
            template,
            closure(fno.Operator.model(invalid), table),
            arguments.of_cmd,
            arguments.shell,
        )
        readiness_config = configuration(readiness_failure)
        assert_clean(readiness_config)
        if readiness_failure.success:
            raise RuntimeError("partial readiness failure unexpectedly succeeded")
        readiness_log = readiness_failure.longship_log.read_text(
            encoding="utf-8", errors="replace"
        )
        if "ModelHost exited before Longship readiness" not in readiness_log:
            raise RuntimeError("partial readiness failure was not detected")
        solver_log = (
            readiness_failure.solver_log.read_text(
                encoding="utf-8", errors="replace"
            )
            if readiness_failure.solver_log.is_file()
            else ""
        )
        if "Starting time loop" in solver_log:
            raise RuntimeError("solver started despite partial ModelHost readiness")

        report = {
            "success": True,
            "programs": len(success_config["programs"]),
            "connector_plugins": success_config["connector_plugins"],
            "max_abs_U": float(np.max(np.abs(velocity))),
            "max_abs_p": float(np.max(np.abs(pressure))),
            "solver_exit_cleanup": True,
            "runtime_failure_cleanup": True,
            "partial_readiness_cleanup": True,
            "runs": {
                "success": str(success.work_dir),
                "runtime_failure": str(runtime_failure.work_dir),
                "readiness_failure": str(readiness_failure.work_dir),
            },
        }
        if report["programs"] != 2 or len(report["connector_plugins"]) != 1:
            raise RuntimeError("closure stages were not consolidated as expected")
        if report["max_abs_U"] != 0.0 or report["max_abs_p"] != 0.0:
            raise RuntimeError("identity closure update changed the solver fields")

    destination = output / "closure-update.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[FoamNordic] Report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
