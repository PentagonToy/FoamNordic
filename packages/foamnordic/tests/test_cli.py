from __future__ import annotations

import json
import os
import subprocess
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from foamnordic._cli import (
    _doctor_checks,
    _installation_coherence,
    _native_build_environment,
    _source_root,
    main,
)
from foamnordic.build.onnxruntime import NativeOnnxRuntime
from foamnordic.core.managed import MARKER, mark_generated
from foamnordic.core.native_plan import available as native_available
from foamnordic.execution.runtime_paths import openfoam_abi_for_toolchain

import foamnordic as fno


def test_native_build_prefers_openfoam_toolchain_compilers(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        toolchain = root / "toolchain/bin"
        tykky = root / "tykky/bin"
        toolchain.mkdir(parents=True)
        tykky.mkdir(parents=True)
        for executable in (toolchain / "gcc", toolchain / "g++"):
            executable.touch()
            executable.chmod(0o755)
        _mock_patch_1 = mocker.patch.dict(
            os.environ,
            {
                "PATH": os.pathsep.join((str(tykky), str(toolchain))),
                "CC": str(toolchain / "gcc"),
                "CXX": str(toolchain / "g++"),
            },
            clear=True,
        )
        try:
            environment = _native_build_environment()
        finally:
            mocker.stop(_mock_patch_1)

        entries = environment["PATH"].split(os.pathsep)
        assert entries[0] == str(toolchain.resolve())
        assert entries.count(str(toolchain.resolve())) == 1
        assert str(tykky) in entries


def test_case_toolchain_probe_discovers_openfoam_abi(mocker) -> None:
    completed = subprocess.CompletedProcess(
        args=("bash",),
        returncode=0,
        stdout=(
            "module noise\n"
            "__FOAMNORDIC_VERSION__=v2512\n"
            "__FOAMNORDIC_OPTIONS__=linux64GccDPInt32-spack\n"
        ),
        stderr="",
    )
    toolchain = SimpleNamespace(
        command="module load openfoam/2512",
        wrapper=False,
        shell="bash",
    )
    _mock_patch_2 = mocker.patch(
        "foamnordic.execution.runtime_paths.subprocess.run", return_value=completed
    )
    try:
        assert (
            openfoam_abi_for_toolchain(toolchain)
            == "openfoam-v2512-linux64GccDPInt32-spack"
        )
    finally:
        mocker.stop(_mock_patch_2)


def test_build_discovers_kit_bundled_beside_installed_package(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        package = root / "site-packages/foamnordic"
        buildkit = package / "buildkit"
        (buildkit / "src/foamnordic/runtime").mkdir(parents=True)
        (buildkit / "CMakeLists.txt").touch()
        unrelated = root / "working-directory"
        unrelated.mkdir()
        previous = Path.cwd()
        try:
            os.chdir(unrelated)
            _mock_patch_3 = mocker.patch(
                "foamnordic._cli.__file__",
                str(package / "_cli.py"),
            )
            try:
                assert _source_root(None) == buildkit.resolve()
            finally:
                mocker.stop(_mock_patch_3)
        finally:
            os.chdir(previous)


def test_public_version_and_directory_are_discoverable() -> None:
    assert fno.__version__ == "1.0.10"
    assert "Longship" in dir(fno)
    assert "Export" in dir(fno)
    assert "export" not in dir(fno)


def test_cli_version_is_discoverable() -> None:
    output = StringIO()
    with pytest.raises(SystemExit) as stopped, redirect_stdout(output):
        main(["--version"])
    assert stopped.value.code == 0
    assert output.getvalue().strip() == "foamnordic 1.0.10"


def test_top_level_help_lists_runtime_and_fnom_commands() -> None:
    output = StringIO()
    with pytest.raises(SystemExit) as stopped, redirect_stdout(output):
        main(["--help"])
    assert stopped.value.code == 0
    assert "[command] ..." in output.getvalue()
    assert "{dir,doctor,build" not in output.getvalue()
    assert "dir" in output.getvalue()
    assert "doctor" in output.getvalue()
    assert "build" in output.getvalue()
    assert "inspect" in output.getvalue()
    assert "validate" in output.getvalue()
    assert "diff" in output.getvalue()
    assert "compile" in output.getvalue()
    assert "clobber" in output.getvalue()


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_fnom_inspect_and_validate_are_discoverable() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = fno.export.onnx(
            b"onnx-payload",
            path=Path(directory) / "model.fnom",
            name="test-model",
            inputs={"x": fno.Tensor.scalar()},
            outputs={"y": fno.Tensor.scalar()},
        )
        output = StringIO()
        with redirect_stdout(output):
            assert main(["inspect", str(path), "--json"]) == 0
        record = json.loads(output.getvalue())
        assert record["name"] == "test-model"
        assert record["backend"] == "onnx"

        output = StringIO()
        with redirect_stdout(output):
            assert main(["validate", str(path)]) == 0
        assert "Valid FNOM" in output.getvalue()


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_fnom_compile_warms_target_cache(mocker) -> None:
    from sklearn.linear_model import Ridge

    with tempfile.TemporaryDirectory() as directory:
        model = Ridge().fit([[0.0], [1.0]], [0.0, 1.0])
        path = fno.export.sklearn(
            model,
            path=Path(directory) / "linear.fnom",
            inputs={"x": fno.Tensor.scalar()},
            outputs={"y": fno.Tensor.scalar()},
        )
        library = Path(directory) / "model.dylib"
        _mock_patch_4 = mocker.patch(
            "foamnordic.build.compiler.compile_source",
            return_value=(library, False, 0.125),
        )
        try:
            output = StringIO()
            with redirect_stdout(output):
                assert main(["compile", str(path)]) == 0
        finally:
            mocker.stop(_mock_patch_4)
        assert "Compile time: 0.125 s" in output.getvalue()


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_fnom_diff_compares_contract_and_embedded_payload() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        left = fno.export.onnx(
            b"same-payload",
            path=root / "left.fnom",
            name="same-model",
            inputs={"x": fno.Tensor.scalar()},
            outputs={"y": fno.Tensor.scalar()},
        )
        right = fno.export.onnx(
            b"same-payload",
            path=root / "right.fnom",
            name="same-model",
            inputs={"x": fno.Tensor.scalar()},
            outputs={"y": fno.Tensor.scalar()},
        )
        output = StringIO()
        with redirect_stdout(output):
            assert main(["diff", str(left), str(right)]) == 0
        assert "Equivalent FNOM" in output.getvalue()

        changed = fno.export.onnx(
            b"changed-payload",
            path=root / "changed.fnom",
            name="same-model",
            inputs={"x": fno.Tensor.scalar()},
            outputs={"y": fno.Tensor.scalar()},
        )
        output = StringIO()
        with redirect_stdout(output):
            assert main(["diff", str(left), str(changed), "--json"]) == 1
        record = json.loads(output.getvalue())
        assert not record["equal"]
        assert "payload_sha256" in record["differences"]


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_fnom_validate_rejects_corrupt_container(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "broken.fnom"
        path.write_bytes(b"FNOMBIN\0")
        errors = mocker.patch("sys.stderr", new_callable=StringIO)
        try:
            assert main(["validate", str(path)]) == 1
        finally:
            mocker.stop(errors)
        assert "foamnordic: error" in errors.getvalue()


def test_dir_reports_active_installation_paths() -> None:
    output = StringIO()
    with redirect_stdout(output):
        status = main(["dir"])
    assert status == 0
    report = output.getvalue()
    assert "FoamNordic directories" in report
    assert "Python environment" in report
    assert "Python package" in report
    assert "Native module" in report
    assert "Templates" in report


def test_dir_prints_packaged_templates_without_openfoam() -> None:
    output = StringIO()
    with redirect_stdout(output):
        assert main(["dir", "--templates"]) == 0
    templates = Path(output.getvalue().strip())
    assert (templates / "openfoam/scaffolds/model-adapter").is_dir()


def test_dir_prints_scriptable_sdk_paths(mocker) -> None:
    selected = SimpleNamespace(runtime_dir=Path("/opt/foamnordic/runtime"))
    _mock_patch_6 = mocker.patch("foamnordic._cli.profile", return_value=selected)
    try:
        output = StringIO()
        with redirect_stdout(output):
            assert main(["dir", "--include"]) == 0
    finally:
        mocker.stop(_mock_patch_6)
    assert output.getvalue().strip() == "/opt/foamnordic/runtime/include"


def test_dir_selects_exact_openfoam_library(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime = Path(directory)
        library = runtime / "lib/libfoamnordicOpenFOAM.so"
        library.parent.mkdir()
        library.touch()
        selected = SimpleNamespace(runtime_dir=runtime)
        _mock_patch_7 = mocker.patch("foamnordic._cli.profile", return_value=selected)
        try:
            output = StringIO()
            with redirect_stdout(output):
                assert main(["dir", "--openfoam-library"]) == 0
        finally:
            mocker.stop(_mock_patch_7)
        assert output.getvalue().strip() == str(library.resolve())


def test_doctor_reports_fast_read_only_environment_checks(mocker) -> None:
    _mock_patch_8 = mocker.patch("foamnordic._cli.profile", return_value=None)
    _mock_patch_9 = mocker.patch.dict(os.environ, {"CXX": "c++"}, clear=False)
    _mock_patch_10 = mocker.patch(
        "foamnordic._cli.shutil.which",
        side_effect=lambda value: f"/usr/bin/{value}" if value == "c++" else None,
    )
    try:
        checks = _doctor_checks()
    finally:
        mocker.stop(_mock_patch_10)
        mocker.stop(_mock_patch_9)
        mocker.stop(_mock_patch_8)
    records = {name: (status, detail) for status, name, detail in checks}
    assert records["FoamNordic"][0] == "PASS"
    assert records["Python"][0] == "PASS"
    assert records["C++ compiler"][0] == "PASS"
    assert records["OpenFOAM"][0] == "WARN"
    assert records["Runtime profile"][0] == "WARN"


def test_installation_coherence_detects_mixed_python_and_native_paths() -> None:
    status, _ = _installation_coherence(
        Path("/source/foamnordic"),
        Path("/venv/site-packages/foamnordic/_native.so"),
    )
    assert status == "WARN"
    status, _ = _installation_coherence(
        Path("/venv/site-packages/foamnordic"),
        Path("/venv/site-packages/foamnordic/_native.so"),
    )
    assert status == "PASS"


def test_build_help_is_discoverable() -> None:
    output = StringIO()
    with pytest.raises(SystemExit) as stopped, redirect_stdout(output):
        main(["build", "--help"])
    assert stopped.value.code == 0
    assert "--source" in output.getvalue()
    assert "--dry-run" in output.getvalue()
    assert "--without-onnx" in output.getvalue()


def test_source_build_dry_run_is_visual_and_non_mutating(mocker) -> None:
    repository = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        build_dir = root / "build"
        prefix = root / "install"
        onnxruntime = NativeOnnxRuntime(
            root / "onnxruntime",
            root / "onnxruntime/include",
            root / "onnxruntime/lib/libonnxruntime.so",
            "test",
        )
        output = StringIO()
        _mock_patch_11 = mocker.patch.dict(
            os.environ,
            {
                "WM_PROJECT_VERSION": "2512",
                "WM_OPTIONS": "linux64GccDPInt32",
            },
            clear=False,
        )
        _mock_patch_12 = mocker.patch(
            "foamnordic._cli.shutil.which",
            side_effect=lambda value: f"/usr/bin/{value}",
        )
        _mock_patch_13 = mocker.patch(
            "foamnordic.execution.runtime_paths.shutil.which",
            return_value="/usr/bin/wmake",
        )
        _mock_patch_14 = mocker.patch(
            "foamnordic._cli.resolve_onnxruntime",
            return_value=onnxruntime,
        )
        try:
            with (
                redirect_stdout(output),
            ):
                status = main(
                    [
                        "build",
                        "--source",
                        str(repository),
                        "--build-dir",
                        str(build_dir),
                        "--prefix",
                        str(prefix),
                        "--jobs",
                        "2",
                        "--dry-run",
                    ]
                )
        finally:
            mocker.stop(_mock_patch_14)
            mocker.stop(_mock_patch_13)
            mocker.stop(_mock_patch_12)
            mocker.stop(_mock_patch_11)
        assert status == 0
        assert "███████╗ ██████╗" in output.getvalue()
        assert " Native build" in output.getvalue()
        assert output.getvalue().count("=" * 66) == 3
        assert "OpenFOAM ABI   : openfoam-v2512-linux64GccDPInt32" in output.getvalue()
        assert "Native runtime:" not in output.getvalue()
        assert "[Step 1/5]" in output.getvalue()
        assert "Configure native SDK" in output.getvalue()
        assert "foamnordic_runtime_bundle" in output.getvalue()
        assert "Install native runtime tools" in output.getvalue()
        assert "Build OpenFOAM integration" in output.getvalue()
        assert "Build progress-variable solver" not in output.getvalue()
        assert not build_dir.exists()
        assert not prefix.exists()


def test_packaged_buildkit_defines_native_model_worker() -> None:
    repository = Path(__file__).resolve().parents[3]
    buildkit = (repository / "packages/foamnordic/buildkit/CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    packaging = (repository / "packages/foamnordic/CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    openfoam_options = (repository / "src/foamnordic/openfoam/Make/options").read_text(
        encoding="utf-8"
    )

    assert "add_subdirectory(src/foamnordic/backend/inference)" in buildkit
    assert "add_subdirectory(tools/resident)" in buildkit
    assert "foamnordic_inference" in buildkit
    assert "add_custom_target(foamnordic_runtime_bundle" in buildkit
    assert (
        "add_dependencies(foamnordic_runtime_bundle foamnordic_onnx_connector)"
        in buildkit
    )
    assert "src/foamnordic/backend/inference" in packaging
    assert "src/foamnordic/backend/connectors" in packaging
    assert "tools/resident" in packaging
    assert "src/foamnordic/openfoam/closureHook.H" in buildkit
    assert "-I$(BINDNORDIC_SOURCE_DIR)/include" in openfoam_options
    assert "tools/openfoam/progressVariableFoam" not in packaging


def test_build_refreshes_only_marker_owned_cache_from_another_source(mocker) -> None:
    repository = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        build_dir = root / "build"
        prefix = root / "runtime"
        mark_generated(build_dir, kind="build")
        (build_dir / "CMakeCache.txt").write_text(
            "CMAKE_HOME_DIRECTORY:INTERNAL=/old/foamnordic/source\n",
            encoding="utf-8",
        )
        output = StringIO()
        _mock_patch_15 = mocker.patch.dict(
            os.environ,
            {
                "WM_PROJECT_VERSION": "2512",
                "WM_OPTIONS": "linux64GccDPInt32",
            },
            clear=False,
        )
        _mock_patch_16 = mocker.patch(
            "foamnordic._cli.shutil.which",
            side_effect=lambda value: f"/usr/bin/{value}",
        )
        _mock_patch_17 = mocker.patch(
            "foamnordic.execution.runtime_paths.shutil.which",
            return_value="/usr/bin/wmake",
        )
        try:
            with (
                redirect_stdout(output),
            ):
                status = main(
                    [
                        "build",
                        "--source",
                        str(repository),
                        "--build-dir",
                        str(build_dir),
                        "--prefix",
                        str(prefix),
                        "--without-onnx",
                        "--dry-run",
                    ]
                )
        finally:
            mocker.stop(_mock_patch_17)
            mocker.stop(_mock_patch_16)
            mocker.stop(_mock_patch_15)
        assert status == 0
        assert "Would refresh build cache" in output.getvalue()
        assert "foamnordic_runtime_bundle" in output.getvalue()
        assert "foamnordic_model_worker" not in output.getvalue()
        assert (build_dir / "CMakeCache.txt").is_file()


@pytest.mark.skipif(
    not native_available(), reason="nanobind extension is not installed"
)
def test_build_outside_source_requests_a_checkout(mocker) -> None:
    previous = Path.cwd()
    output = StringIO()
    try:
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            _mock_patch_18 = mocker.patch(
                "foamnordic._cli._source_root", return_value=None
            )
            try:
                with (
                    redirect_stdout(output),
                ):
                    status = main(["build"])
            finally:
                mocker.stop(_mock_patch_18)
    finally:
        os.chdir(previous)
    assert status == 1


def test_clobber_removes_only_valid_marker_owned_runs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory) / "workspace"
        owned = workspace / "runs/owned"
        unowned = workspace / "runs/unowned"
        mark_generated(owned, kind="run")
        unowned.mkdir(parents=True)
        (unowned / MARKER).write_text(
            '{"kind": "run", "root": "/copied"}\n',
            encoding="utf-8",
        )
        output = StringIO()
        with redirect_stdout(output):
            status = main(["clobber", "--workspace", str(workspace)])
        assert status == 0
        assert not owned.exists()
        assert unowned.exists()
        assert "Removed 1" in output.getvalue()


def test_clobber_dry_run_preserves_marked_run() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory) / "workspace"
        owned = workspace / "runs/owned"
        mark_generated(owned, kind="run")
        output = StringIO()
        with redirect_stdout(output):
            status = main(["clobber", "--workspace", str(workspace), "--dry-run"])
        assert status == 0
        assert owned.exists()
        assert "would remove" in output.getvalue()


def test_clobber_removes_only_explicit_marker_owned_virtual() -> None:
    with tempfile.TemporaryDirectory() as directory:
        virtual = Path(directory) / "Virtual/FoamNordic"
        mark_generated(virtual, kind="virtual-environment")
        output = StringIO()
        with redirect_stdout(output):
            status = main(["clobber", "--virtual", str(virtual)])
        assert status == 0
        assert not virtual.exists()
        assert "virtual-environment" in output.getvalue()


def test_scoped_clobber_does_not_select_active_runtime(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory) / "workspace"
        owned = workspace / "runs/owned"
        mark_generated(owned, kind="run")
        active = Path(directory) / "runtime/generations/current"
        mark_generated(active, kind="native-runtime")
        _mock_patch_19 = mocker.patch(
            "foamnordic._cli.active_runtime_candidates",
            return_value=(active,),
        )
        try:
            with (
                redirect_stdout(StringIO()),
            ):
                status = main(["clobber", "--workspace", str(workspace)])
        finally:
            mocker.stop(_mock_patch_19)
        assert status == 0
        assert not owned.exists()
        assert active.exists()


def test_clobber_discovers_all_marker_owned_abi_runtimes(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        home = Path(directory)
        runtime = home / ".local/share/foamnordic/runtime/linux-x86_64/openfoam-v2512"
        build = home / ".cache/foamnordic/build/linux-x86_64/openfoam-v2512"
        unowned = home / ".local/share/foamnordic/runtime/linux-x86_64/preserve"
        mark_generated(runtime, kind="native-runtime")
        mark_generated(build, kind="build")
        unowned.mkdir(parents=True)
        _mock_patch_20 = mocker.patch("pathlib.Path.home", return_value=home)
        _mock_patch_21 = mocker.patch.dict(os.environ, {}, clear=True)
        try:
            with (
                redirect_stdout(StringIO()),
            ):
                status = main(["clobber"])
        finally:
            mocker.stop(_mock_patch_21)
            mocker.stop(_mock_patch_20)
        assert status == 0
        assert not runtime.exists()
        assert not build.exists()
        assert unowned.exists()


def test_clobber_removes_current_selector_with_active_generation(mocker) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "runtime/linux-aarch64/openfoam-v2512"
        generation = root / "generations/build-one"
        mark_generated(generation, kind="native-runtime")
        root.mkdir(parents=True, exist_ok=True)
        (root / "current").symlink_to(
            generation.relative_to(root), target_is_directory=True
        )
        _mock_patch_22 = mocker.patch(
            "foamnordic._cli.active_runtime_candidates", return_value=(generation,)
        )
        _mock_patch_23 = mocker.patch("foamnordic._cli.profile", return_value=None)
        _mock_patch_24 = mocker.patch(
            "pathlib.Path.home", return_value=Path(directory) / "empty-home"
        )
        try:
            with (
                redirect_stdout(StringIO()),
            ):
                status = main(["clobber"])
        finally:
            mocker.stop(_mock_patch_24)
            mocker.stop(_mock_patch_23)
            mocker.stop(_mock_patch_22)
        assert status == 0
        assert not generation.exists()
        assert not (root / "current").is_symlink()
