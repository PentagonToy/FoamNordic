"""Create and read compact, lossless OpenFOAM field datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import subprocess
from typing import Iterable, Sequence

import numpy as np
import onsaemiro as osm
from bindnordic.fdata import FData, read_fdata, write_fdata
from bindnordic.openfoam import OpenFOAMDecodeError, read_field as read_openfoam_field

from ._field import _cell_count, numeric_times, processor_directories
from ..execution.shell import quote_command, toolchain_shell
from ..openfoam import _configured_toolchain


_GRADIENT = re.compile(r"grad([A-Za-z_]\w*)$")
_DELTA_COEFFICIENT = re.compile(r"\bdeltaCoeff\s+([-+0-9.eE]+)\s*;")


def _normalise_fields(fields: Sequence[str]) -> tuple[str, ...]:
    values = tuple(fields)
    if not values:
        raise ValueError("fields must contain at least one OpenFOAM field name")
    if len(set(values)) != len(values):
        raise ValueError("fields must not contain duplicates")
    if any(not value or Path(value).name != value for value in values):
        raise ValueError("each field must be one non-empty path component")
    return values


def _normalise_times(
    case_dir: Path,
    times: Sequence[float] | tuple[float, float] | None,
) -> tuple[float, ...]:
    available = tuple(sorted(numeric_times(case_dir)))
    processors = processor_directories(case_dir)
    if processors:
        shared = set(available) if available else set(numeric_times(processors[0]))
        for processor in processors:
            shared.intersection_update(numeric_times(processor))
        available = tuple(sorted(shared))
    if not available:
        raise FileNotFoundError(f"No OpenFOAM time directories were found in {case_dir}")
    if times is None:
        return available
    values = tuple(float(value) for value in times)
    if len(values) == 2 and values[0] <= values[1]:
        selected = tuple(value for value in available if values[0] <= value <= values[1])
    else:
        selected = tuple(
            value for value in available if any(math.isclose(value, requested, rel_tol=1.0e-12, abs_tol=1.0e-14) for requested in values)
        )
    if not selected:
        raise ValueError("times selected no stored OpenFOAM time directories")
    return selected


def _time_path(directory: Path, physical_time: float) -> Path:
    for value, name in numeric_times(directory).items():
        if math.isclose(value, physical_time, rel_tol=1.0e-12, abs_tol=1.0e-14):
            return directory / name
    raise FileNotFoundError(f"{directory} does not contain time {physical_time:g}")


def _read_values(field_path: Path) -> tuple[np.ndarray, str, str | None]:
    raw = field_path.read_bytes()
    uniform = re.search(rb"\binternalField\s+uniform\b", raw) is not None
    field = read_openfoam_field(
        field_path,
        cell_count=_cell_count(field_path) if uniform else None,
    )
    values = np.asarray(field.values)
    if values.ndim == 0:
        raise ValueError(f"Uniform field {field_path} requires a mesh cell count")
    return values, field.field_class, field.dimensions


def _read_addressing(processor: Path) -> np.ndarray:
    address_path = processor / "constant/polyMesh/cellProcAddressing"
    if not address_path.is_file():
        raise FileNotFoundError(f"{address_path} is required to reconstruct processor fields")
    raw = address_path.read_bytes()
    header = re.search(rb"\bobject\s+cellProcAddressing\s*;", raw)
    if header is None:
        raise ValueError(f"Could not decode {address_path}")
    listing = re.search(rb"(?m)^\s*(\d+)\s*\(", raw[header.end() :])
    if listing is None:
        raise ValueError(f"Could not decode {address_path}")
    count = int(listing.group(1))
    payload_start = header.end() + listing.end()
    binary = re.search(rb"\bformat\s+binary\s*;", raw) is not None
    if binary:
        architecture = re.search(rb'\barch\s+"([^"]+)"\s*;', raw)
        arch = architecture.group(1).decode("ascii") if architecture else ""
        match_label = re.search(r"(?:^|;)label=(32|64)(?:;|$)", arch)
        bits = int(match_label.group(1)) if match_label else 32
        byte_order = ">" if arch.startswith("MSB;") else "<"
        dtype = np.dtype(f"{byte_order}i{bits // 8}")
        start = payload_start
        if raw[start : start + 2] == b"\r\n":
            start += 2
        elif raw[start : start + 1] == b"\n":
            start += 1
        end = start + count * dtype.itemsize
        if re.match(rb"\s*(?:\)\s*(?:;|//)|//)", raw[end:]) is None:
            raise ValueError(f"{address_path} binary payload is incomplete")
        values = np.frombuffer(raw[start:end], dtype=dtype).astype(np.int64, copy=False)
    else:
        end = raw.find(b")", payload_start)
        if end < 0:
            raise ValueError(f"{address_path} is unterminated")
        values = np.fromstring(raw[payload_start:end], dtype=np.int64, sep=" ")
    if values.size != count:
        raise ValueError(f"{address_path} count does not match its contents")
    if np.any(values < 0):
        raise ValueError(f"{address_path} contains a negative global cell index")
    return values


def _reconstruct_processor_field(case_dir: Path, physical_time: float, field_name: str) -> tuple[np.ndarray, str, str | None]:
    processors = processor_directories(case_dir)
    arrays: list[np.ndarray] = []
    addresses: list[np.ndarray] = []
    field_class: str | None = None
    dimensions: str | None = None
    for processor in processors:
        values, current_class, current_dimensions = _read_values(_time_path(processor, physical_time) / field_name)
        address = _read_addressing(processor)
        if len(values) != len(address):
            raise ValueError(f"{processor.name} field {field_name!r} does not match cellProcAddressing")
        if field_class is None:
            field_class, dimensions = current_class, current_dimensions
        elif (field_class, dimensions) != (current_class, current_dimensions):
            raise ValueError(f"Processor field {field_name!r} has incompatible metadata")
        arrays.append(values)
        addresses.append(address)
    joined_address = np.concatenate(addresses)
    if joined_address.size == 0 or np.unique(joined_address).size != joined_address.size:
        raise ValueError(f"Processor field {field_name!r} has duplicate global cell addressing")
    if not np.array_equal(np.sort(joined_address), np.arange(joined_address.max() + 1)):
        raise ValueError(f"Processor field {field_name!r} has non-contiguous global cell addressing")
    joined_values = np.concatenate(arrays, axis=0)
    reconstructed = np.empty((joined_address.max() + 1, *joined_values.shape[1:]), dtype=joined_values.dtype)
    reconstructed[joined_address] = joined_values
    assert field_class is not None
    return reconstructed, field_class, dimensions


def _field_names(field_name: str) -> tuple[str, ...]:
    match = _GRADIENT.fullmatch(field_name)
    return (field_name, f"grad({match.group(1)})") if match else (field_name,)


def _field_exists(case_dir: Path, physical_time: float, field_name: str) -> bool:
    for candidate in _field_names(field_name):
        try:
            direct = _time_path(case_dir, physical_time) / candidate
        except FileNotFoundError:
            direct = case_dir / "__missing__"
        if direct.is_file():
            return True
        processors = processor_directories(case_dir)
        if processors:
            try:
                paths = [
                    _time_path(processor, physical_time) / candidate
                    for processor in processors
                ]
            except FileNotFoundError:
                paths = []
            if paths and all(path.is_file() for path in paths):
                return True
    return False


def _read_case_field(case_dir: Path, physical_time: float, field_name: str) -> tuple[np.ndarray, str, str | None]:
    try:
        time_dir = _time_path(case_dir, physical_time)
    except FileNotFoundError:
        time_dir = case_dir / "__missing__"
    root_error: OpenFOAMDecodeError | None = None
    for candidate in _field_names(field_name):
        direct = time_dir / candidate
        if direct.is_file():
            try:
                return _read_values(direct)
            except OpenFOAMDecodeError as error:
                root_error = error
    if not processor_directories(case_dir):
        if root_error is not None:
            raise root_error
        raise FileNotFoundError(f"OpenFOAM field {field_name!r} was not found at time {physical_time:g}")
    for candidate in _field_names(field_name):
        if _field_exists(case_dir, physical_time, candidate):
            return _reconstruct_processor_field(case_dir, physical_time, candidate)
    if root_error is not None:
        raise root_error
    raise FileNotFoundError(f"OpenFOAM field {field_name!r} was not found at time {physical_time:g}")


def _missing_fields(case_dir: Path, physical_times: Sequence[float], fields: Sequence[str]) -> tuple[tuple[str, float], ...]:
    return tuple(
        (field_name, physical_time)
        for physical_time in physical_times
        for field_name in fields
        if not _field_exists(case_dir, physical_time, field_name)
    )


def _dictionary_block(payload: str, name: str) -> str | None:
    match = re.search(rf"\b{re.escape(name)}\s*\{{", payload)
    if match is None:
        return None
    depth = 1
    for index in range(match.end(), len(payload)):
        character = payload[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return payload[match.end() : index]
    return None


def _control_output_function(case_dir: Path, field_name: str) -> bool:
    control_dict = case_dir / "system/controlDict"
    if not control_dict.is_file():
        return None
    payload = re.sub(
        r"//.*?$|/\*.*?\*/",
        "",
        control_dict.read_text(encoding="utf-8", errors="replace"),
        flags=re.MULTILINE | re.DOTALL,
    )
    functions = _dictionary_block(payload, "functions")
    if functions is None:
        return None
    block = _dictionary_block(functions, field_name)
    return block is not None and re.search(rf"\b(?:field|result)\s+{re.escape(field_name)}\s*;", block) is not None


def _cube_root_vol_coefficient(case_dir: Path) -> float | None:
    properties = case_dir / "constant/turbulenceProperties"
    if not properties.is_file():
        return None
    payload = re.sub(
        r"//.*?$|/\*.*?\*/",
        "",
        properties.read_text(encoding="utf-8", errors="replace"),
        flags=re.MULTILINE | re.DOTALL,
    )
    if re.search(r"\bdelta\s+cubeRootVol\s*;", payload) is None:
        return None
    coefficients = _dictionary_block(payload, "cubeRootVolCoeffs")
    if coefficients is None:
        return 1.0
    match = _DELTA_COEFFICIENT.search(coefficients)
    return 1.0 if match is None else float(match.group(1))


def _application(case_dir: Path) -> str:
    control_dict = case_dir / "system/controlDict"
    payload = control_dict.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\bapplication\s+([^;\s]+)\s*;", payload)
    if match is None:
        raise ValueError(f"{control_dict} does not declare an OpenFOAM application")
    return match.group(1)


def _derivation(case_dir: Path, field_name: str) -> tuple[str, float | None] | None:
    if field_name == "C":
        return "writeCellCentres", None
    if field_name == "delta":
        if _control_output_function(case_dir, "delta"):
            return "controlDict", None
        coefficient = _cube_root_vol_coefficient(case_dir)
        if coefficient is not None:
            return "cubeRootVol", coefficient
    match = _GRADIENT.fullmatch(field_name)
    if match:
        if _control_output_function(case_dir, field_name):
            return "controlDict", None
        return f"grad({match.group(1)})", None
    return None


def _scalar_field(name: str, values: np.ndarray) -> str:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    body = "\n".join(f"{value:.17g}" for value in flattened)
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object {name};
}}
dimensions [0 1 0 0 0 0 0];
internalField nonuniform List<scalar>
{len(flattened)}
(
{body}
)
;
boundaryField
{{
}}
"""


def _derive_cube_root_vol(case_dir: Path, physical_time: float, coefficient: float) -> None:
    _post_process(case_dir, "writeCellVolumes", physical_time)
    locations = [case_dir, *processor_directories(case_dir)]
    for location in locations:
        try:
            time_dir = _time_path(location, physical_time)
        except FileNotFoundError:
            continue
        volume = time_dir / "V"
        if volume.is_file():
            values, _, _ = _read_values(volume)
            (time_dir / "delta").write_text(
                _scalar_field("delta", coefficient * np.cbrt(values)),
                encoding="utf-8",
            )


def _copy_case_entry(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination)


def _stage_case(case_dir: Path, destination: Path, physical_times: Sequence[float]) -> Path:
    staged = destination / "case"
    staged.mkdir(parents=True)
    for name in ("constant", "system"):
        source = case_dir / name
        if source.is_dir():
            _copy_case_entry(source, staged / name)
    for physical_time in physical_times:
        source = _time_path(case_dir, physical_time)
        _copy_case_entry(source, staged / source.name)
    for processor in processor_directories(case_dir):
        staged_processor = staged / processor.name
        staged_processor.mkdir()
        constant = processor / "constant"
        if constant.is_dir():
            _copy_case_entry(constant, staged_processor / "constant")
        for physical_time in physical_times:
            source = _time_path(processor, physical_time)
            _copy_case_entry(source, staged_processor / source.name)
    control_dict = staged / "system/controlDict"
    if control_dict.is_file():
        payload = control_dict.read_text(encoding="utf-8", errors="replace")
        payload = re.sub(
            r"\b(executeControl|writeControl)\s+writeTime\s*;",
            r"\1 timeStep;",
            payload,
        )
        control_dict.write_text(payload, encoding="utf-8")
    return staged


def _post_process(case_dir: Path, function: str | None, physical_time: float) -> None:
    if function is None:
        command = [
            _application(case_dir),
            "-case",
            str(case_dir),
            "-postProcess",
            "-time",
            f"{physical_time:g}",
        ]
    else:
        command = [
            "postProcess",
            "-case",
            str(case_dir),
            "-time",
            f"{physical_time:g}",
        ]
        command.extend(("-func", function))
    processors = processor_directories(case_dir)
    if processors:
        command.append("-parallel")
    if processors:
        ranks = str(len(processors))
        if os.environ.get("SLURM_JOB_ID") and shutil.which("srun"):
            available = int(
                os.environ.get(
                    "SLURM_CPUS_ON_NODE",
                    os.environ.get("SLURM_CPUS_PER_TASK", "0"),
                )
            )
            tasks = int(os.environ.get("SLURM_NTASKS", "0"))
            if available and available < len(processors):
                raise RuntimeError(
                    f"Parallel field derivation needs {len(processors)} Slurm CPUs for {len(processors)} processor directories, but the allocation has {available}."
                )
            if tasks and tasks < len(processors):
                raise RuntimeError(
                    f"Parallel field derivation needs {len(processors)} Slurm tasks for {len(processors)} processor directories, but the allocation has {tasks}."
                )
            command = ["srun", "--ntasks", ranks, *command]
        elif shutil.which("mpirun"):
            command = ["mpirun", "-np", ranks, *command]
        else:
            raise RuntimeError(
                "Parallel field derivation requires srun in a Slurm allocation or mpirun"
            )
    toolchain = _configured_toolchain()
    if toolchain.command:
        invocation = toolchain_shell(toolchain, quote_command(command))
    elif function is None or shutil.which("postProcess"):
        invocation = command
    elif shutil.which("openfoam"):
        invocation = ["openfoam", "-c", shlex.join(command)]
    else:
        raise RuntimeError(
            "OpenFOAM postProcess is unavailable; call OpenFOAM.setup(...) or load OpenFOAM before Dataset.parse"
        )
    subprocess.run(invocation, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def _derive_missing(
    case_dir: Path,
    output_dir: Path,
    physical_times: tuple[float, ...],
    fields: tuple[str, ...],
    *,
    coordinates: bool,
) -> Path:
    missing = list(_missing_fields(case_dir, physical_times, fields))
    coordinate_times: tuple[float, ...] = ()
    if coordinates:
        try:
            _coordinate_work(case_dir, physical_times)
        except FileNotFoundError:
            coordinate_times = physical_times if _mesh_changes(case_dir, physical_times) else (min(numeric_times(case_dir), default=physical_times[0]),)
            missing.extend(("C", physical_time) for physical_time in coordinate_times)
    if not missing:
        return case_dir
    unsupported = sorted({field_name for field_name, _ in missing if _derivation(case_dir, field_name) is None})
    if unsupported:
        names = ", ".join(unsupported)
        raise FileNotFoundError(f"OpenFOAM fields are missing and cannot be derived: {names}")
    for field_name, physical_time in missing:
        print(f"[FoamNordic] Sailor reported: {field_name} is missing at t = {physical_time:g}.")
    staging = output_dir / f".dataset-staging-{secrets.token_hex(4)}"
    required_times = tuple(sorted({physical_time for _, physical_time in missing}.union(physical_times)))
    completed = False
    try:
        staged = _stage_case(case_dir, staging, required_times)
        for field_name, physical_time in missing:
            derivation = _derivation(case_dir, field_name)
            assert derivation is not None
            function, coefficient = derivation
            if function == "cubeRootVol":
                assert coefficient is not None
                print(f"[FoamNordic] Captain ordered: delta = {coefficient:g} * cbrt(V) at t = {physical_time:g}.")
                _derive_cube_root_vol(staged, physical_time, coefficient)
            elif function == "controlDict":
                print(f"[FoamNordic] Captain ordered: configured functions at t = {physical_time:g}.")
                _post_process(staged, None, physical_time)
            else:
                print(f"[FoamNordic] Captain ordered: {function} at t = {physical_time:g}.")
                _post_process(staged, function, physical_time)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() if error.stderr else "OpenFOAM postProcess failed"
        raise RuntimeError(f"Dataset field derivation failed: {detail}") from error
    else:
        completed = True
        return staged
    finally:
        if not completed:
            shutil.rmtree(staging, ignore_errors=True)


def _mesh_changes(case_dir: Path, physical_times: Sequence[float]) -> bool:
    directories = (case_dir, *processor_directories(case_dir))
    for directory in directories:
        for physical_time in physical_times:
            try:
                time_dir = _time_path(directory, physical_time)
            except FileNotFoundError:
                continue
            if (time_dir / "polyMesh").is_dir():
                return True
    return False


def _coordinate_work(case_dir: Path, physical_times: tuple[float, ...]) -> tuple[tuple[int | None, float], ...]:
    if _mesh_changes(case_dir, physical_times):
        return tuple(enumerate(physical_times))
    candidates = tuple(sorted(numeric_times(case_dir))) or physical_times
    for physical_time in candidates:
        try:
            _read_case_field(case_dir, physical_time, "C")
        except FileNotFoundError:
            continue
        return ((None, physical_time),)
    raise FileNotFoundError(
        "Cell centres are unavailable. Store OpenFOAM field C or set coordinates=False."
    )


def _coordinates(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("OpenFOAM field C must be a three-component cell-centre vector")
    return values


def _dataset_workers(n_jobs: int | None, work_count: int, *, decomposed: bool) -> int:
    if n_jobs is None:
        return 1
    if n_jobs != -1:
        return n_jobs
    storage_cap = 2 if decomposed else 4
    return min(work_count, storage_cap, os.cpu_count() or 1)


@dataclass(frozen=True, slots=True)
class _Dataset:
    """A compact, table-like history of OpenFOAM internal fields."""

    _container: FData

    @property
    def path(self) -> Path:
        return self._container.path

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self._container.metadata["fields"])

    @property
    def times(self) -> tuple[float, ...]:
        return tuple(self._container.metadata["times"])

    @property
    def note(self) -> str | None:
        return self._container.metadata.get("note")

    @property
    def coordinates(self) -> bool:
        return bool(self._container.metadata.get("coordinates"))

    def load(
        self,
        *,
        fields: Sequence[str] | None = None,
        times: Sequence[float] | tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Load selected rows as a structured NumPy table with `time` and `cell` columns."""

        selected_fields = self.fields if fields is None else _normalise_fields(fields)
        missing = set(selected_fields).difference(self.fields)
        if missing:
            raise KeyError(f"FData fields are unavailable: {', '.join(sorted(missing))}")
        available_times = self.times
        if times is None:
            selected_times = available_times
        else:
            values = tuple(float(value) for value in times)
            if len(values) == 2 and values[0] <= values[1]:
                selected_times = tuple(
                    value for value in available_times if values[0] <= value <= values[1]
                )
            else:
                selected_times = tuple(
                    value
                    for value in available_times
                    if any(
                        math.isclose(
                            value,
                            requested,
                            rel_tol=1.0e-12,
                            abs_tol=1.0e-14,
                        )
                        for requested in values
                    )
                )
        rows = []
        descriptors: list[tuple[object, ...]] | None = None
        for time_index, physical_time in enumerate(available_times):
            if physical_time not in selected_times:
                continue
            arrays = [self._container.read(f"{time_index}/{name}") for name in selected_fields]
            count = len(arrays[0])
            if any(len(array) != count for array in arrays[1:]):
                raise ValueError(f"FData time {physical_time:g} has incompatible field lengths")
            if descriptors is None:
                descriptors = [("time", "<f8"), ("cell", "<i8")]
                if self.coordinates:
                    descriptors.extend((("x", "<f8"), ("y", "<f8"), ("z", "<f8")))
                descriptors.extend((name, array.dtype, array.shape[1:]) for name, array in zip(selected_fields, arrays, strict=True))
            table = np.empty(count, dtype=descriptors)
            table["time"] = physical_time
            table["cell"] = np.arange(count, dtype=np.int64)
            if self.coordinates:
                key = f"{time_index}/coordinates" if self._container.metadata["coordinates"]["time_varying"] else "coordinates"
                centres = self._container.read(key)
                if len(centres) != count:
                    raise ValueError(f"FData time {physical_time:g} has incompatible cell-centre coordinates")
                table["x"], table["y"], table["z"] = centres.T
            for name, array in zip(selected_fields, arrays, strict=True):
                table[name] = array
            rows.append(table)
        if not rows:
            raise ValueError("times selected no stored FData rows")
        return np.concatenate(rows)


def parse(
    *,
    case_dir: str | Path,
    fields: Sequence[str],
    times: Sequence[float] | tuple[float, float] | None = None,
    coordinates: bool = True,
    derive_missing: bool = True,
    output_dir: str | Path | None = None,
    note: str | None = None,
    n_jobs: int | None = None,
    progress: bool = False,
    overwrite: bool = False,
) -> _Dataset:
    """Parse an OpenFOAM case into one lossless `dataset.fdata` container."""

    if n_jobs is not None and (
        isinstance(n_jobs, bool)
        or not isinstance(n_jobs, int)
        or n_jobs == 0
        or n_jobs < -1
    ):
        raise ValueError("n_jobs must be -1, a positive integer, or None")
    if not isinstance(progress, bool):
        raise TypeError("progress must be a boolean")
    if not isinstance(coordinates, bool):
        raise TypeError("coordinates must be a boolean")
    if not isinstance(derive_missing, bool):
        raise TypeError("derive_missing must be a boolean")
    selected_case = Path(case_dir).expanduser().resolve()
    if not selected_case.is_dir():
        raise FileNotFoundError(f"OpenFOAM case directory does not exist: {selected_case}")
    selected_output = selected_case / "dataset" if output_dir is None else Path(output_dir).expanduser().resolve()
    destination = selected_output / "dataset.fdata"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"FData dataset already exists: {destination}")
    selected_fields = _normalise_fields(fields)
    selected_times = _normalise_times(selected_case, times)
    working_case = (
        _derive_missing(
            selected_case,
            selected_output,
            selected_times,
            selected_fields,
            coordinates=coordinates,
        )
        if derive_missing
        else selected_case
    )
    field_work = [
        ("field", time_index, physical_time, name)
        for time_index, physical_time in enumerate(selected_times)
        for name in selected_fields
    ]
    coordinate_work = _coordinate_work(working_case, selected_times) if coordinates else ()
    work = field_work + [("coordinates", index, physical_time, "C") for index, physical_time in coordinate_work]
    workers = _dataset_workers(
        n_jobs,
        len(work),
        decomposed=bool(processor_directories(working_case)),
    )
    if n_jobs == -1:
        print(f"[FoamNordic] Dataset workers: n_jobs=-1 resolved to {workers}.")

    def read_one(item: tuple[str, int | None, float, str]) -> tuple[str, int | None, str, np.ndarray, str, str | None]:
        kind, time_index, physical_time, name = item
        values, field_class, dimensions = _read_case_field(working_case, physical_time, name)
        return kind, time_index, name, values, field_class, dimensions

    metadata_fields: dict[str, dict[str, object]] = {}
    metadata: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "note": note,
        "fields": list(selected_fields),
        "times": list(selected_times),
        "field_metadata": metadata_fields,
        "coordinates": {"components": ["x", "y", "z"], "time_varying": len(coordinate_work) > 1} if coordinates else None,
        "layout": "time-cell",
    }
    reporter = osm.Progress(total=len(work), desc="FoamNordic dataset") if progress else None

    def arrays() -> Iterable[tuple[str, np.ndarray]]:
        try:
            for item in work:
                kind, time_index, name, values, field_class, dimensions = read_one(item)
                if kind == "coordinates":
                    key = f"{time_index}/coordinates" if time_index is not None else "coordinates"
                    yield key, _coordinates(values)
                else:
                    assert time_index is not None
                    contract = {
                        "class": field_class,
                        "dimensions": dimensions,
                        "shape": list(values.shape[1:]),
                        "dtype": values.dtype.str,
                    }
                    previous = metadata_fields.setdefault(name, contract)
                    if previous != contract:
                        raise ValueError(
                            f"OpenFOAM field {name!r} changes its stored contract across selected times"
                        )
                    yield f"{time_index}/{name}", values
                if reporter is not None:
                    reporter.update()
        finally:
            if reporter is not None:
                reporter.finish()
    try:
        result = _Dataset(
            write_fdata(
                destination,
                metadata=metadata,
                arrays=arrays(),
                workers=workers,
            )
        )
        return result
    finally:
        if working_case != selected_case:
            shutil.rmtree(working_case.parent, ignore_errors=True)


def read(*, path: str | Path) -> _Dataset:
    """Open one previously created `dataset.fdata` container."""

    return _Dataset(read_fdata(path))


def __dir__() -> list[str]:
    return ["load", "parse", "read"]


def load(*, path: str | Path, fields: Sequence[str] | None = None, times: Sequence[float] | tuple[float, float] | None = None) -> np.ndarray:
    """Open and load rows from one `dataset.fdata` container."""

    return read(path=path).load(fields=fields, times=times)
