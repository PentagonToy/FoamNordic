"""Baseline-to-model OpenFOAM field comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._field import MESH_FILES, digest, processor_directories
from .case import Case, field_names, reduced_values


def _require_same_file(reference: Path, candidate: Path, description: str) -> None:
    if not reference.is_file() or not candidate.is_file():
        raise ValueError(f"strict mesh comparison requires {description}")
    if digest(reference) != digest(candidate):
        raise ValueError(f"strict mesh comparison found different {description}")


def _mesh_directory(reference: Path, candidate: Path, description: str) -> None:
    for name in MESH_FILES:
        _require_same_file(reference / name, candidate / name, f"{description}/{name}")


def _strict_mesh(reference: Path, candidate: Path) -> None:
    reference_processors = processor_directories(reference)
    candidate_processors = processor_directories(candidate)
    if bool(reference_processors) != bool(candidate_processors):
        raise ValueError("cannot strictly compare reconstructed and decomposed cases")
    if not reference_processors:
        _mesh_directory(
            reference / "constant/polyMesh",
            candidate / "constant/polyMesh",
            "constant/polyMesh",
        )
        return
    if len(reference_processors) != len(candidate_processors):
        raise ValueError("decomposed cases use different processor counts")
    for reference_processor, candidate_processor in zip(
        reference_processors, candidate_processors, strict=True
    ):
        name = reference_processor.name
        reference_mesh = reference_processor / "constant/polyMesh"
        candidate_mesh = candidate_processor / "constant/polyMesh"
        _mesh_directory(reference_mesh, candidate_mesh, f"{name}/constant/polyMesh")
        _require_same_file(
            reference_mesh / "cellProcAddressing",
            candidate_mesh / "cellProcAddressing",
            f"{name}/constant/polyMesh/cellProcAddressing",
        )


def _display(rows: list[list[str]], physical_time: float) -> None:
    import onsaemiro as osm

    table = osm.TableMaker(
        title=f"OpenFOAM Case Comparison (t={physical_time:g})",
        columns=["Field", "MAE", "RMSE", "Max Abs", "Relative L2"],
        mode="static",
    )
    for row in rows:
        table.add_row(row)
    table.display()


def compare(
    reference: Any,
    candidate: Any,
    *,
    fields: str | tuple[str, ...] | list[str],
    time_idx: int | None = None,
    physical_time: float | None = None,
    mesh: str = "strict",
    verbose: bool = False,
) -> dict[str, dict[str, float]]:
    """Compare matching stored fields from a baseline and model result.

    ``reference`` and ``candidate`` accept paths, OpenFOAM case declarations,
    completed Results, or existing :class:`Case` readers. ``time_idx`` and
    ``physical_time`` are mutually exclusive. ``mesh='strict'`` verifies mesh
    identity before cell-wise comparison; ``mesh='shape'`` explicitly checks
    field shapes only. ``verbose=True`` adds an Onsaemiro table.
    """

    if mesh not in {"strict", "shape"}:
        raise ValueError("mesh must be 'strict' or 'shape'")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean")
    reference_case = reference if isinstance(reference, Case) else Case(reference)
    candidate_case = candidate if isinstance(candidate, Case) else Case(candidate)
    reference_time = reference_case._time(
        time_idx=time_idx, physical_time=physical_time
    )
    candidate_time = candidate_case._time(
        time_idx=time_idx, physical_time=physical_time
    )
    if not np.isclose(reference_time, candidate_time, rtol=1.0e-12, atol=1.0e-14):
        raise ValueError(
            f"reference and candidate physical times differ: "
            f"{reference_time:g} != {candidate_time:g}"
        )
    if mesh == "strict":
        _strict_mesh(reference_case.path, candidate_case.path)
    result: dict[str, dict[str, float]] = {}
    rows: list[list[str]] = []
    for name in field_names(fields):
        reference_values = np.asarray(
            reference_case.field(name, physical_time=reference_time), dtype=np.float64
        )
        candidate_values = np.asarray(
            candidate_case.field(name, physical_time=candidate_time), dtype=np.float64
        )
        if reference_values.shape != candidate_values.shape:
            raise ValueError(
                f"field {name!r} shapes differ: "
                f"{reference_values.shape} != {candidate_values.shape}"
            )
        difference = candidate_values - reference_values
        errors = reduced_values(difference)
        reference_norm = float(np.linalg.norm(reference_values.reshape(-1)))
        difference_norm = float(np.linalg.norm(difference.reshape(-1)))
        metrics = {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "max_abs": float(np.max(np.abs(errors))),
            "relative_l2": (
                difference_norm / reference_norm
                if reference_norm
                else (0.0 if difference_norm == 0.0 else float("inf"))
            ),
        }
        result[name] = metrics
        rows.append([name, *(f"{metrics[key]:.6e}" for key in metrics)])
    if verbose:
        _display(rows, reference_time)
    return result
