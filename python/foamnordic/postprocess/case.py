"""Read and summarize one completed OpenFOAM case."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ..core.paths import path_from
from ._field import available_times, read_case_field, select_time


def case_path(value: Any) -> Path:
    if isinstance(value, (str, Path)) or hasattr(value, "__fspath__"):
        path = path_from(value)
    elif hasattr(value, "case") and isinstance(value.case, Path):
        path = value.case
    elif hasattr(value, "case_dir"):
        path = Path(value.case_dir)
    else:
        raise TypeError(
            "postprocess case must be a path, OpenFOAM.Case, or FoamNordic Result"
        )
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"OpenFOAM case directory was not found: {path}")
    return path


def field_names(value: str | Iterable[str]) -> tuple[str, ...]:
    names = (value,) if isinstance(value, str) else tuple(value)
    if not names:
        raise ValueError("at least one field name is required")
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError("field names must be non-empty strings")
    return names


def reduced_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim <= 1:
        return array.reshape(-1)
    return np.linalg.norm(array.reshape(array.shape[0], -1), axis=1)


def display_statistics(
    title: str,
    rows: list[list[str]],
) -> None:
    import onsaemiro as osm

    table = osm.TableMaker(
        title=title,
        columns=["Field", "Min", "Max", "Mean", "Std", "RMS"],
        mode="static",
    )
    for row in rows:
        table.add_row(row)
    table.display()


class Case:
    """Read durable fields from one reconstructed or decomposed OpenFOAM case.

    ``source`` accepts a path, :class:`foamnordic.OpenFOAM.Case`, or completed
    :class:`foamnordic.Result`. Use :attr:`times` to discover stored physical
    times. Field access defaults to the final numerically sorted time.
    """

    def __init__(self, source: Any):
        self._path = case_path(source)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def times(self) -> tuple[float, ...]:
        return available_times(self._path)

    def _time(
        self,
        *,
        time_idx: int | None,
        physical_time: float | None,
    ) -> float:
        return select_time(
            self._path,
            time_idx=time_idx,
            physical_time=physical_time,
        )

    def field(
        self,
        name: str,
        *,
        time_idx: int | None = None,
        physical_time: float | None = None,
    ) -> np.ndarray:
        """Read one stored internal field as a NumPy array.

        ``time_idx`` selects the numerically sorted time-directory position
        and accepts negative indices. ``physical_time`` selects a matching
        OpenFOAM physical time. They are mutually exclusive; omitting both is
        equivalent to ``time_idx=-1``.
        """

        selected = self._time(time_idx=time_idx, physical_time=physical_time)
        return read_case_field(self._path, name, selected)

    def statistics(
        self,
        fields: str | Iterable[str],
        *,
        time_idx: int | None = None,
        physical_time: float | None = None,
        verbose: bool = False,
    ) -> dict[str, float] | dict[str, dict[str, float]]:
        """Calculate scalar or per-cell vector/tensor magnitude statistics.

        A single field returns one metric dictionary; multiple fields return
        dictionaries keyed by field name. ``verbose=True`` additionally
        displays the returned values as an Onsaemiro table.
        """

        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        names = field_names(fields)
        selected = self._time(time_idx=time_idx, physical_time=physical_time)
        result: dict[str, dict[str, float]] = {}
        rows: list[list[str]] = []
        for name in names:
            values = reduced_values(read_case_field(self._path, name, selected))
            metrics = {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "rms": float(np.sqrt(np.mean(values**2))),
            }
            result[name] = metrics
            rows.append([name, *(f"{metrics[key]:.6e}" for key in metrics)])
        if verbose:
            display_statistics(f"OpenFOAM Field Statistics (t={selected:g})", rows)
        return result if len(names) > 1 else result[names[0]]


__all__ = ["Case"]
