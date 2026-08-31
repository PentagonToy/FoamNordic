"""Bounded file-backed observation consumption without context management."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from types import MappingProxyType
import time
from typing import Iterator, Mapping, TextIO, TYPE_CHECKING

if TYPE_CHECKING:
    from .run import Run


@dataclass(frozen=True, slots=True)
class FieldSummary:
    """Available reductions for one observed field."""

    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    l2: float | None = None
    count: int | None = None


@dataclass(frozen=True, slots=True)
class ObservationTiming:
    """Native timing values attached to one observation record."""

    closure_wait: float = 0.0
    evaluate: float = 0.0


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    """One immutable summary record copied out of the solver path."""

    exchange_index: int
    time: float
    summary: Mapping[str, FieldSummary]
    timing: ObservationTiming

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ObservationRecord:
        summaries = {}
        raw_summaries = value.get("summary", {})
        if not isinstance(raw_summaries, Mapping):
            raise ValueError("observation summary must be a mapping")
        for name, reductions in raw_summaries.items():
            if not isinstance(reductions, Mapping):
                raise ValueError("field reductions must be a mapping")
            summaries[str(name)] = FieldSummary(
                minimum=_optional_float(reductions.get("minimum", reductions.get("min"))),
                maximum=_optional_float(reductions.get("maximum", reductions.get("max"))),
                mean=_optional_float(reductions.get("mean")),
                l2=_optional_float(reductions.get("l2")),
                count=_optional_int(reductions.get("count")),
            )
        raw_timing = value.get("timing", {})
        if not isinstance(raw_timing, Mapping):
            raise ValueError("observation timing must be a mapping")
        return cls(
            exchange_index=int(value["exchange_index"]),
            time=float(value["time"]),
            summary=MappingProxyType(summaries),
            timing=ObservationTiming(
                closure_wait=float(raw_timing.get("closure_wait", 0.0)),
                evaluate=float(raw_timing.get("evaluate", 0.0)),
            ),
        )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


class ObservationStream:
    """Iterate JSONL observations until the associated run terminates."""

    def __init__(
        self,
        run: Run,
        path: Path,
        *,
        poll_interval: float,
        expected_sources: int = 1,
        progress: bool = False,
        progress_stream: TextIO | None = None,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("observation poll_interval must be positive")
        if not isinstance(progress, bool):
            raise TypeError("observation progress must be a boolean")
        self._run = run
        self.path = Path(path)
        self.poll_interval = poll_interval
        self.expected_sources = expected_sources
        self._progress = progress
        self._progress_stream = (
            progress_stream if progress_stream is not None else sys.stdout
        )
        self._progress_width = 0
        self._progress_rendered = False
        self._closed = False

    def __iter__(self) -> Iterator[ObservationRecord]:
        offsets: dict[Path, int] = {}
        pending: dict[int, dict[Path, ObservationRecord]] = {}
        terminal_seen = False
        try:
            while not self._closed:
                emitted = False
                # A completed run is renamed from its private preparation path to
                # its durable local/Slurm identity. Follow that atomic rename so
                # an observation iterator opened before completion can drain the
                # final records afterwards.
                current = self._run._work_dir / "observations" / self.path.name
                if current != self.path:
                    previous_parent = self.path.parent
                    self.path = current
                    offsets = {
                        current.parent / path.name: offset
                        for path, offset in offsets.items()
                        if path.parent == previous_parent
                    }
                    pending = {
                        exchange_index: {
                            current.parent / path.name: record
                            for path, record in group.items()
                        }
                        for exchange_index, group in pending.items()
                    }
                paths = sorted(self.path.parent.glob(f"{self.path.stem}*.jsonl"))
                for path in paths:
                    try:
                        with path.open("r", encoding="utf-8") as stream:
                            stream.seek(offsets.get(path, 0))
                            while True:
                                line = stream.readline()
                                if not line or not line.endswith("\n"):
                                    break
                                offsets[path] = stream.tell()
                                if not line.strip():
                                    continue
                                record = ObservationRecord.from_dict(json.loads(line))
                                if self.expected_sources == 1:
                                    emitted = True
                                    self._render_progress(record)
                                    yield record
                                    continue
                                group = pending.setdefault(record.exchange_index, {})
                                group[path] = record
                                if len(group) >= self.expected_sources:
                                    emitted = True
                                    merged = _merge_records(group.values())
                                    self._render_progress(merged)
                                    yield merged
                                    del pending[record.exchange_index]
                    except FileNotFoundError:
                        pass
                if self._run.status.value != "running":
                    if terminal_seen and not emitted:
                        for exchange_index in sorted(pending):
                            merged = _merge_records(pending[exchange_index].values())
                            self._render_progress(merged)
                            yield merged
                        pending.clear()
                        break
                    terminal_seen = True
                if not emitted:
                    time.sleep(self.poll_interval)
        finally:
            self._clear_progress()

    def _render_progress(self, record: ObservationRecord) -> None:
        if not self._progress:
            return
        message = (
            f"[FoamNordic] Observing OpenFOAM: t = {record.time:.6g}"
            f" | exchange = {record.exchange_index}"
        )
        padding = " " * max(0, self._progress_width - len(message))
        print(
            f"\r{message}{padding}",
            end="",
            flush=True,
            file=self._progress_stream,
        )
        self._progress_width = len(message)
        self._progress_rendered = True

    def _clear_progress(self) -> None:
        if not self._progress_rendered:
            return
        print(
            f"\r{' ' * self._progress_width}\r",
            end="",
            flush=True,
            file=self._progress_stream,
        )
        self._progress_rendered = False

    def close(self) -> None:
        """Stop local consumption; the solver and retained records are unaffected."""

        self._closed = True
        self._clear_progress()


def _merge_records(records) -> ObservationRecord:
    values = list(records)
    if not values:
        raise ValueError("cannot merge an empty observation group")
    names = set.intersection(*(set(record.summary) for record in values))
    summary = {}
    for name in sorted(names):
        fields = [record.summary[name] for record in values]
        counts = [field.count for field in fields]
        total = sum(count for count in counts if count is not None)
        means = [
            (field.mean, field.count)
            for field in fields
            if field.mean is not None and field.count is not None
        ]
        summary[name] = FieldSummary(
            minimum=min(
                field.minimum for field in fields if field.minimum is not None
            ),
            maximum=max(
                field.maximum for field in fields if field.maximum is not None
            ),
            mean=(
                sum(mean * count for mean, count in means) / total
                if means and total
                else None
            ),
            l2=(
                math.sqrt(sum(field.l2 * field.l2 for field in fields))
                if all(field.l2 is not None for field in fields)
                else None
            ),
            count=total or None,
        )
    return ObservationRecord(
        exchange_index=values[0].exchange_index,
        time=values[0].time,
        summary=MappingProxyType(summary),
        timing=ObservationTiming(
            closure_wait=max(record.timing.closure_wait for record in values),
            evaluate=max(record.timing.evaluate for record in values),
        ),
    )
