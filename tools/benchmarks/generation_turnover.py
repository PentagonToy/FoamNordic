#!/usr/bin/env python3
"""Stress immutable runtime publication, pinning, and garbage collection."""

from __future__ import annotations

from pathlib import Path
import statistics
import tempfile
import time

from foamnordic.build.generations import collect_garbage, complete, pin, publish


GENERATIONS = 256
REPETITIONS = 7
KEEP = 2
PAYLOAD_BYTES = 64 * 1024


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def run_once(root: Path) -> tuple[float, float, float, int]:
    generations = root / "generations"
    generations.mkdir(parents=True)
    payload = bytes((index % 251 for index in range(PAYLOAD_BYTES)))
    publish_times = []
    gc_times = []
    removed_total = 0

    first = generations / "generation-0000"
    first.mkdir()
    (first / "runtime.bin").write_bytes(payload)
    publish(root, first, "generation-0000")
    permanent_marker = pin(first, "long-lived active run")
    rolling_marker = None
    rolling_generation = None

    started = time.perf_counter()
    for index in range(1, GENERATIONS):
        identity = f"generation-{index:04d}"
        generation = generations / identity
        generation.mkdir()
        (generation / "runtime.bin").write_bytes(payload)

        before = time.perf_counter()
        publish(root, generation, identity)
        publish_times.append(time.perf_counter() - before)

        if index % 16 == 0:
            next_marker = pin(generation, f"rolling run at {index}")
            if rolling_marker is not None:
                rolling_marker.unlink()
            rolling_marker = next_marker
            rolling_generation = generation

        before = time.perf_counter()
        removed_total += len(collect_garbage(root, keep=KEEP))
        gc_times.append(time.perf_counter() - before)

        current = (root / "current").resolve()
        if current != generation.resolve() or not complete(current, identity):
            raise RuntimeError("current generation changed or became incomplete")
        if not first.is_dir() or not permanent_marker.is_file():
            raise RuntimeError("long-lived pinned generation was collected")
        if rolling_generation is not None and not rolling_generation.is_dir():
            raise RuntimeError("rolling pinned generation was collected")
        if len(tuple(generations.iterdir())) > KEEP + 2:
            raise RuntimeError("garbage collection exceeded its retention bound")

    elapsed = time.perf_counter() - started
    permanent_marker.unlink()
    if rolling_marker is not None:
        rolling_marker.unlink()
    removed_total += len(collect_garbage(root, keep=KEEP))
    survivors = tuple(generations.iterdir())
    if len(survivors) != KEEP or first.exists():
        raise RuntimeError("released pins were not collected")
    if (root / "current").resolve() not in tuple(path.resolve() for path in survivors):
        raise RuntimeError("current generation was not retained")

    return (
        elapsed,
        statistics.median(publish_times),
        statistics.median(gc_times),
        removed_total,
    )


def main() -> None:
    elapsed = []
    publish_latency = []
    gc_latency = []
    removed = []
    with tempfile.TemporaryDirectory(prefix="foamnordic-generations-") as directory:
        base = Path(directory)
        for repetition in range(REPETITIONS):
            result = run_once(base / f"run-{repetition}")
            elapsed.append(result[0])
            publish_latency.append(result[1])
            gc_latency.append(result[2])
            removed.append(result[3])

    print(f"Series: {REPETITIONS}")
    print(f"Generations/series: {GENERATIONS}")
    print(f"Payload/generation: {PAYLOAD_BYTES / 1024:.0f} KiB")
    print(f"Retained generations: {KEEP}")
    print(f"Median series time: {statistics.median(elapsed):.6f} s")
    print(f"Maximum series deviation: {max(abs(value - statistics.median(elapsed)) for value in elapsed):.6f} s")
    print(f"Median publish latency: {statistics.median(publish_latency) * 1e6:.3f} us")
    print(f"Median GC latency: {statistics.median(gc_latency) * 1e6:.3f} us")
    print(f"Series-time p95: {percentile(elapsed, 0.95):.6f} s")
    print(f"Collected generations/series: {statistics.median(removed):.0f}")
    print("Current selector: PASS")
    print("Long-lived pin: PASS")
    print("Rolling pin: PASS")
    print("Released-pin collection: PASS")
    print("FoamNordic generation turnover stress: PASS")


if __name__ == "__main__":
    main()
