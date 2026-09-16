"""Lossless, indexed storage for FoamNordic field datasets."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Iterable
import zlib

import numpy as np


class FDataError(ValueError):
    """Raised when an FData container is incomplete or incompatible."""


_MAGIC = b"FNOFDAT1"
_TRAILER = struct.Struct("<8sQQ")
_TRAILER_MAGIC = b"FNOFEND1"
_COMPRESSION_BLOCK_BYTES = 16 * 1024 * 1024


def _encode_array(item: tuple[str, np.ndarray]) -> tuple[str, bytes, int, str, list[int]]:
    key, array = item
    contiguous = np.ascontiguousarray(array)
    compressor = zlib.compressobj(level=6)
    source = memoryview(contiguous).cast("B")
    encoded = bytearray()
    for start in range(0, contiguous.nbytes, _COMPRESSION_BLOCK_BYTES):
        encoded.extend(compressor.compress(source[start : start + _COMPRESSION_BLOCK_BYTES]))
    encoded.extend(compressor.flush())
    return key, bytes(encoded), contiguous.nbytes, contiguous.dtype.str, list(contiguous.shape)


@dataclass(frozen=True, slots=True)
class FDataChunk:
    """One independently compressed array in an FData container."""

    key: str
    offset: int
    compressed_bytes: int
    raw_bytes: int
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FData:
    """An indexed FData container with lazy chunk reads."""

    path: Path
    metadata: dict[str, object]
    chunks: dict[str, FDataChunk]

    def read(self, key: str) -> np.ndarray:
        """Read one named array without inflating unrelated chunks."""

        try:
            chunk = self.chunks[key]
        except KeyError as error:
            raise KeyError(f"FData array {key!r} was not found") from error
        with self.path.open("rb") as stream:
            stream.seek(chunk.offset)
            payload = stream.read(chunk.compressed_bytes)
        if len(payload) != chunk.compressed_bytes:
            raise FDataError(f"FData array {key!r} is truncated")
        raw = zlib.decompress(payload)
        if len(raw) != chunk.raw_bytes:
            raise FDataError(f"FData array {key!r} has an invalid uncompressed size")
        return np.frombuffer(raw, dtype=np.dtype(chunk.dtype)).reshape(chunk.shape).copy()


def read_fdata(path: str | Path) -> FData:
    """Open an FData container and read only its compact footer index."""

    selected = Path(path).expanduser().resolve()
    with selected.open("rb") as stream:
        if stream.read(len(_MAGIC)) != _MAGIC:
            raise FDataError(f"{selected} is not an FData container")
        stream.seek(-_TRAILER.size, 2)
        trailer = stream.read(_TRAILER.size)
        if len(trailer) != _TRAILER.size:
            raise FDataError(f"{selected} has no FData footer")
        marker, footer_offset, footer_bytes = _TRAILER.unpack(trailer)
        if marker != _TRAILER_MAGIC:
            raise FDataError(f"{selected} has an invalid FData footer")
        stream.seek(footer_offset)
        encoded = stream.read(footer_bytes)
    try:
        footer = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FDataError(f"{selected} has an invalid FData index") from error
    if footer.get("schema_version") != 1:
        raise FDataError(f"{selected} has an unsupported FData schema")
    chunks = {
        item["key"]: FDataChunk(
            key=item["key"],
            offset=item["offset"],
            compressed_bytes=item["compressed_bytes"],
            raw_bytes=item["raw_bytes"],
            dtype=item["dtype"],
            shape=tuple(item["shape"]),
        )
        for item in footer["chunks"]
    }
    return FData(selected, footer["metadata"], chunks)


class FDataWriter:
    """Append independently compressed arrays to one atomic FData container."""

    def __init__(self, path: str | Path, *, metadata: dict[str, object], workers: int = 1) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self._path = Path(path).expanduser().resolve()
        self._temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        self._metadata = metadata
        self._workers = workers
        self._chunks: list[dict[str, object]] = []
        self._pending = set()
        self._executor: ThreadPoolExecutor | None = None
        self._stream = None

    def __enter__(self) -> FDataWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self._temporary.open("wb")
        self._stream.write(_MAGIC)
        self._executor = ThreadPoolExecutor(max_workers=self._workers)
        return self

    def append(self, key: str, array: np.ndarray) -> None:
        """Queue one array while bounding resident compression work."""

        if self._executor is None:
            raise RuntimeError("FDataWriter is not open")
        self._pending.add(self._executor.submit(_encode_array, (key, array)))
        if len(self._pending) >= self._workers:
            self._drain(final=False)

    def _drain(self, *, final: bool) -> None:
        if final:
            while self._pending:
                self._drain(final=False)
            return
        if not self._pending:
            return
        complete, self._pending = wait(
            self._pending,
            return_when=FIRST_COMPLETED,
        )
        for future in complete:
            key, compressed, raw_bytes, dtype, shape = future.result()
            assert self._stream is not None
            offset = self._stream.tell()
            self._stream.write(compressed)
            self._chunks.append(
                {
                    "key": key,
                    "offset": offset,
                    "compressed_bytes": len(compressed),
                    "raw_bytes": raw_bytes,
                    "dtype": dtype,
                    "shape": shape,
                }
            )

    def close(self) -> FData:
        """Finish the footer and atomically publish the container."""

        try:
            self._drain(final=True)
            if self._executor is not None:
                self._executor.shutdown()
                self._executor = None
            assert self._stream is not None
            footer = json.dumps(
                {"schema_version": 1, "metadata": self._metadata, "chunks": self._chunks},
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            footer_offset = self._stream.tell()
            self._stream.write(footer)
            self._stream.write(_TRAILER.pack(_TRAILER_MAGIC, footer_offset, len(footer)))
            self._stream.close()
            self._stream = None
            self._temporary.replace(self._path)
            return read_fdata(self._path)
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        """Discard an incomplete temporary container."""

        if self._executor is not None:
            self._executor.shutdown(cancel_futures=True)
            self._executor = None
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._temporary.unlink(missing_ok=True)

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        if exception_type is None:
            self.close()
        else:
            self.abort()


def write_fdata(
    path: str | Path,
    *,
    metadata: dict[str, object],
    arrays: Iterable[tuple[str, np.ndarray]],
    workers: int = 1,
) -> FData:
    """Write one atomic, lossless FData container from named arrays."""

    writer = FDataWriter(path, metadata=metadata, workers=workers)
    with writer:
        for key, array in arrays:
            writer.append(key, array)
    return read_fdata(path)
