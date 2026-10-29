"""Slurm memory units used by resource estimates and batch requests."""

import pytest

from foamnordic.execution.resources import memory_bytes, slurm_memory


@pytest.mark.parametrize("value", ["2048", "2048M", "2048MiB", "2G"])
def test_slurm_memory_defaults_to_mebibytes(value):
    assert memory_bytes(value) == 2 * 1024**3
    assert slurm_memory(memory_bytes(value)) == "2048M"


def test_explicit_bytes_and_unspecified_memory():
    assert memory_bytes("2048B") == 2048
    assert memory_bytes(None) is None
