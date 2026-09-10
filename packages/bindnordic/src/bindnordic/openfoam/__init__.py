"""Focused OpenFOAM field-file I/O independent of solver orchestration."""

from .field import FoamField, OpenFOAMDecodeError, read_field

__all__ = ["FoamField", "OpenFOAMDecodeError", "read_field"]
