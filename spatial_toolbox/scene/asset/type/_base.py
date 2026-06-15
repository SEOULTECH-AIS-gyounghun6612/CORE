"""Base asset schema shared by asset loaders and caches."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from python_toolbox.data_schema import Data_Schema


@dataclass
class Base_Asset(Data_Schema):
    """Base class for file-backed assets.

    Attributes:
        label: Human-readable asset label.
        source_path: Original source path or cache key.
        unit_length: Asset-space unit scale.
    """

    label: str = "asset"
    source_path: str | None = None
    unit_length: float = 1.0

    __exclude_serialize__: ClassVar[set[str]] = set()

__all__ = ["Base_Asset"]
