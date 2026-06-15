"""Point-cloud asset type."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from ..register import ASSET_REGISTRY
from ._base import Base_Asset


@ASSET_REGISTRY.Register_module("Point_Cloud")
@dataclass
class Point_Cloud(Base_Asset):
    """Asset wrapper for point-cloud data."""

    points: np.ndarray | None = field(default=None, repr=False)
    colors: np.ndarray | None = field(default=None, repr=False)
    normals: np.ndarray | None = field(default=None, repr=False)

    __exclude_serialize__: ClassVar[set[str]] = {"points", "colors", "normals"}

    @property
    def count(self) -> int:
        """Returns the number of stored points."""
        if self.points is None:
            return 0
        return len(self.points)

__all__ = ["Point_Cloud"]
