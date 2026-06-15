"""Mesh asset type."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..register import ASSET_REGISTRY
from ._base import Base_Asset


@ASSET_REGISTRY.Register_module("Mesh")
@dataclass
class Mesh(Base_Asset):
    """Asset wrapper for mesh geometry."""

    geometry: Any | None = field(default=None, repr=False)

    __exclude_serialize__: ClassVar[set[str]] = {"geometry"}

__all__ = ["Mesh"]
