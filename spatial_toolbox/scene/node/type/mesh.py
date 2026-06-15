"""Mesh node type backed by cached geometry assets."""
from __future__ import annotations
from dataclasses import dataclass

from ..register import NODE_REGISTRY
from ._base import Base_Node, PrimType


@NODE_REGISTRY.Register_module("Mesh")
@dataclass
class Mesh(Base_Node):
    """Scene node that references geometry through ``source_key``."""

    prim_type: PrimType = "Mesh"

__all__ = ["Mesh"]
