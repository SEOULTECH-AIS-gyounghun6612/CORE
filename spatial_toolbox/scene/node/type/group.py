"""Structural group node type."""
from __future__ import annotations
from dataclasses import dataclass

from ..register import NODE_REGISTRY
from ._base import Base_Node


@NODE_REGISTRY.Register_module("Xform")
@dataclass
class Group(Base_Node):
    """Convenience type for structural ``Xform`` nodes."""

    pass

__all__ = ["Group"]
