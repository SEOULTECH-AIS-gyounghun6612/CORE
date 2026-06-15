"""Shared scene-serialization payload types."""
from __future__ import annotations

from dataclasses import dataclass

from ..node.type._base import Base_Node


@dataclass
class Scene_State:
    """Transport object used when loading serialized scenes.

    Attributes:
        root: Root node of the imported scene graph.
        unit_length: Imported scene unit scale.
    """

    root: Base_Node
    unit_length: float = 1.0
