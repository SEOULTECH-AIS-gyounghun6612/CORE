"""Lazy Blender Python session holder."""
from __future__ import annotations

from typing import Any


class Blender_Session:
    """Caches the imported ``bpy`` module for renderer use."""

    def __init__(self) -> None:
        self._bpy: Any | None = None

    @property
    def bpy(self) -> Any:
        """Imports and returns ``bpy`` on first access."""
        if self._bpy is None:
            import bpy

            self._bpy = bpy
        return self._bpy
