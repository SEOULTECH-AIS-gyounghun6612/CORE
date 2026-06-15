"""Depth image loaders for Blender compositor outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .rgb import _Load_pixels


def Load_depth(bpy: Any, path: Path) -> np.ndarray:
    """Loads a depth output and normalizes invalid values to zero."""
    _rgba = _Load_pixels(bpy, path)
    _depth = np.array(_rgba[..., 0], dtype=np.float32)
    _depth[~np.isfinite(_depth)] = 0.0
    return _depth
