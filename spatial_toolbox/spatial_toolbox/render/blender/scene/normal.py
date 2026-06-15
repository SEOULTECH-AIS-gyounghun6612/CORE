"""Normal image loaders for Blender compositor outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .rgb import _Load_pixels


def Load_normal(bpy: Any, path: Path) -> np.ndarray:
    """Loads normals and encodes them into the shared ``uint8`` format."""
    _rgba = _Load_pixels(bpy, path)
    _normal = np.clip(_rgba[..., :3], -1.0, 1.0)
    _encoded = np.clip(np.rint((_normal + 1.0) * 127.5), 0.0, 255.0).astype(np.uint8)
    _background = _rgba[..., 3] <= 0.0
    _encoded[_background] = np.array([128, 128, 255], dtype=np.uint8)
    return _encoded
