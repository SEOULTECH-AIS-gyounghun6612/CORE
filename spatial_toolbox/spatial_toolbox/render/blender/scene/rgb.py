"""RGB image loaders for Blender compositor outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _Load_pixels(bpy: Any, path: Path) -> np.ndarray:
    """Loads a Blender image file into an ``H x W x 4`` float array."""
    _image = bpy.data.images.load(str(path), check_existing=False)
    try:
        _w, _h = int(_image.size[0]), int(_image.size[1])
        _pixels = np.array(_image.pixels[:], dtype=np.float32)
        return _pixels.reshape(_h, _w, 4)
    finally:
        bpy.data.images.remove(_image)


def Load_rgb(bpy: Any, path: Path) -> np.ndarray:
    """Loads an RGB output and converts it to ``uint8`` image space."""
    _rgba = _Load_pixels(bpy, path)
    return np.clip(np.rint(_rgba[..., :3] * 255.0), 0.0, 255.0).astype(np.uint8)
