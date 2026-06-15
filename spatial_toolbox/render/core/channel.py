"""Shared render-channel names used across render backends."""
from __future__ import annotations

from typing import Literal

RGB = "rgb"
DEPTH = "depth"
NORMAL = "normal"
SEGMENTATION = "segmentation"

Render_Channel = Literal["rgb", "depth", "normal", "segmentation"]
