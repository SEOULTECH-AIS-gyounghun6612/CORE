"""Depth pass for the OpenGL renderer."""
from __future__ import annotations

import numpy as np

from ....scene.node.type.camera import Camera
from ._base import OpenGL_Base_Pass


class Depth_Pass(OpenGL_Base_Pass):
    """Renders a linearized depth image."""
    name = "depth"
    _readback_format = "depth"
    _use_lighting = False

    def _On_readback(self, width: int, height: int, **kwargs) -> np.ndarray:
        """Reads and linearizes depth values using camera clip planes."""
        _raw = super()._On_readback(width, height, **kwargs)
        _cam: Camera = kwargs["camera_node"]
        if not isinstance(_cam, Camera) or _cam.intrinsic is None:
            raise ValueError
        _near = _cam.intrinsic.near_clip
        _far = _cam.intrinsic.far_clip
        _linear = (2.0 * _near * _far) / (_far + _near - _raw * (_far - _near))
        _linear[_raw >= 1.0] = 0.0
        return _linear
