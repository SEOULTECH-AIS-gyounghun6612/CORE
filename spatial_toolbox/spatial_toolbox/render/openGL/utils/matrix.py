"""Matrix helpers for the OpenGL backend."""
from __future__ import annotations

import numpy as np

from ....scene.node.type.camera import Camera_Intrinsic


def to_gl_matrix(matrix: np.ndarray) -> np.ndarray:
    """Converts a matrix into contiguous OpenGL column-major layout."""
    return np.ascontiguousarray(matrix.astype(np.float32).T)


def Build_gl_projection(
    intrinsic: Camera_Intrinsic, unit_length: float = 1.0
) -> np.ndarray:
    """Builds an OpenGL projection matrix from camera intrinsics."""
    _near = intrinsic.near_clip / unit_length
    _far = intrinsic.far_clip / unit_length
    _d = _far - _near
    _cx_n = 1.0 - 2.0 * intrinsic.cx / intrinsic.width
    _cy_n = 2.0 * intrinsic.cy / intrinsic.height - 1.0
    return np.array([
        [2*intrinsic.fx/intrinsic.width,                 0,   _cx_n,               0],
        [                             0, 2*intrinsic.fy/intrinsic.height, _cy_n,    0],
        [                             0,                 0, -(_far+_near)/_d, -2*_far*_near/_d],
        [                             0,                 0,              -1,               0],
    ], dtype=np.float32).T
