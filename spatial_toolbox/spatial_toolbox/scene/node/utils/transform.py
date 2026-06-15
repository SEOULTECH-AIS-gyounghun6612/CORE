"""Transform helpers for scene nodes."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def Build_transform(
    tx: float = 0.0, ty: float = 0.0, tz: float = 0.0,
    rx: float = 0.0, ry: float = 0.0, rz: float = 0.0
) -> np.ndarray:
    """Builds a 4x4 transform from XYZ Euler angles and translation."""
    _m = np.eye(4, dtype=np.float32)
    _m[:3, :3] = Rotation.from_euler(
        "xyz", (rx, ry, rz), degrees=True
    ).as_matrix()
    _m[:3, 3] = (tx, ty, tz)
    return _m


def Decompose_transform(matrix: np.ndarray) -> tuple[float, ...]:
    """Decomposes a 4x4 transform into translation and XYZ Euler angles."""
    _tx, _ty, _tz = (float(_v) for _v in matrix[:3, 3])
    _rx, _ry, _rz = (
        float(_v) for _v in
        Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
    )
    return (_tx, _ty, _tz, _rx, _ry, _rz)
