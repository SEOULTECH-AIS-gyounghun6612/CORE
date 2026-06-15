"""Camera node schemas used by scene and render modules."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field, InitVar
from typing import ClassVar, Any

import numpy as np

from python_toolbox.data_schema import Data_Schema
from ..register import NODE_REGISTRY
from ._base import Base_Node, PrimType


@dataclass
class Camera_Intrinsic(Data_Schema):
    """Stores pinhole camera intrinsics and clip planes.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        fx: Focal length on the X axis in pixels.
        fy: Focal length on the Y axis in pixels.
        cx: Principal point X coordinate in pixels.
        cy: Principal point Y coordinate in pixels.
        distortion: OpenCV-style distortion coefficients.
        near_clip: Near clipping plane distance.
        far_clip: Far clipping plane distance.
    """

    width: int = 1920
    height: int = 1080
    fx: float = 1000.0
    fy: float = 1000.0
    cx: float = 960.0
    cy: float = 540.0
    distortion: list = field(default_factory=lambda: [0.0] * 8)
    near_clip: float = 0.1
    far_clip: float = 1000.0

    @property
    def fov_x(self) -> float:
        """Returns the horizontal field of view in degrees."""
        return float(np.degrees(2.0 * np.arctan(self.width * 0.5 / self.fx)))

    @property
    def fov_y(self) -> float:
        """Returns the vertical field of view in degrees."""
        return float(np.degrees(2.0 * np.arctan(self.height * 0.5 / self.fy)))


@NODE_REGISTRY.Register_module("Camera")
@dataclass
class Camera(Base_Node):
    """Scene node representing a camera with intrinsic parameters.

    Attributes:
        prim_type: Node type identifier for registry and serialization.
        intrinsic: Camera intrinsic parameters.
        intrinsic_meta: Serialized intrinsic payload accepted during
            deserialization.
    """

    prim_type: PrimType = "Camera"
    intrinsic: Camera_Intrinsic = field(
        default_factory=Camera_Intrinsic, repr=False
    )
    intrinsic_meta: InitVar[dict[str, Any] | None] = None

    __custom_keys__: ClassVar[dict[str, str]] = {"intrinsic": "intrinsic_meta"}

    def __post_init__(
        self,
        local_rigid_meta: list | None,
        scale_meta: list | None,
        intrinsic_meta: dict[str, Any] | None,
    ):
        """Normalizes serialized metadata into runtime camera state."""
        super().__post_init__(local_rigid_meta, scale_meta)
        if intrinsic_meta is not None:
            self.intrinsic = Camera_Intrinsic(**intrinsic_meta)

    def Clone(self, label_name: str | None = None) -> "Camera":
        """Clones the node while deep-copying intrinsic parameters."""
        _new_node = Camera(
            label=self.label if label_name is None else label_name,
            prim_type=self.prim_type,
            local_rigid=self.local_rigid.copy(),
            scale=self.scale.copy(),
            unit_scale=self.unit_scale,
            source_key=self.source_key,
            visible=self.visible,
            intrinsic=deepcopy(self.intrinsic),
        )
        for _child in self.children:
            _cloned_child = _child.Clone()
            _cloned_child.Set_parent(_new_node)
            _new_node.children.append(_cloned_child)
        return _new_node

__all__ = ["Camera", "Camera_Intrinsic"]
