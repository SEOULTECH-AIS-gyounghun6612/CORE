from ._base import OpenGL_Base_Pass
from .depth import Depth_Pass
from .normal import Normal_Pass
from .rgb import RGB_Pass
from .segmentation import Segmentation_Pass

PASS_TYPES = {
    "rgb": RGB_Pass,
    "depth": Depth_Pass,
    "normal": Normal_Pass,
    "segmentation": Segmentation_Pass,
}

__all__ = [
    "OpenGL_Base_Pass",
    "Depth_Pass",
    "Normal_Pass",
    "RGB_Pass",
    "Segmentation_Pass",
    "PASS_TYPES",
]
