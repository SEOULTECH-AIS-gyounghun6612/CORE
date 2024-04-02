"""
File object
=====
    When write down python program, be used cv2 custom function.

Requirement
=====
    cv2     (pip install python-opencv)
    numpy
"""

# Import module
from __future__ import annotations
from enum import Enum
from typing import List, Tuple, Optional, Union, Dict
from dataclasses import dataclass, field
import cv2
import numpy as np
from numpy import ndarray

from .system import Path  # , File


class Format_of():
    class Image(Enum):
        BGR = cv2.IMREAD_COLOR
        BGRA = cv2.IMREAD_UNCHANGED
        GRAY = cv2.IMREAD_GRAYSCALE

    class Codex(Enum):
        MP4 = "DIVX"


@dataclass
class Scene():
    data_profile: Dict[str, List] = field(default_factory=dict)
    data_root: str = Path.ABSOLUTE_HERE

    def Get_file_list_from(
        self,
        ext: Union[str, List[str]] = [".jpg", ".png"],
        data_key: Optional[str] = None
    ):
        _dir = self.data_root
        _key = "img" if data_key is None else data_key
        _list = Path.Search(
            Path.Join(_key, _dir), Path.Type.FILE, ext_filter=ext)

        if _key in self.data_profile.keys():
            self.data_profile[_key] += _list
        else:
            self.data_profile[_key] = _list

    def Get_frame(self, frame_num: int):
        raise NotImplementedError


@dataclass
class Camera():
    cam_id: int
    frame: int = 0

    image_size: List[int] = field(default_factory=list)
    intrinsic: ndarray = field(default_factory=lambda: np.eye(4))

    # check it in later
    rectification: ndarray = field(default_factory=lambda: np.eye(4))

    scene_data: Scene = field(default_factory=Scene)

    def Set_save_root(self, save_root: str):
        # self.save_root
        ...

    def Get_camera_info(self, **kwarg):
        raise NotImplementedError

    def Get_image(self) -> np.ndarray:
        raise NotImplementedError

    def Get_depth(self) -> np.ndarray:
        raise NotImplementedError

    def Get_pose(self) -> np.ndarray:
        raise NotImplementedError

    def Capture(self, save_root: str | None):
        raise NotImplementedError

    def Convert_pixel_to_camera_point(self):
        ...

    def Convert_camera_point_to_pixel(self):
        ...

    def world_to_img(
        self,
        transform_to_cam: ndarray,
        points: ndarray,
        limit_z: Tuple[Optional[int], Optional[int]] = (0, None)
    ):
        """

        """
        _ex = self.rectification @ transform_to_cam
        _point_c = np.matmul(_ex, points)
        _projection: ndarray = np.matmul(self.intrinsic, _point_c)

        _z_min, _z_max = limit_z
        _z_min = 0 if limit_z[0] is None else limit_z[0]
        _z_max = np.inf if limit_z[1] is None else limit_z[1]
        _mask = (_projection[2] >= _z_min) * (_projection[2] < _z_max)
        _points_n = _projection[:, _mask]

        _depth = _points_n[2]
        if _z_min == 0:
            _depth[_depth == 0] = -1e-6

        _u = np.round(_points_n[0, :] / _depth).astype(int)
        _v = np.round(_points_n[1, :] / _depth).astype(int)

        # filtering the point of that over the image size
        _filter = (
            (_v >= 0)
            * (_v < self.image_size[1])
            * (_u >= 0)
            * (_u < self.image_size[0])
        )

        return _u[_filter], _v[_filter], _depth[_filter]


class Image_Process():
    ...


class Vision_Toolbox():
    @staticmethod
    def _format_converter(image: ndarray):
        return ...
