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

from ._System import Path, File
from ._Array import Array_IO


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

    def Set_file_list_from(self, data_dir: str, ext: Union[str, List[str]] = [".jpg", ".png"], data_key: Optional[str] = None):
        _data_key = "img" if data_key is None else data_key
        if _data_key in self.data_profile.keys():
            self.data_profile[_data_key] += Path.Search(Path.Join(_data_key, data_dir), Path.Type.FILE, ext_filter=ext)
        else:
            self.data_profile[_data_key] = Path.Search(Path.Join(_data_key, data_dir), Path.Type.FILE, ext_filter=ext)

    def Get_frame(self, frame_num: int):
        raise NotImplementedError


@dataclass
class Camera():
    cam_id: int
    frame: int = 0

    image_size: List[int] = field(default_factory=list)
    intrinsic: ndarray = field(default_factory=lambda: np.eye(4))

    rectification: ndarray = field(default_factory=lambda: np.eye(4))  # check it in later

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

    def world_to_img(self, transform_to_cam: ndarray, points: ndarray, limit_z: Tuple[int, Optional[int]] = (0, None)):
        """

        """
        _extrict = self.rectification @ transform_to_cam
        _points_on_cam = np.matmul(_extrict, points)
        _projection: ndarray = np.matmul(self.intrinsic, _points_on_cam)

        _z_min_limit, _z_max_limit = limit_z

        _points_in_front = _projection[:, _projection[2] >= _z_min_limit]
        _points_in_front = _points_in_front if _z_max_limit is None else _points_in_front[:, _points_in_front[2] < _z_max_limit]

        _depth = _points_in_front[2]
        if _z_min_limit == 0:
            _depth[_depth == 0] = -1e-6

        _u = np.round(_points_in_front[0, :] / _depth).astype(int)
        _v = np.round(_points_in_front[1, :] / _depth).astype(int)

        # filtering the point of that over the image size
        _filter = (_v >= 0) * (_v < self.image_size[1]) * (_u >= 0) * (_u < self.image_size[0])

        return _u[_filter], _v[_filter], _depth[_filter]


class Image_Process():
    ...


class Vision_IO(File.Basement):
    class Image():
        def __init__(self, save_dir: str) -> None:
            Vision_IO._Path_check("", save_dir)  # If pass this code, save_dir is exist
            self.save_dir = save_dir

        def _Make_image_list(self):
            _image_file_list = Path.Search(self.save_dir, Path.Type.FILE)
            self.file_stream_list = [open(_file, "rb") for _file in _image_file_list]
            self.image_file_list = _image_file_list

        def _Read_data_in_list(self, id: int, data_format: Format_of.Image):
            _file_stream = self.file_stream_list[id]
            _encoded_img = Array_IO._Read_from_binary(_file_stream, np.uint8)
            return cv2.imdecode(_encoded_img, data_format.value)

    class Video():
        class Capture():
            def __init__(self, source_name: Union[int, str], save_dir: str = ""):
                # get capute source
                if isinstance(source_name, int):
                    _source = source_name
                else:
                    _, _source = Vision_IO._Path_check(source_name, save_dir, raise_error=True)

                _cap = cv2.VideoCapture(_source)

                if not _cap.isOpened():
                    raise RuntimeError("Camera open failed!")

                self.capture = _cap

            def _Get_video_info(self):
                _w = round(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                _h = round(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                _fps = self.capture.get(cv2.CAP_PROP_FPS)

                return _w, _h, _fps

            def __call__(self):
                _ret, _frame = self.capture.read()

                if not _ret:
                    self.capture.release()
                    return False, None
                else:
                    return True, _frame

        class Writer():
            def __init__(self, img_size: Tuple[int, int], file_name: str, save_dir: str, video_codex: Format_of.Codex, frame: int):
                _img_h, _img_w = img_size

                # make the vidoe codex and string of ext
                _ext = video_codex.name.lower()
                _fourcc = cv2.VideoWriter.fourcc(*video_codex.value)

                _, _file_path = Vision_IO._Path_check(file_name, save_dir, _ext)
                self._writer = cv2.VideoWriter(_file_path, _fourcc, frame, (_img_w, _img_h))

            def __call__(self, frame: ndarray):
                self._writer.write(frame)

            def _Close(self):
                self._writer.release()


class Vision_Toolbox():
    @staticmethod
    def _format_converter(image: ndarray):
        return ...

    @staticmethod
    def _channel_converter(image: ndarray, convert_to_last_ch: bool = False):
        _shape = image.shape

        if len(_shape) == 2:
            # [h, w] -> [h, w, 1] or [1, h, w]
            return image[:, :, np.newaxis] if convert_to_last_ch else image[np.newaxis, :, :]

        else:
            # [h, w, c] -> [c, h, w] or [c, h, w] -> [h, w, c]
            return np.moveaxis(image, 0, -1) if convert_to_last_ch else np.moveaxis(image, -1, 0)
