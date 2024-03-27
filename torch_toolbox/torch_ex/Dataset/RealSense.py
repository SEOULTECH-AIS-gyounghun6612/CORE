from typing import Dict, Any
import yaml

import numpy as np
import cv2

from python_ex._System import Path

from .Basement import __Basement__


class Realsense_Dataset(__Basement__):
    def __init__(self, data_root: str, data_category: str, data_transform_config: Dict[str, Any], **kwarg) -> None:
        super().__init__(data_root, data_category, data_transform_config, **kwarg)

    def Make_data_list(self, data_root: str, data_category: str, **kwarg):
        # Get camera info
        with open(Path.Join([data_category, "realsense.yaml"], data_root)) as f:
            self.camera_info = yaml.load(f, Loader=yaml.FullLoader)

        # Get input and target file list
        _input_files = Path.Search(Path.Join([data_category, "rgb"], data_root), Path.Type.FILE, ext_filter="jpg")
        _depth_files = Path.Search(Path.Join([data_category, "depth"], data_root), Path.Type.FILE, ext_filter="png")
        _pose_files = Path.Search(Path.Join([data_category, "poses"], data_root), Path.Type.FILE, ext_filter="npy")
        return _input_files, [(_depth_file, _pose_file) for _depth_file, _pose_file in zip(_depth_files, _pose_files)]

    def Make_data_transform(self, data_transform_config: Dict[str, Any]):
        ...

    def __len__(self):
        return len(self.input_datas)

    def __getitem__(self, index: int):
        # rgb
        _input_file: str = self.input_datas[index]
        _input_img: np.ndarray = cv2.imread(_input_file, -1)
        _input_img = cv2.cvtColor(_input_img, cv2.COLOR_BGR2RGB)
        _input_img = _input_img / 255.0

        # depth and pose (target)
        _depth_file, _pose_file = self.target_datas[index]
        _depth_img: np.ndarray = cv2.imread(_depth_file, -1)  # uint16
        _depth_img = _depth_img / self.camera_info["camera_params"]["png_depth_scale"]

        _pose_data: np.ndarray = np.load(_pose_file)

        _file_name: str = _input_file.split("/")[-1].split(".")[0]

        return _input_img, _depth_img, _pose_data, _file_name
