from typing import Dict, Any
import yaml

import numpy as np
import cv2

from python_ex.system import Path

from .basement import __Basement__


class Realsense_Dataset(__Basement__):
    def __init__(
        self, root: str, category: str, trans_config: Dict[str, Any], **kwarg
    ) -> None:
        super().__init__(root, category, trans_config, **kwarg)

    def Make_datalist(self, root: str, category: str, **kwarg):
        # Get camera info
        with open(Path.Join([category, "realsense.yaml"], root)) as f:
            self.camera_info = yaml.load(f, Loader=yaml.FullLoader)

        # Get input and target file list
        _input_files = Path.Search(
            Path.Join([category, "rgb"], root), Path.Type.FILE, "*", "jpg")

        _depth_files = Path.Search(
            Path.Join([category, "depth"], root), Path.Type.FILE, "*", "png")
        _pose_files = Path.Search(
            Path.Join([category, "poses"], root), Path.Type.FILE, "*", "npy")
        _targets = [(
            _depth_file, _pose_file
        ) for _depth_file, _pose_file in zip(_depth_files, _pose_files)]

        return _input_files, _targets

    def Make_transform(self, data_transform_config: Dict[str, Any]):
        ...

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # rgb
        _input_file: str = self.inputs[index]
        _input_img: np.ndarray = cv2.imread(_input_file, -1)
        _input_img = cv2.cvtColor(_input_img, cv2.COLOR_BGR2RGB)
        _input_img = _input_img / 255.0

        # depth and pose (target)
        _depth_file, _pose_file = self.targets[index]
        _depth_scale = self.camera_info["camera_params"]["png_depth_scale"]
        _depth_img: np.ndarray = cv2.imread(_depth_file, -1)  # uint16
        _depth_img = _depth_img / _depth_scale

        _pose_data: np.ndarray = np.load(_pose_file)

        _file_name: str = _input_file.split("/")[-1].split(".")[0]

        return _input_img, _depth_img, _pose_data, _file_name
