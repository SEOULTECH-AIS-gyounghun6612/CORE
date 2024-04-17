import yaml

import numpy as np
import cv2

from python_ex.system import Path

from .basement import __Basement__


class CustomDataset(__Basement__):
    def Make_datalist(self, root: str, mode: str | None = None, **kwarg):
        # Get camera info
        with open(
            Path.Join("realsense.yaml", root), encoding="UTF-8"
        ) as f:
            self.camera_info = yaml.load(f, Loader=yaml.FullLoader)

        # Get input and target file list
        _input_files = Path.Search(
            Path.Join("rgb", root), Path.Type.FILE, "*", "jpg")

        _depth_files = Path.Search(
            Path.Join("depth", root), Path.Type.FILE, "*", "png")
        _pose_files = Path.Search(
            Path.Join("poses", root), Path.Type.FILE, "*", "npy")
        _targets = list(zip(_depth_files, _pose_files))

        return _input_files, _targets

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
