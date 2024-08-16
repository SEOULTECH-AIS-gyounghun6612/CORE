from typing import Type

from numpy.typing import NDArray
import numpy as np
import cv2

from python_ex.system import Path, File

from . import Parser, PARSER, Dataset_Basement

try:
    import pyrealsense2 as rs
except Exception:
    pass


def _Read_pose_file(file: str) -> NDArray:
    _raw_data: np.ndarray = np.load(file)
    _h, _w = _raw_data.shape[:2]

    _sp_h = (_h - 4) if (_h - 4) else None
    _sp_w = (_w - 4) if (_w - 4) else None

    _pose = np.eye(4, 4)
    _pose[:_sp_h, :_sp_w] = _raw_data

    return _pose


class Realsense_Parser(Parser):
    def Get_data_from(self, data_dir: str, is_live: bool = False, **kwarg):
        if is_live:
            ...
        else:
            self.data_block = {
                "rgb": Path.Search(
                    Path.Join("rgb", data_dir),
                    Path.Type.FILE,
                    ext_filter="jpg"
                ),
                "depth": Path.Search(
                    Path.Join("depth", data_dir),
                    Path.Type.FILE,
                    ext_filter="png"
                ),
                "pose": [
                    _Read_pose_file(_file) for _file in Path.Search(
                        Path.Join("poses", data_dir),
                        Path.Type.FILE,
                        ext_filter="npy"
                    )
                ]
            }

            self.data_info = {
                "cam_info": File.YAML.Read("realsense.yaml", data_dir)
            }


class Realsense(Dataset_Basement):
    def __init__(
        self,
        data_dir: str,
        is_live: bool = False,
        use_depth: bool = True,
        data_parser: Type[PARSER] = Realsense_Parser
    ) -> None:
        if is_live:
            raise NotImplementedError  # live dataset is not suport yet
        super().__init__(data_dir, data_parser, is_live=is_live)
        self.is_live = is_live
        self.use_depth = use_depth

        # in later, make disorted matrix from self.data_info

    def __len__(self):
        return len(self.data_block["rgb"])

    def __getitem__(self, index):
        _rgb_file = self.data_block["rgb"][index]
        _rgb = cv2.imread(_rgb_file, cv2.IMREAD_UNCHANGED)
        # in later add apply disorted
        # ex) cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        _depth = None
        if self.use_depth:
            _depth_file = self.data_block["depth"][index]
            _depth = cv2.imread(_depth_file, cv2.IMREAD_UNCHANGED)

        _pose = self.data_block["pose"][index]

        return _rgb, _depth, _pose
