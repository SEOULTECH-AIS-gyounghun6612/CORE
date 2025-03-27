from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Generic, Annotated, Union

from numpy import ndarray, array, int_, float_
from numpy.typing import NDArray

import torch
from torch.nn import parameter

from python_ex.vision import Viewpoints, Toolbox, Flag


INTRINSIC = Annotated[torch.Tensor, tuple[3, 3]]
EXTRINSIC = Annotated[torch.Tensor, tuple[4, 4]]
ROTATE = Annotated[torch.Tensor, tuple[3, 3]]
TRANSFER = Annotated[torch.Tensor, tuple[1, 3]]

IMG_WITHOUT_CH = Annotated[torch.Tensor, tuple[int, int]]
IMG_WITH_CH = Annotated[torch.Tensor, tuple[int, int, int]]


class Viewpoints_Tenser():
    class Camera(Viewpoints.Camera):
        def __init__(
            self,
            cam_id: int,
            fps: float,
            shape: tuple[int, int, int],
            intrinsic: INTRINSIC = torch.eye(3),
            cuda: str = "cuda:0",
            is_trainable: bool = True
        ) -> None:
            super().__init__(cam_id, fps, shape)

            self._intrinsic: INTRINSIC = intrinsic if (
                is_trainable
            ) else parameter.Parameter(intrinsic)

            self.cuda = cuda
            self.is_trainable = is_trainable

        @property
        def Intrinsic(self):
            return self._intrinsic

        @Intrinsic.setter
        def Intrinsic(
            self,
            intrinsic_value: Union[
                INTRINSIC, torch.Tensor, NDArray[int_ | float_], list]
        ):
            _intrinsic = array(intrinsic_value) if (
                isinstance(intrinsic_value, list)
            ) else intrinsic_value

            _intrinsic = torch.from_numpy(_intrinsic) if (
                isinstance(_intrinsic, ndarray)
            ) else _intrinsic

            assert len(_intrinsic.shape) in (1, 2)

            if len(_intrinsic.shape) == 1:
                _intrinsic = _intrinsic.reshape(3, 3)

            _h, _w = _intrinsic.shape[:2]

            _sp_h = (_h - 3) if _h < 3 else 3
            _sp_w = (_w - 3) if _w < 3 else 3

            if "cuda" in self.cuda:
                _intrinsic = _intrinsic.cuda(self.cuda)

            if self.is_trainable:
                _intrinsic = parameter.Parameter(_intrinsic)

            self._intrinsic[:_sp_h, :_sp_w] = _intrinsic

    @dataclass
    class Img_Group():
        def Get_image(
            self
        ) -> dict[
            str,
            IMG_WITH_CH | IMG_WITHOUT_CH | torch.Tensor
        ]:
            raise NotImplementedError

        def Set_image(
            self,
            **frame_images: IMG_WITH_CH | IMG_WITHOUT_CH | torch.Tensor
        ):
            raise NotImplementedError

    IMG_GROUP = TypeVar("IMG_GROUP", bound=Img_Group)

    class Scene(Viewpoints.Scene, Generic[IMG_GROUP]):
        def __init__(
            self,
            cam_info: Viewpoints_Tenser.Camera,
            images: Viewpoints_Tenser.IMG_GROUP,
            frame_id: int = 0,
            extrinsic: EXTRINSIC = torch.eye(4),
            cuda: str = "cuda:0",
            is_trainable: bool = True
        ):
            super().__init__(cam_info, images, frame_id)

            self._extrinsic: EXTRINSIC = extrinsic if (
                is_trainable
            ) else parameter.Parameter(extrinsic)

            self.cuda = cuda
            self.is_trainable = is_trainable

        @property
        def Extrinsic(self):
            """ ### 장면의 위치 정보를 반환하는 함수"""
            return self._extrinsic

        @Extrinsic.setter
        def Extrinsic(
            self,
            extrinsic_value: Union[
                EXTRINSIC, torch.Tensor, NDArray[int_ | float_], list]
        ):
            _extrinsic = array(extrinsic_value) if (
                isinstance(extrinsic_value, list)
            ) else extrinsic_value

            _extrinsic = torch.from_numpy(_extrinsic) if (
                isinstance(_extrinsic, ndarray)
            ) else _extrinsic

            assert len(_extrinsic.shape) in (1, 2)

            if len(_extrinsic.shape) == 1:
                _extrinsic = _extrinsic.reshape(-1, 4)

            _h, _w = _extrinsic.shape[:2]

            _sp_h = (_h - 4) if _h < 4 else 4
            _sp_w = (_w - 4) if _w < 4 else 4

            if "cuda" in self.cuda:
                _extrinsic = _extrinsic.cuda(self.cuda)

            if self.is_trainable:
                _extrinsic = parameter.Parameter(_extrinsic)

            self._extrinsic[:_sp_h, :_sp_w] = _extrinsic

        def Cam_to_world(self):
            return torch.linalg.inv(self._extrinsic)

        def World_to_Cam(
            self,
            translate: torch.Tensor = torch.zeros(3),
            scale: float = 1.0
        ):
            _c2w = torch.linalg.inv(self._extrinsic)
            cam_center = _c2w[:3, 3]
            cam_center = (cam_center + translate) * scale
            _c2w[:3, 3] = cam_center
            return torch.linalg.inv(self._extrinsic)

        def Set_rotate_from_angle(
            self, is_override: bool = False, **delta: tuple[float, bool]
        ):
            _rotate_order = self._rotate_order
            _rotate_metrix: ROTATE = torch.eye(3) if (
                is_override
            ) else self._extrinsic[:3, :3]

            for _k in _rotate_order:
                if _rotate_order in delta:
                    _delta_matrix = torch.from_numpy(
                        Toolbox.Rotate_Matrix.Get_matrix(
                            delta[_k][0], Flag.Axis(_k), delta[_k][1]
                        )
                    )
                    _rotate_metrix = torch.matmul(
                        _rotate_metrix,
                        _delta_matrix
                    )
            self._extrinsic = _rotate_metrix

        def Set_rotate_from_list(
            self,
            rotate: Union[
                ROTATE, torch.Tensor, NDArray[int_ | float_], list]
        ):
            _rotate = array(rotate) if (
                isinstance(rotate, list)
            ) else rotate

            _rotate = torch.from_numpy(_rotate) if (
                isinstance(_rotate, ndarray)
            ) else _rotate

            assert len(_rotate.shape) in (1, 2)

            if len(_rotate.shape) == 1:
                _rotate = _rotate.reshape((3, 3))

            self._extrinsic[:3, :3] = _rotate.reshape((3, 3))

        def Set_transfer_from_list(
            self,
            tansfer: Union[
                ROTATE, torch.Tensor, NDArray[int_ | float_], list]
        ):
            _tansfer = array(tansfer) if (
                isinstance(tansfer, list)
            ) else tansfer

            _tansfer = torch.from_numpy(_tansfer) if (
                isinstance(_tansfer, ndarray)
            ) else _tansfer

            assert len(_tansfer.shape) in (1, 2)

            if len(_tansfer.shape) == 1:
                _tansfer = _tansfer.reshape((1, 3))

            self._extrinsic[3, :3] = _tansfer.reshape((1, 3))
