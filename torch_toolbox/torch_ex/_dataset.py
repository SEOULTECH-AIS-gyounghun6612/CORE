from typing import Dict, List, Any, Optional, Union
from enum import Enum

from dataclasses import dataclass, field

from torch import Tensor, tensor
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from python_ex._base import Utils


if __package__ == "":
    # if this file in local project
    from torch_ex._label import Data_Profile, Label
else:
    # if this file in package folder
    from ._label import Data_Profile, Label


# -- DEFINE CONSTNAT -- #
class Supported_Augment(Enum):
    ALBUIMIENTATIONS = "Albumentations"


# -- Mation Function -- #
class Supported_Transform():
    @dataclass
    class Base(Utils.Config):
        def _Get_augment(self, aug_type: str):
            raise NotImplementedError

    @dataclass
    class Rotate(Base):
        angle_limit: int

        def _Get_augment(self, aug_type: str):
            if aug_type == "Albumentations":
                return [A.Rotate(self.angle_limit), ]
            else:
                raise ValueError(f"{aug_type} augmentation is not support")

    @dataclass
    class Normalization(Base):
        mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
        std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

        def _Get_augment(self, aug_type: str):
            if aug_type == "Albumentations":
                return [A.Normalize(self.mean, self.std), ]
            else:
                raise ValueError(f"{aug_type} augmentation is not support")

    @dataclass
    class Random_Flip(Base):
        horizontal_rate: float
        vertical_rate: float

        def _Get_augment(self, aug_type: str):
            if aug_type == "Albumentations":
                _list = []
                _list.append(A.HorizontalFlip(p=self.horizontal_rate)) if self.horizontal_rate else ...
                _list.append(A.VerticalFlip(p=self.vertical_rate)) if self.vertical_rate else ...
                return _list
            else:
                raise ValueError(f"{aug_type} augmentation is not support")

    @dataclass
    class Resize(Base):
        size: List[int]

        def _Get_augment(self, aug_type: str):
            if aug_type == "Albumentations":
                _h, _w, = self.size[:2]
                return [A.Resize(_h, _w), ]
            else:
                raise ValueError(f"{aug_type} augmentation is not support")

    @dataclass
    class Random_Crop(Base):
        size: List[int]

        def _Get_augment(self, aug_type: str):
            if aug_type == "Albumentations":
                _h, _w, = self.size[:2]
                return [A.RandomCrop(_h, _w), ]

            else:
                raise ValueError(f"{aug_type} augmentation is not support")

    @dataclass
    class To_Tenor(Base):
        def _Get_augment(self, aug_type: str):
            if aug_type == "Albumentations":
                return [ToTensorV2(), ]
            else:
                raise ValueError(f"{aug_type} augmentation is not support")


class Augment():
    class Basement():
        def __init__(self, transform_recipe: List[Supported_Transform.Base], **augment_constructer) -> None:
            _transform_list = self._Make_transform_list(transform_recipe)
            self._transform = self._Set_augment(_transform_list, **augment_constructer)

        def _Make_transform_list(self, transform_recipe: List[Supported_Transform.Base]):
            _is_tensor = False
            _transform_list = []
            for _tr in transform_recipe:
                _is_tensor = isinstance(_tr, Supported_Transform.To_Tenor)
                _transform_list += _tr._Get_augment(self.__class__.__name__)
            if not _is_tensor:
                _transform_list += Supported_Transform.To_Tenor()._Get_augment(self.__class__.__name__)  # don't forgot this
            return _transform_list

        def _Set_augment(self, transform_list, **augment_constructer):
            raise NotImplementedError

        def __call__(self, data: Dict[str, Any]) -> Dict[str, Tensor]:
            return self._transform(**data)

    class Albumentations(Basement):
        def _Set_augment(
                self,
                transform_list,
                bbox_parameter: Optional[Dict[str, str]] = None,
                keypoints_parameter: Optional[Dict[str, str]] = None,
                group_parmaeter: Optional[Dict[str, str]] = None):
            return A.Compose(transform_list, bbox_parameter, keypoints_parameter, additional_targets=group_parmaeter)


class Custom_Dataset(Dataset):
    def __init__(self, label_process: Label.Process.Basement, data_profiles: List[Data_Profile], amplification: int, augmentation: Union[Augment.Basement, List[Augment.Basement]]):
        self._label_process = label_process
        self._data_profiles = data_profiles
        self._amplification = amplification
        self._augment = augmentation

    # Freeze function
    def __len__(self):
        return len(self._data_profiles[0]._data_list) * self._amplification

    def __getitem__(self, index) -> Dict[str, Tensor]:
        _source_index = index // self._amplification
        return self._Pre_process(self._label_process._work(self._data_profiles, _source_index))

    # Un-Freeze function
    def _Pre_process(self, _pick_data: Dict[str, Any]) -> Dict[str, Tensor]:
        if isinstance(self._augment, Augment.Basement):
            _datas = self._augment(_pick_data)
        else:
            raise ValueError("If you want use multiple augment, must make the apply funcion")

        _datas.update({"data_info": tensor(_pick_data["index"])})
        return _datas
