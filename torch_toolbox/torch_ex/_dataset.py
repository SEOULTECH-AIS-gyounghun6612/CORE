from typing import Dict, List, Any, Optional
from enum import Enum

from dataclasses import dataclass, asdict, field

from torch import Tensor, tensor
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


if __package__ == "":
    # if this file in local project
    from torch_ex._label import File_Profile, Label
else:
    # if this file in package folder
    from ._label import File_Profile, Label


# -- DEFINE CONSTNAT -- #
class Supported_Augment(Enum):
    ALBUIMIENTATIONS = "Albumentations"


# -- Mation Function -- #
class Supported_Transform():
    @dataclass
    class Base():
        def _Get_augment(self, aug_type: str):
            raise NotImplementedError

        def Convert_to_dict(self):
            return asdict(self)

    @dataclass
    class Rotate(Base):
        angle_limit: int

        def _Get_augment(self, aug_type: str):
            if aug_type == "Albumentations":
                return [A.Rotate(self.angle_limit), ]
            else:
                raise ValueError

    @dataclass
    class Normalization(Base):
        mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
        std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

        def _Get_augment(self, aug_type: str):
            if aug_type == "Albumentations":
                return [A.Normalize(self.mean, self.std), ]
            else:
                raise ValueError

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
                raise ValueError

    @dataclass
    class Resize(Base):
        size: List[int]

        def _Get_augment(self, aug_type: str):
            if aug_type == "Albumentations":
                _h, _w, = self.size[:2]
                return [A.Resize(_h, _w), ]
            else:
                raise ValueError

    @dataclass
    class Random_Crop(Base):
        size: List[int]

        def _Get_augment(self, aug_type: str):
            if aug_type == "Albumentations":
                _h, _w, = self.size[:2]
                return [A.RandomCrop(_h, _w), ]

            else:
                raise ValueError

    @dataclass
    class To_Tenor(Base):
        def _Get_augment(self, aug_type: str):
            if aug_type == "Albumentations":
                return [ToTensorV2(), ]
            else:
                raise ValueError


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
    def __init__(self, label_process: Label.Process.Basement, file_profiles: List[File_Profile], amplification: int, augmentation: Augment.Basement):
        self._label_process = label_process
        self._file_profiles = file_profiles
        self._Amplification = amplification
        self._Augment = augmentation

    def __len__(self):
        return len(self._file_profiles[0]._file_list) * self._Amplification

    def __getitem__(self, index) -> Dict[str, Tensor]:
        _source_index = index // self._Amplification

        _datas: Dict[str, Tensor] = {}

        _pick_data = self._label_process._work(self._file_profiles, _source_index)
        _datas.update({"index": tensor(_pick_data["index"])})
        _datas.update(self._Augment(_pick_data))

        return _datas
