from typing import Dict, List, Any, Optional

from torch import Tensor, tensor
from torch.utils.data import Dataset

from python_ex._numpy import ndarray

import albumentations as A
from albumentations.pytorch import ToTensorV2


if __package__ == "":
    # if this file in local project
    from torch_ex._label import File_Profile, Label
else:
    # if this file in package folder
    from ._label import File_Profile, Label


# -- DEFINE CONSTNAT -- #


# -- Mation Function -- #

class Augment():
    class Basement():
        def __init__(self, **transform_constructer) -> None:
            self._transform = self._Make_transform(**transform_constructer)

        def _Make_transform(self, **transform_constructer):
            raise NotImplementedError

        def __call__(self, *args: Any, **kwds: Any) -> Any:
            raise NotImplementedError

    class Albumentations(Basement):
        def _Make_transform(
            self,
            transform_list: List[A.BasicTransform],
            bbox_parameter: Optional[Dict[str, str]] = None,
            keypoints_parameter: Optional[Dict[str, str]] = None,
            group_parmaeter: Optional[Dict[str, str]] = None
        ):
            if not isinstance(transform_list[-1], ToTensorV2):
                transform_list.append(ToTensorV2())  # don't forgot this

            return A.Compose(transform_list, bbox_parameter, keypoints_parameter, additional_targets=group_parmaeter)

        def __call__(self, data: Dict[str, ndarray]) -> Dict[str, Tensor]:
            return self._transform(**data)


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
