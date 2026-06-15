from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.export import Dim
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

from .... import CFGS
from ... import DATASETS
from ...functional.image_net import Get_transform
from ._base import Classification_Dataset, Classification_Dataset_Config


OBJECT_TYPE = "classification_image_dataset"
CONFIG_NAME = f"{OBJECT_TYPE}_Config"


@CFGS.Register_module(CONFIG_NAME)
@dataclass
class Classification_Image_Dataset_Config(Classification_Dataset_Config):
    config_type: str = CONFIG_NAME
    object_type: str = OBJECT_TYPE

    name: str = "classification"
    category: str = "classification"
    transform: str | None = "ImageNet"


@DATASETS.Register_module(OBJECT_TYPE)
class Classification_Image_Dataset(Classification_Dataset):
    """폴더 구조 기반 image classification dataset.

    data_dir/name/
    ├── id_map.yaml   {class_name: {class_id, category_id}}
    ├── class_a/image1.jpg, ...
    └── class_b/...

    초기화 흐름은 부모 Builder에 위임한다.
    이 클래스는 이미지 파일 스캔(_Scan_samples), 픽셀 읽기(__getitem__),
    ONNX 메타데이터(Info_for_onnx)만 담당한다.
    """

    layout: str = "NCHW"
    data_format: str = "RGB"
    shape_profile: dict[str, list[int]] = {
        "min": [], "opt": [1, 3, 224, 224], "max": []
    }

    samples: list[tuple[Path, int, int]]
    transform: transforms.Compose | None

    def Builder(
        self,
        data_dir: str,
        name: str,
        category: str,
        id_map_file: str = "id_map.yaml",
        transform: str | None = "ImageNet",
        extensions: list[str] | None = None,
        **kwargs,
    ):
        # 공통 초기화(id_map 로드, 샘플 스캔)를 부모에 위임
        super().Builder(
            data_dir, name, category,
            id_map_file=id_map_file,
            extensions=extensions,
            **kwargs,
        )
        self.transform = Get_transform(transform)

    def _Scan_samples(
        self, root: Path, extensions: list[str] | None = None, **kwargs
    ) -> list[tuple[Path, int, int]]:
        _exts = set(extensions or [".jpg", ".jpeg", ".png", ".bmp"])
        _samples = []
        for _class_dir in sorted(root.iterdir()):
            if not _class_dir.is_dir() or _class_dir.name not in self.id_map:
                continue
            _cid, _catid = self.id_map[_class_dir.name]
            for _p in sorted(_class_dir.iterdir()):
                if _p.suffix.lower() in _exts:
                    _samples.append((_p, _cid, _catid))
        return _samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        _path, _class_id, _category_id = self.samples[index]
        _img = read_image(str(_path), mode=ImageReadMode.RGB)
        if self.transform:
            _img = self.transform(_img)
        return {
            "image": _img,
            "class_id": torch.tensor(_class_id, dtype=torch.long),
            "category_id": torch.tensor(_category_id, dtype=torch.long),
        }

    def Info_for_onnx(self) -> tuple[
        nn.Module | None, tuple[Tensor, ...], dict[str, Any], dict[str, Any]
    ]:
        if self.transform is None:
            _layer = None
        elif isinstance(self.transform, transforms.Compose):
            _layer = nn.Sequential(*self.transform.transforms)
        else:
            _layer = nn.Sequential(self.transform)

        _input_names = ["image"]
        _output_names = ["class_id", "category_id"]
        _min = self.shape_profile.get("min", [])
        _max = self.shape_profile.get("max", [])
        _opt = self.shape_profile.get("opt", [])

        _dynamic_shapes: dict[str, Any] | None = None
        if _min and _max and len(_min) == len(_opt) == len(_max):
            _axes = {
                _i: Dim(f"{_input_names[0]}_dim{_i}", min=_lo, max=_hi)
                for _i, (_lo, _hi) in enumerate(zip(_min, _max))
                if _lo != _hi
            }
            if _axes:
                _dynamic_shapes = {_input_names[0]: _axes}

        return (
            _layer,
            (torch.zeros(1, 3, 224, 224, dtype=torch.uint8),),
            {"input_names": _input_names, "output_names": _output_names,
             "dynamic_shapes": _dynamic_shapes},
            {"data_profiles": {_input_names[0]: dict(self.shape_profile)}},
        )
