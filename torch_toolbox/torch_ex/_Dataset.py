from __future__ import annotations

from typing import Tuple, Dict, List, Any, Type
from dataclasses import dataclass, field, asdict
import operator
import itertools

from enum import Enum
from math import pi, cos, sin, ceil
from random import randrange, random

from numpy import ndarray, array

from torch import Tensor, tensor, zeros, min, all
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision.transforms import functional as F

from python_ex._System import File
from python_ex._Project import Config


class Utils():
    class Masked_Image_Tensor():
        def __init__(self, images: List[Tensor]):
            _shape_array: ndarray = array([image.shape for image in images])
            _b_size = _shape_array.shape[0]
            _max_shape: ndarray = _shape_array.max(axis=0)

            # make image tensor
            _holder = zeros((_b_size,) + tuple(_max_shape))
            _mask = zeros((_b_size,) + tuple(_max_shape)[:-1])

            for _ct, _img in enumerate(images):
                _h, _w, _ = _shape_array[_ct]
                _holder[_ct, : _h, : _w] = _img
                _mask[_ct, : _h, : _w] = False

            self.t_image = _holder
            self.t_mask = _mask[:, :, :, None]
            self.t_shape = tensor(_shape_array)[:, :2, None]

        def to(self, device):
            self.t_image.to(device)
            self.t_mask.to(device)
            self.t_shape.to(device)

        def cuda(self, gpu_id):
            self.t_image.cuda(gpu_id)
            self.t_mask.cuda(gpu_id)
            self.t_shape.cuda(gpu_id)
            return self

        def decompose(self):
            return self.t_image, self.t_mask, self.t_shape

        def __repr__(self):
            return str(self.t_image)


class Label():
    '''
    라벨과 관련된 일련의 처리 과정을 위한 모듈
    -------------------------------------------------------------------------------------------
    '''
    class Style(Enum):
        CLASSIFICATION = "classification"
        SEMENTIC = "sementic"
        INSTANCES = "instances"
        KEY_POINTS = "key_Points"

    class Structure():
        @dataclass(order=True)
        class Label_Base():
            '''
            라벨 데이터 자료 기본 구조

            ----------
            ### Parameters - Attributes
            identity_num :
                객체의 고유번호
            train_num :
                학습 과정 중 객체에 할당된 호출번호
            cateogry :
                객체의 서술 범주
            ignore_in_eval :
                평가 인자 계산 과정 중 포함 여부
            name :
                객체 이름
            value :
                객체 구분 정보

            ----------
            '''
            identity_num: int
            train_num: int
            cateogry: str
            ignore_in_eval: bool
            name: str
            value_of_label: Any

        @dataclass(order=True)
        class Classification(Label_Base):
            value_of_label: str

        @dataclass(order=True)
        class Mask(Label_Base):
            value_of_label: List[int]

    @staticmethod
    def _Read_label_data(label_file: str, label_dir: str, label_structure: Type[Label.Structure.Label_Base]) -> List[Label.Structure.Label_Base]:
        _label_type: str = label_structure.__name__.lower()
        return [label_structure(**_data) for _data in File.Json._Read(label_file, label_dir)[_label_type]]

    @staticmethod
    def _Write_label_data(label_file: str, label_dir: str, label_data: List[Label.Structure.Label_Base]):
        _label_type: str = label_data[0].__class__.__name__.lower()

        File.Json._Write(label_file, label_dir, {_label_type: [asdict(_data) for _data in label_data]})


class Augment():
    class Supported(Enum):
        ALBUIMIENTATIONS = "Albumentations"
        TORCHVISION = "Torchvision"

    class Process():
        class Basement():
            def __init__(
                self,
                output_size: List[int],
                amplification: int,
                rotate_limit: int = 0,
                hflip_rate: float = 0.0,
                vflip_rate: float = 0.0,
                is_norm: bool = True,
                norm_mean: List[float] = [0.485, 0.456, 0.406],
                norm_std: List[float] = [0.229, 0.224, 0.225],
                apply_to_tensor: bool = True,
                group_parmaeter: Dict | None = None,
                **augment_constructer
            ) -> None:

                self.amp: int = amplification
                _componant_list = self._Make_componant_list(output_size, rotate_limit, hflip_rate, vflip_rate, is_norm, norm_mean, norm_std, apply_to_tensor)
                self._transform = self._Build_transform(_componant_list, group_parmaeter, **augment_constructer)

            def _Make_componant_list(
                self,
                output_size: List[int],
                rotate_limit: int = 0,
                hflip_rate: float = 0.0,
                vflip_rate: float = 0.0,
                is_norm: bool = True,
                norm_mean: List[float] = [0.485, 0.456, 0.406],
                norm_std: List[float] = [0.229, 0.224, 0.225],
                apply_to_tensor: bool = False
            ) -> List:
                raise NotImplementedError

            def _Get_padding_size(self, output_size: List[int], rotate_limit: int = 0) -> List[int]:
                _rad = pi * rotate_limit / 180
                _h = ceil(output_size[1] * sin(_rad) + output_size[0] * cos(_rad))
                _w = ceil(output_size[0] * sin(_rad) + output_size[1] * cos(_rad))

                return [_h, _w]

            def _To_tensor(self):
                raise NotImplementedError

            def _Resize(self, size: List[int]):
                raise NotImplementedError

            def _Random_Crop(self, size: List[int]):
                raise NotImplementedError

            def _Rotate_within(self, limit_angle: int | List[int]):
                raise NotImplementedError

            def _Random_Flip(self, horizontal_rate: float, vertical_rate: float) -> List:
                raise NotImplementedError

            def _Normalization(self, mean: List[float], std: List[float]):
                raise NotImplementedError

            def _Build_transform(self, transform_list: List, group_parmaeter: Dict | None, **augment_constructer):
                raise NotImplementedError

            def __call__(self, data: Dict[str, Any]) -> Dict[str, Tensor]:
                return self._transform(**data)

        class Torchvision(Basement):
            class Tr_Comp():
                def __call__(self, image: Tensor | List[Tensor], target: Dict[str, Any] | None) -> Tuple[Tensor | List[Tensor], Dict[str, Any] | None]:
                    raise NotImplementedError

            class Random_Crop(Tr_Comp):
                def __init__(self, size: Tuple[int, int]) -> None:
                    self._section_x, self._section_y = size

                def __call__(self, image: Tensor | List[Tensor], target: Dict[str, Tensor] | None) -> Any:
                    _section_y = self._section_y
                    _section_x = self._section_x
                    _c, _h, _w = image[0].shape if isinstance(image, list) else image.shape

                    _lt_y = randrange(0, max(1, _h - _section_y))
                    _lt_x = randrange(0, max(1, _w - _section_x))

                    # imput image
                    if isinstance(image, list):
                        for _ct, _img in enumerate(image):
                            if _h < _section_y or _w < _section_x:
                                _new_h = _section_y if _h < _section_y else _h
                                _new_w = _section_x if _w < _section_x else _w
                                _new_img = zeros((_c, _new_h, _new_w), dtype=_img.dtype)
                                _new_img[:, 0: _h, 0: _w] = _img
                                _img = _new_img

                            image[_ct] = _img[:, _lt_y: _lt_y + _section_y, _lt_x: _lt_x + _section_x]

                    else:
                        if _h < _section_y or _w < _section_x:
                            _new_h = _section_y if _h < _section_y else _h
                            _new_w = _section_x if _w < _section_x else _w
                            _new_img = zeros((_c, _new_h, _new_w), dtype=image.dtype)
                            _new_img[:, 0: _h, 0: _w] = image
                            image = _new_img

                        image = image[:, _lt_y: _lt_y + _section_y, _lt_x: _lt_x + _section_x]

                    # target
                    if isinstance(target, dict):
                        for _key, items in target.items():
                            if _key.find("mask") != -1:
                                # items: ndarray
                                if _h < _section_y or _w < _section_x:
                                    _new_h = _section_y if _h < _section_y else _h
                                    _new_w = _section_x if _w < _section_x else _w

                                    _new_img = zeros((_new_h, _new_w, _c), dtype=items.dtype)
                                    _new_img[:, 0: _h, 0: _w] = items
                                    items = _new_img

                                target[_key] = items[_lt_y: _lt_y + _section_y, _lt_x: _lt_x + _section_x]

                            elif _key.find("bbox") != -1:
                                _bbox: Tensor = min(items["bbox"].reshape([-1, 2, 2]), tensor([_section_x, _section_y])).clamp(0)
                                _keep = all(_bbox[:, 1, :] > _bbox[:, 0, :], dim=1)

                                target[_key] = {
                                    "class_id": items["class_id"][_keep],
                                    "bbox": _bbox[_keep].reshape(-1, 4)
                                }

                    return image, target

            class Random_Flip(Tr_Comp):
                def __init__(self, horizontal_rate: float, vertical_rate: float) -> None:
                    self._horiz_p = horizontal_rate
                    self._verti_p = vertical_rate

                def __call__(self, image: Tensor | List[Tensor], target: Dict[str, Tensor] | None):
                    _horiz = random() >= self._horiz_p
                    _verti = random() >= self._verti_p

                    # imput image
                    if isinstance(image, list):
                        _c, _h, _w = image[0].shape

                        for _ct, _img in enumerate(image):
                            if _horiz : _img = F.hflip(_img)
                            if _verti : _img = F.vflip(_img)

                            image[_ct] = _img

                    else:
                        _c, _h, _w = image.shape
                        if _horiz : image = F.hflip(image)
                        if _verti : image = F.vflip(image)

                    # target
                    if isinstance(target, dict):
                        for _key, items in target.items():
                            if _key.find("mask") != -1:
                                if _horiz : items = F.hflip(items)
                                if _verti : items = F.vflip(items)
                                target[_key] = items
                            elif _key.find("bbox") != -1:
                                if _horiz:  # <->
                                    _bbox = (tensor([_w, 0, _w, 0]) + items["bbox"] * tensor([-1, 1, -1, 1]))[:, [2, 1, 0, 3]]
                                if _verti:
                                    _bbox = (tensor([0, _h, 0, _h]) + items["bbox"] * tensor([1, -1, 1, -1]))[:, [0, 3, 2, 1]]

                                items = {
                                    "class_id": items["class_id"],
                                    "bbox": _bbox
                                }

                    return image, target

            class Normalize(Tr_Comp):
                def __init__(self, mean: List[float], std: List[float]) -> None:
                    self._mean = mean
                    self._std = std

                def __call__(self, image: Tensor | List[Tensor], target: Any):
                    image = [F.normalize(_img, mean=self._mean, std=self._std) for _img in image] if isinstance(image, list) else F.normalize(image, mean=self._mean, std=self._std)
                    return image, target

            class Compose(Tr_Comp):
                class Data_Converter():
                    def __call__(self, image: ndarray | List[ndarray], target: Dict[str, Any] | None):
                        # imput image
                        if isinstance(image, list):
                            _tensor_img = [F.to_tensor(_img) for _img in image]
                        else:
                            _tensor_img = F.to_tensor(image)

                        # target
                        if isinstance(target, dict):
                            for _key, items in target.items():
                                if _key.find("mask") != -1:
                                    target[_key] = [F.to_tensor(_mask) for _mask in items]
                                elif _key.find("bbox") != -1:
                                    target[_key] = {
                                    "class_id": tensor(items["class_id"]),
                                    "bbox": tensor(items["bbox"])
                                }

                        return _tensor_img, target

                def __init__(self, transforms: List[Augment.Process.Torchvision.Tr_Comp]):
                    self._data_converter = self.Data_Converter()
                    self._transforms = transforms

                def __call__(self, image: ndarray | List[ndarray], target: Dict[str, Any] | None = None, **info):
                    _image, _target = self._data_converter(image, target)

                    for _t in self._transforms:
                        _image, _target = _t(_image, _target)

                    info.update(
                        {"image": _image} if _target is None else {"image": _image, "target": _target}
                    )

                    return info

                def __repr__(self):
                    _format_string = self.__class__.__name__ + "("
                    for _t in self._transforms:
                        _format_string += "\n"
                        _format_string += "    {0}".format(_t)
                    _format_string += "\n)"
                    return _format_string

            # def _Resize(self, size: List[int]):
            #     return T.Resize(size)

            def _Random_Crop(self, size: Tuple[int, int]):
                return self.Random_Crop(size)

            # def _Rotate_within(self, limit_angle: int):
            #     return T.RandomRotation(limit_angle)

            def _Random_Flip(self, horizontal_rate: float, vertical_rate: float):
                return self.Random_Flip(horizontal_rate, vertical_rate)

            def _Normalization(self, mean: List[float], std: List[float]):
                return self.Normalize(mean, std)

            def _Make_componant_list(
                self,
                output_size: List[int],
                rotate_limit: int = 0,
                hflip_rate: float = 0,
                vflip_rate: float = 0,
                is_norm: bool = True,
                norm_mean: List[float] = [0.485, 0.456, 0.406],
                norm_std: List[float] = [0.229, 0.224, 0.225],
                apply_to_tensor: bool = True
            ) -> List:
                _transform_list = []

                if output_size[0] != -1 and output_size[1] != -1:
                    _transform_list.append(self._Random_Crop((output_size[0], output_size[1])))

                # about flip
                if hflip_rate or vflip_rate:
                    _transform_list.append(self._Random_Flip(hflip_rate, vflip_rate))

                # apply norm
                if is_norm:
                    _transform_list.append(self._Normalization(norm_mean, norm_std))

                return _transform_list

            def _Build_transform(
                self,
                transform_list,
                group_parmaeter,
                # keypoints_parameter: Dict[str, str] | None = None
            ):
                return self.Compose(transform_list)

        class Albumentations(Basement):
            def _Make_componant_list(
                self,
                output_size: List[int],
                rotate_limit: int = 0,
                hflip_rate: float = 0,
                vflip_rate: float = 0,
                is_norm: bool = True,
                norm_mean: List[float] = [0.485, 0.456, 0.406],
                norm_std: List[float] = [0.229, 0.224, 0.225],
                apply_to_tensor: bool = False
            ) -> List:
                _transform_list = []

                # about rotate
                if (rotate_limit if isinstance(rotate_limit, int) else any(rotate_limit)):  # use rotate
                    _transform_list.append(self._Resize(self._Get_padding_size(output_size, rotate_limit)))
                    _transform_list.append(self._Rotate_within(rotate_limit))
                    _transform_list.append(self._Random_Crop(output_size))
                else:
                    _transform_list.append(self._Resize(output_size))

                # about flip
                if hflip_rate or vflip_rate:
                    _transform_list += self._Random_Flip(hflip_rate, vflip_rate)

                # apply norm
                if is_norm:
                    _transform_list.append(self._Normalization(norm_mean, norm_std))

                _transform_list.append(self._To_tensor()) if apply_to_tensor else ...

                return _transform_list

            def _To_tensor(self):
                return ToTensorV2()

            def _Resize(self, size: List[int]):
                return A.Resize(height=size[0], width=size[1])

            def _Random_Crop(self, size: List[int]):
                return A.RandomCrop(height=size[0], width=size[1])

            def _Rotate_within(self, limit_angle: int):
                return A.Rotate(limit_angle)

            def _Random_Flip(self, horizontal_rate: float, vertical_rate: float):
                _list = []
                _list.append(A.HorizontalFlip(p=horizontal_rate)) if horizontal_rate else ...
                _list.append(A.VerticalFlip(p=vertical_rate)) if vertical_rate else ...
                return _list

            def _Normalization(self, mean: List[float], std: List[float]):
                return A.Normalize(mean, std)

            def _Build_transform(
                self,
                transform_list,
                group_parmaeter: Dict[str, str] | None = None,
                bbox_parameter: Dict[str, str] | None = None,
                keypoints_parameter: Dict[str, str] | None = None
            ):
                return A.Compose(transform_list, bbox_parameter, keypoints_parameter, additional_targets=group_parmaeter)

        @staticmethod
        def _Build(
            apply_method: Augment.Supported,
            output_size: List[int],
            rotate_limit: int | List[int] = 0,
            hflip_rate: float = 0.0,
            vflip_rate: float = 0.0,
            is_norm: bool = True,
            norm_mean: List[float] = [0.485, 0.456, 0.406],
            norm_std: List[float] = [0.229, 0.224, 0.225],
            **augment_constructer
        ) -> Basement:
            return Augment.Process.__dict__[apply_method.value](output_size, rotate_limit, hflip_rate, vflip_rate, is_norm, norm_mean, norm_std, **augment_constructer)


class Data():
    class Basement(Dataset):
        def __init__(self, root_dir: str, mode: str, aug: Augment.Process.Basement, **data_initialize_arg):
            # Set dataset parameter
            self.input_list, self.target_list, self.label_list = self._Initialize_data_full(root_dir, mode, **data_initialize_arg)
            self.augment = aug

        def __len__(self):
            return len(self.input_list) * self.augment.amp

        def __getitem__(self, index) -> Dict[str, Tensor]:
            return self._Convert_to_tensor(index // self.augment.amp)

        def _Initialize_data_full(self, root_dir: str, mode: str, **data_initialize_arg) -> Tuple[List, List, List]:
            raise NotImplementedError

        def _Convert_to_tensor(self, source_index: int) -> Dict[str, Tensor]:
            raise NotImplementedError

    # class COCO(Basement):
    #     def _Initialize_data_full(self, root_dir: str, mode: str, annotaion: List[str]) -> Tuple[List, List, List]:
    #         # set data dir
    #         _data_root = Path._Join("COCO", root_dir)
    #         _img_dir = Path._Join(f"{mode}2017", _data_root)
    #         _annotation_dir = Path._Join("annotations", _data_root)

    #         # input img list
    #         _img_list = Path._Search(_img_dir, Path.Type.FILE, ext_filter="jpg")

    #         # make datalist for test
    #         if mode == "test":
    #             return _img_list, [], []

    #         # make datalist for train or validation
    #         # --- make annotation list --- #
    #         _label_data = Label._Read_label_data(Path._Join(["data_file", "COCO.json"]), Path._Devide(__file__)[0], Label.Structure.Classification)

    #         # When write this code, I just used instance data. So i make code that just use instance annotation.
    #         _annotation_data: Dict[str, List[Dict[str, Any]]] = {}   # {img_id: List[ instance -> {key_word: value} ]}
    #         _meta_data = File.Json._Read(f"instnaces_{mode}2017", _annotation_dir)

    #         for _data in _meta_data["annotations"]:
    #             _img_id = _data["image_id"]
    #             _img_file = Path._Join(f"{_img_id}.jpg", _img_dir)

    #             if _img_file not in _img_list:
    #                 continue  # image that matched this annotation, not exist in image list

    #             if _data["iscrowd"]:
    #                 #
    #                 ...
    #             else:
    #                 _bbox_info = _data["bbox"]
    #                 _bbox = (_bbox_info[0], _bbox_info[1], _bbox_info[0] + _bbox_info[2], _bbox_info[1] + _bbox_info[3])
    #                 _seg = _data["segmentation"],
    #                 _class_id = _data["category_id"]

    #                 # updata annotation data
    #                 if _img_id not in _annotation_data.keys():  # The first time, that this image has label data in this process
    #                     _annotation_data.update({
    #                         _img_id: [{"seg": _seg, "bbox": _bbox, "class_id": _class_id}]
    #                     })
    #                 else:
    #                     _annotation_data[_img_id].append({"seg": _seg, "bbox": _bbox, "class_id": _class_id})

    #         # --- make input list and target list --- #
    #         _input_list = []
    #         _target_list = []

    #         for _img_id, _targets in _annotation_data.items():
    #             if len(_targets) == len(data_info):
    #                 _input_list.append(f"{_img_dir}{_img_id}.jpg")
    #                 for _ct, (_, _data_list) in enumerate(_targets.items()):
    #                     _target_list[_ct].append(_data_list)

    #         return _input_list, _target_list, ""


# @dataclass
# class Data_Config(Config):
#     # dataset
#     dataset_dir: str = "./data"
#     dataset_name: str = "dataset_name"
#     target_info: List[Tuple[str, str]] = field(default_factory=lambda: ([("sementic", "image")]))

#     # augmentation option
#     augmentation_method: str = "Torchvision"
#     amplification: int = 1

#     output_size: List[int] = field(default_factory=lambda: ([256, 256]))
#     rotate_limit: int = 0
#     hflip_rate: float = 0.0
#     vflip_rate: float = 0.0
#     is_norm: bool = True
#     norm_mean: List[float] = field(default_factory=lambda: ([0.485, 0.456, 0.406]))
#     norm_std: List[float] = field(default_factory=lambda: ([0.229, 0.224, 0.225]))

#     # dataloader
#     batch_size: int = 64
#     num_workers: int = 8
#     drop_last: bool = True

#     def _Get_parameter(self, ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
#         return (
#             self.dataset_name,
#             {
#                 "dataset_name": self.dataset_name,
#                 "root_dir": self.dataset_dir,
#                 "data_info": [(Label.Style(_style), Data.File_Format(_format)) for _style, _format in self.target_info],
#                 "label_function": self._Get_label_function(),
#                 "augment_process": Augment.Process._Build(
#                     Augment.Supported(self.augmentation_method),
#                     self.output_size,
#                     self.rotate_limit,
#                     self.hflip_rate,
#                     self.vflip_rate,
#                     self.is_norm,
#                     self.norm_mean,
#                     self.norm_std
#                 ),
#                 "amplification": self.amplification
#             },
#             {
#                 "batch_size": self.batch_size,
#                 "num_workers": self.num_workers,
#                 "drop_last": self.drop_last
#             }
#         )

#     def _Get_label_function(self):
#         return Label.Process.__class__.__dict__[self.dataset_name]
