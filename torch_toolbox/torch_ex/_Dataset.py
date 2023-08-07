from __future__ import annotations

from typing import Tuple, Dict, List, Any, Generator, Callable
from enum import Enum
from math import pi, cos, sin, ceil
from dataclasses import dataclass
from random import randrange, random

from numpy import ndarray

from torch import Tensor, tensor, zeros, min, all
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import transforms as T
from torchvision.transforms import functional as F

from python_ex._Base import Directory, File

from torch_ex._Base import Process_Name


class Label():
    '''
    라벨과 관련데 일련의 처리 과정을 위한 모듈

    - ```Style``` : 처리 가능한 라벨의 스타일과 그에 따른 라벨 데이터 구조를 명시
    - ```Structure``` : 라벨 데이터 구조를 명시
    - ```Organization``` : 라벨 정보 파일을 읽고, 라벨 데이터 구조를 생성


    -------------------------------------------------------------------------------------------
    '''
    class Style(Enum):
        '''
        라벨의 스타일에 따른 라벨 데이터 구조 맵핑

        각 라벨의 스타일에 따라 매칭되는 라벨 데이터 구조는 아래와 같음
        - ```Style.CLASSIFICATION``` -> ```Label.Classification```
        - ```Style.SEMENTIC``` -> ```Label.Mask```

        -------------------------------------------------------------------------------------------
        '''

        CLASSIFICATION = "classification"
        SEMENTIC = "sementic"
        INSTANCES = "instances"
        KEY_POINTS = "key_Points"


    class Structure():
        @dataclass
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
            class_info :
                객체 구분 정보

            ----------
            '''
            identity_num: int
            train_num: int
            cateogry: str
            ignore_in_eval: bool
            name: str
            class_info: Any

        @dataclass
        class Classification(Label_Base):
            class_info: str

        @dataclass
        class Mask(Label_Base):
            class_info: List[int]

    class Organization():
        '''

        -------------------------------------------------------------------------------------------
        '''
        class Basement():
            '''
            라벨 처리 기본 구조

            ----------
            ### Parameters
            - active_style : 주어진 라벨 정보 파일을 통해 표현하려는 라벨 스타일
            - label_dir : 라벨 정보 파일의 디렉토리
            - label_file : 라벨 정보 파일 이름

            ----------
            ### Attributes
            _active_label :
                주어진 라벨 스타일을 바탕으로 구성한 라벨 데이터 구조

            ----------
            '''
            def __init__(self, label_dir: str = "", label_file: str = ""):
                # Get label data
                self._meta_data = self._Load_from_file(label_dir, label_file)

                self._active_label: Dict[Label.Style, Dict[int, List[Label.Structure.Label_Base]]] = {}

            def _Save_to_file(self, label_dir: str, label_file: str):
                raise NotImplementedError

            def _Load_from_file(self, label_dir: str = "", label_file: str = ""):
                if File._Exist_check(label_dir, label_file):
                    return File._Json(file_dir=label_dir, file_name=label_file)
                else:
                    assert self.__class__.__name__ != "Basement"

                    _default_dir = f"{Directory._Devide(__file__)[0]}data_file{ Directory._Divider}"
                    _default_file_name = f"{self.__class__.__name__}.json"

                    return File._Json(_default_dir, _default_file_name)

            # Freeze function
            def _Make_active_label(self, active_style: List[Label.Style]) -> Dict[Label.Style, Dict[int, List[Label.Structure.Label_Base]]]:
                '''


                ----------
                ### Parameters
                active_style :
                    주어진 라벨 정보 파일을 통해 표현하려는 라벨 스타일
                label_data :
                    라벨 구조를 구성하기 위한 정보 데이터

                ----------
                ### Return
                active_label :
                    주어진 라벨 스타일을 바탕으로 구성한 라벨 데이터 구조

                ----------
                '''
                _holder: Dict[Label.Style, Dict[int, List[Label.Structure.Label_Base]]] = {}

                for _style in active_style:
                    _holder.update({_style: {}})

                    if _style in [Label.Style.SEMENTIC]:
                        _key = "Mask"
                    else:
                        _key = "Classification"

                    _labels: Generator[Label.Structure.Label_Base, None, None] = (Label.Structure.__dict__[_key](**data) for data in self._meta_data[_key])

                    for _label in _labels:
                        _holder[_style].update({
                            _label.train_num: _holder[_style][_label.train_num] + [_label,] if _label.train_num in _holder[_style].keys() else [_label,]
                        })

                return _holder

            def _Pick_label(self, style: Label.Style, train_num: int | List[int]):
                if isinstance(train_num, int):
                    return self._active_label[style][train_num]

                else:
                    return [self._active_label[style][_ct] for _ct in train_num]

        class Cityscape(Basement):
            ...

        class BDD_100k(Basement):
            ...

        class COCO(Basement):
            ...


class Data():
    class Format(Enum):
        image = "image"
        colormaps = "colormaps"
        polygons = "polygons"
        annotations = "annotations"

    class Organize():
        class Basement():
            """
            ### 데이터 증폭을 위한 변형 설정

            -------------------------------------------------------------------------------------------
            ## Parameters
                root_dir (str)
                    :
                mode: (List[Process_Name])
                    :
                data_format: (List[Tuple[Label.Style | None, Target.Format]])
                    :
                label_meta_file: (str | None)
                    :
            -------------------------------------------------------------------------------------------
            """
            _active_mode: Process_Name = Process_Name.TRAIN

            def __init__(self, root_dir: str, mode_list: List[Process_Name], data_info: List[Tuple[Label.Style | None, Data.Format]], label_process: Label.Organization.Basement | None):
                # set learning info
                self._apply_mode = mode_list
                # data root dir
                self._root_dir = root_dir
                # called data info
                self._data_info = data_info

                # make label style in label info
                self._label = self._Label_update(label_process) if label_process is not None else {}

                # make data_blocks
                self._all_data = self._Make_data_block(mode_list, data_info)

                self._Set_active_mode_from(mode_list[0])

            def __len__(self):
                return self._this_data[-1].__len__()

            def _Set_active_mode_from(self, mode: Process_Name):
                if mode in self._apply_mode:
                    self._this_data = self._all_data[mode]
                else:
                    pass  # in later, warning coment here!

            def _Label_update(self, label_process: Label.Organization.Basement):
                return label_process._Make_active_label([_data[0] for _data in self._data_info if _data[0] is not None])

            def _Make_data_block(self, mode_list: List[Process_Name], data_info: List[Tuple[Label.Style | None, Data.Format]]) -> Dict[Process_Name, List[List]]:
                raise NotImplementedError

            def _Pick_up(self, num: int) -> Dict[str, Any]:
                raise NotImplementedError

        class BDD_100k(Basement):
            def __init__(self, root_dir: str, mode_list: List[Process_Name], data_info: List[Tuple[Label.Style | None, Data.Format]], label_process: Label.Organization.Basement | None):
                # data root dir
                _root_dir = Directory._Divider_check(root_dir)

                self._data_keyward = {
                    Label.Style.SEMENTIC: {
                        Data.Format.colormaps: "".join([f"{_root_dir}bdd100k/labels/sem_seg/", "{}/{}/*.jpg"])
                    }                    
                }

                super().__init__(root_dir, mode_list, data_info, label_process)

        class COCO(Basement):
            def __init__(self, root_dir: str, mode_list: List[Process_Name], data_info: List[Tuple[Label.Style | None, Data.Format]], label_process: Label.Organization.Basement | None):
                # data root dir
                _root_dir = Directory._Divider_check(root_dir)

                self._data_keyward = {
                    Label.Style.INSTANCES: {
                        Data.Format.annotations: "".join([f"{_root_dir}COCO/annotations/instances_", "{}2017.json"])
                    },
                    Label.Style.KEY_POINTS: {
                        Data.Format.annotations: "".join([f"{_root_dir}COCO/annotations/person_keypoints_", "{}2017.json"])
                    }
                }

                super().__init__(root_dir, mode_list, data_info, label_process)
            
            def _Make_data_block(self, mode_list: List[Process_Name], data_info: List[Tuple[Label.Style, Data.Format]]) -> Dict[Process_Name, List[List]]:
                _block: Dict[Process_Name, List[List]] = {}

                for _mode in mode_list:
                    _block[_mode] = []
                    _img_dir = "".join([self._root_dir, f"COCO/{_mode.value}2017/"])

                    if _mode is not Process_Name.TEST:
                        _block[_mode].append([])  # input image
                        _id_block: Dict[int, int] = {}
                        _annotation_block: Dict[int, List[Dict[str, List]]] = {}

                        # make data index
                        for _info_ct, (_label_style, _data_format) in enumerate(data_info):
                            assert _data_format is Data.Format.annotations, "Calling an unsupported data foramt"
                            _block[_mode].append([])  # annotation

                            _dir, _file_name = File._Extrect_file_name(self._data_keyward[_label_style][_data_format].format(_mode.value), False)
                            _meta_data = File._Json(_dir, _file_name)

                            for _data in _meta_data["annotations"]:
                                _img_id = _data["image_id"]

                                # check image id
                                if _img_id in _id_block.keys():
                                    if (_id_block[_img_id] + 1) < _info_ct:
                                        _id_block.pop(_img_id)
                                        _annotation_block.pop(_img_id)
                                        continue
                                    
                                    _id_block[_img_id] = _info_ct
                                else:
                                    if _info_ct: continue  # in before data info, this image not use 
                                    _id_block.update({_img_id: 0})
                                    _annotation_block.update({_img_id: [{} for _ in range(len(data_info))]})

                                # select annotation file
                                _pick_block = _annotation_block[_img_id][_info_ct]

                                if _label_style is Label.Style.INSTANCES:  # instance
                                    # make annotation data
                                    if _data["iscrowd"]:
                                        ...
                                    else:
                                        _bbox_info = _data["bbox"]
                                        _bbox = (_bbox_info[0], _bbox_info[1], _bbox_info[0] + _bbox_info[2], _bbox_info[1] + _bbox_info[3])
                                        _seg = _data["segmentation"],
                                        _class_id = _data["category_id"]

                                    # updata annotation data
                                    _pick_block["seg"].append(_seg) if "seg" in _pick_block.keys() else _pick_block.update({"seg": [_seg]})
                                    _pick_block["bbox"].append(_bbox) if "bbox" in _pick_block.keys() else _pick_block.update({"bbox": [_bbox]})
                                    _pick_block["class_id"].append(_class_id) if "class_id" in _pick_block.keys() else _pick_block.update({"class_id": [_class_id]})

                                elif _label_style is Label.Style.KEY_POINTS:
                                    ...
                                else:  # captions
                                    ...

                        for _id, _annotations in zip(_id_block.keys(), _annotation_block.values()):
                            _block[_mode][0].append(f"{_img_dir}{_id:0>12d}.jpg")
                            for _ct, _anno in enumerate(_annotations): _block[_mode][1 + _ct].append(_anno)

                    else:  # for test
                        _block[_mode].append(Directory._Search(_img_dir, ext_filter=[".jpg", ]))

                return _block

        # @staticmethod
        # def _Build(
        #     dataset: str, root_dir: str, mode: List[Process_Name], data_info: List[Tuple[Label.Style, Data.Format]], label_meta_file: str | None = None
        # ) -> Basement:
        #     assert dataset in Data_Organization.__dict__.keys()
        #     return Data_Organization.__dict__[dataset](root_dir, mode, data_info, label_meta_file)


class Augment():
    class Supported(Enum):
        ALBUIMIENTATIONS = "Albumentations"
        TORCHVISION = "Torchvision"

    @dataclass
    class Plan():
        amplification: int
        augmentations: List[Augment.Process.Basement]

    class Process():
        class Basement():
            def __init__(
                self,
                output_size: List[int],
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
                apply_to_tensor: bool = False
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
            apply_to_tensor: bool = True,
            **augment_constructer
        ) -> Basement:
            return Augment.Process.__dict__[apply_method.value](output_size, rotate_limit, hflip_rate, vflip_rate, is_norm, norm_mean, norm_std, apply_to_tensor, **augment_constructer)


class Custom_Dataset_Process(Dataset):
    def __init__(
        self,
        data_process: Data.Organize.Basement,
        plan_for_process: Dict[Process_Name, Augment.Plan]
    ):
        self._organization = data_process
        self._plan_for_process = plan_for_process

    # Freeze function
    def __len__(self):
        return len(self._organization) * self._plan_for_process[self._organization._active_mode].amplification

    def __getitem__(self, index) -> Dict[str, Tensor]:
        _augment = self._plan_for_process[self._organization._active_mode]
        _source_index = index // _augment.amplification
        return self._Convert_to_tensor(_source_index, _augment.augmentations)

    def _Set_active_mode_from(self, mode: Process_Name):
        self._organization._Set_active_mode_from(mode)

    # Un-Freeze function
    def _Convert_to_tensor(self, source_index: int, augment: List[Augment.Process.Basement]) -> Dict[str, Tensor]:
        _datas = augment[0](self._organization._Pick_up(source_index))
        _datas.update({"data_info": tensor(source_index)})
        return _datas
