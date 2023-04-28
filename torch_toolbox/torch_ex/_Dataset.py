from typing import Tuple, Dict, List, Any
from enum import Enum
from math import pi, cos, sin, ceil
from dataclasses import dataclass

from torch import Tensor, tensor
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from python_ex._base import Directory, File, Utils
# from python_ex._vision import File_IO
# from python_ex._numpy import Array_Process, Np_Dtype, ndarray, Image_Process

if __package__ == "":
    # if this file in local project
    from torch_ex._Torch_base import Process_Name
else:
    # if this file in package folder
    from ._Torch_base import Process_Name


# -- DEFINE CONSTANT -- #
class Label_Style(Enum):
    CLASSIFICATION = "Classification"
    SEMENTIC_SEG = "Mask"


class Data_Format(Enum):
    image = "image"
    colormaps = "colormaps"
    polygons = "polygons"


class Label():
    @dataclass
    class Basement(Utils.Config):
        """
        ### 훈련에 사용되는 물체 정보
        -------------------------------------------------------------------------------------------
        ## Parameters
            identity_num (int)
                : 고유번호
            train_num: (int)
                : 훈련에서 사용되는 구분번호
            cateogry: (str)
                : 물체 범주
            ignore_in_eval: (bool)
                : 평가 포함 여부
            name: (str)
                : 물체 이름
            class_info (Any)
                : 물체 표현 정보
        -------------------------------------------------------------------------------------------
        """
        identity_num: int
        train_num: int
        cateogry: str
        ignore_in_eval: bool
        name: str
        class_info: Any

    @dataclass
    class Classification(Basement):
        class_info: str

    @dataclass
    class Mask(Basement):
        class_info: List[int]


@dataclass
class Data_Profile():
    data_format: Data_Format
    data_list: List[Any]

    def __len__(self):
        return len(self.data_list)

    def _Get_data(self, num: int):
        ...


class Supported_Augment(Enum):
    ALBUIMIENTATIONS = "Albumentations"


# -- Main code -- #
class Label_Organization():
    class Basement():
        # initialize
        def __init__(self, active_style: List[Label_Style], meta_file: str | None = None):
            # Get label data
            # Directory check
            if meta_file is None:
                _meta_dir = Directory._Devide(__file__)[0]
                _meta_file = f"{self.__class__.__name__}.json"
            else:
                _meta_dir, _meta_file = Directory._Devide(meta_file)
                _meta_dir = Directory._Divider_check(_meta_dir) if Directory._Exist_check(_meta_dir) else Directory._Devide(__file__)[0]
                _meta_file = _meta_file if meta_file is not None and File._Exist_check(meta_file) else f"{self.__class__.__name__}.json"

            _raw_data: Dict[str, List[Dict]] = File._Json(file_dir=_meta_dir, file_name=_meta_file)

            self._infomation: Dict[Label_Style, Dict[int, List[Label.Basement]]] = {}

            # Make active_label from meta data
            for _style in active_style:
                self._Make_active_label(_style, [Label.__dict__[_style.value](**data) for data in _raw_data[_style.value]])

        # Freeze function
        def _Make_active_label(self, style: Label_Style, raw_label: List[Label.Basement]):
            self._infomation[style] = {}
            for _label in raw_label:
                if _label.train_num in self._infomation[style].keys():
                    self._infomation[style][_label.train_num].append(_label)
                else:
                    self._infomation[style][_label.train_num] = [_label, ]

        def _Get_label_fron_info_data(self, style: Label_Style, key: str | List[int]):
            _selected_datas = self._infomation[style]
            _pick_info = {}

            for _info, _obj_datas in _selected_datas.items():
                _is_pick = False

                for _data in _obj_datas:
                    if all(key == _data.class_info):
                        _is_pick = True
                        break

                if _is_pick:
                    _pick_info[_info] = _obj_datas
                    break

            return _pick_info

    class Cityscape(Basement):
        ...

    class BDD_100k(Basement):
        ...

    class COCO(Basement):
        ...

    # class Imagenet_1k(Basement):
    #     ...

    # class Imagenet_22k(Basement):
    #     ...

    # class PASCAL(Basement):
    #     ...

    # class CDnet(Basement):
    #     ...

    @staticmethod
    def _Build(label: str, active_style: List[Label_Style], meta_file: str | None = None):
        if meta_file is None:
            assert label in Label_Organization.__dict__.keys()
            Label_Organization.__dict__[label](active_style, meta_file)
        else:
            Label_Organization.Basement(active_style, meta_file)


class Data_Organization():
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
        _default_label: str
        _active_mode: Process_Name = Process_Name.TRAIN

        _data_dir: Dict[Label_Style, Dict[str, str]]

        def __init__(self, root_dir: str, activate_mode: List[Process_Name], data_info: List[Tuple[Label_Style, Data_Format]], label_meta_file: str | None = None):
            # set learning info
            self._apply_mode = activate_mode

            # make label process
            _label = [_data[0] for _data in data_info if _data[0] is not None]
            self._label = Label_Organization._Build(self._default_label, _label, label_meta_file)

            # make meta data from dir
            _root_dir = Directory._Divider_check(root_dir)

            # _data_profile: Dict[Process_Name, Dict[Label.Style, Target.Format]] = {}

            # self._data_holder: Dict[Process_Name, Dict[str, List[Tuple[Target.Format, ]]]] = {}
            # for _mode in activate_mode:
            #     if _mode is Process_Name.TEST:
            #         self._data_holder[_mode] = dict((_target.value, dict()) for _style, _target in data_format if _style is None)
            #     else:
            #         self._data_holder[_mode] = dict((_target.value, dict()) for _, _target in data_format)
            self._data_profiles = self._Make_dataprofiles(_root_dir, activate_mode, data_info)

        def __len__(self):
            raise NotImplementedError

        def _Set_active_mode_from(self, mode: Process_Name):
            if mode in self._apply_mode:
                self._active_mode = mode
            else:
                # in later, warning coment here!
                pass

        def _Make_dataprofiles(
            self, root_dir: str, activate_mode: List[Process_Name], data_info: List[Tuple[Label_Style, Data_Format]]
        ) -> Dict[Process_Name, Dict[str, List[Data_Profile]]]:
            raise NotImplementedError

        def _Get_data_clusturing(self):
            _profiles = self._data_profiles[self._active_mode]
            _clustur = {}
            for _key, _data in _profiles.items():
                _clustur.update(dict((f"{_key}{_ct}", _key) for _ct in range(1, len(_data))))

            return _clustur if len(_clustur) else None

        def _Get_target(self, num: int) -> Dict[str, Any]:
            raise NotImplementedError

    class BDD_100k(Basement):
        _default_label = "BDD_100k"

        _data_dir = {
            Label_Style.SEMENTIC_SEG: {
                "image": "bdd100k/images/10k/{}/",
                "label": "bdd100k/labels/sem_seg/{}/{}/"
            },
        }

        def __len__(self):
            return len(self._data_profiles[self._active_mode]["image"][0])

        def _Make_dataprofiles(
            self, root_dir: str, activate_mode: List[Process_Name], data_info: List[Tuple[Label_Style, Data_Format]]
        ) -> Dict[Process_Name, Dict[str, List[Data_Profile]]]:

            _profiles: Dict[Process_Name, Dict[str, List[Data_Profile]]] = {}
            for _mode in activate_mode:
                _profiles[_mode] = {}
                for _style, _format in data_info:
                    if _style is Label_Style.SEMENTIC_SEG:
                        _input_list = Directory._Search(Directory._Divider.join([root_dir, self._data_dir[_style]["image"].format(_mode.value)]))
                        _profiles[_mode]["image"] = [Data_Profile(Data_Format.image, _input_list), ]

                        if _mode is not Process_Name.TEST:
                            _label_list = Directory._Search(Directory._Divider.join([root_dir, self._data_dir[_style]["label"].format(_format.value, _mode.value)]))
                            _profiles[_mode]["mask"] = [Data_Profile(_format, _label_list), ]

            return _profiles

        def _Get_data_clusturing(self):
            _profiles = self._data_profiles[self._active_mode]
            _clustur = {}
            for _key, _data in _profiles.items():
                _clustur.update(dict((f"{_key}{_ct}", _key) for _ct in range(1, len(_data))))

            return _clustur

        def _Get_target(self, num: int):
            _actrive_data = self._data_profiles[self._active_mode]
            return dict(
                (
                    f"{_info_style}{_ct - 1}" if _ct else f"{_info_style}",
                    _profile
                ) for _info_style, _profiles in _actrive_data.items() for _ct, _profile in enumerate(_profiles))

    class COCO(Basement):
        ...

    @staticmethod
    def _Build(
        dataset: str, root_dir: str, mode: List[Process_Name], data_info: List[Tuple[Label_Style, Data_Format]], label_meta_file: str | None = None
    ) -> Basement:
        assert dataset in Data_Organization.__dict__.keys()
        return Data_Organization.__dict__[dataset](root_dir, mode, data_info, label_meta_file)


class Augment():
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
            rotate_limit: int | List[int] = 0,
            hflip_rate: float = 0.0,
            vflip_rate: float = 0.0,
            is_norm: bool = True,
            norm_mean: List[float] = [0.485, 0.456, 0.406],
            norm_std: List[float] = [0.229, 0.224, 0.225],
            apply_to_tensor: bool = False
        ) -> List:
            _transform_list = [self._To_tensor(), ] if apply_to_tensor else []

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

            return _transform_list

        def _Get_padding_size(self, output_size: List[int], rotate_limit: int | List[int] = 0) -> List[int]:
            raise NotImplementedError

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
        def _Get_padding_size(self, output_size: List[int], rotate_limit: int = 0) -> List[int]:
            _rad = pi * rotate_limit / 180
            _h = ceil(output_size[1] * sin(_rad) + output_size[0] * cos(_rad))
            _w = ceil(output_size[0] * sin(_rad) + output_size[1] * cos(_rad))

            return [_h, _w]

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
                keypoints_parameter: Dict[str, str] | None = None):
            return A.Compose(transform_list, bbox_parameter, keypoints_parameter, additional_targets=group_parmaeter)

    @staticmethod
    def _Build(
        apply_mathod: Supported_Augment,
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
        return Augment.__dict__[apply_mathod.value](output_size, rotate_limit, hflip_rate, vflip_rate, is_norm, norm_mean, norm_std, apply_to_tensor, **augment_constructer)


class Custom_Dataset_Process(Dataset):
    def __init__(
        self,
        organization: Data_Organization.Basement,
        amplification: Dict[Process_Name, int],
        augmentations: Dict[Process_Name, List[Augment.Basement]]
    ):
        self._apply_mode = organization._apply_mode
        self._organization = organization
        self._amplification = amplification
        self._augment = augmentations

    # Freeze function
    def __len__(self):
        return len(self._organization) * self._amplification[self._organization._active_mode]

    def __getitem__(self, index) -> Dict[str, Tensor]:
        _source_index = index // self._amplification
        return self._Convert_to_tensor(_source_index)

    def _Set_active_mode_from(self, mode: Process_Name):
        self._organization._Set_active_mode_from(mode)

    # Un-Freeze function
    def _Convert_to_tensor(self, _source_index: int) -> Dict[str, Tensor]:
        _datas = self._augment[self._organization._active_mode][0](self._organization._Get_target(_source_index))
        _datas.update({"data_info": tensor(_source_index)})
        return _datas
