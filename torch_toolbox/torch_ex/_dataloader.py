from dataclasses import dataclass, field
from typing import Dict, List, Any, Union, Optional, Tuple
from enum import Enum
from math import pi, cos, sin, ceil

from torch import Tensor, empty
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, RandomRotation, Normalize, Resize, CenterCrop, Compose, InterpolationMode
import torchvision.transforms.functional as TF

from python_ex._base import Utils
from python_ex._numpy import ndarray
from python_ex._label import Label_Process_Config, Input_Style, Label_Style, IO_Style, Label_Process


if __package__ == "":
    # if this file in local project
    from torch_ex._torch_base import Learning_Mode, JSON_WRITEABLE
else:
    # if this file in package folder
    from ._torch_base import Learning_Mode, JSON_WRITEABLE


# -- DEFINE CONSTNAT -- #
DATA_TYPING = Optional[Union[Tensor, List[Tensor]]]
INFO_TYPING = Optional[Union[ndarray, List[ndarray]]]


DEFUALT_INTERPOLATION: Dict[str, InterpolationMode] = {
    "input": InterpolationMode.NEAREST,
    "label": InterpolationMode.BILINEAR,
}


class Augment_Mode(Enum):
    INPUT = "input"
    LABEL = "label"


class Flip_Direction(Enum):
    NO_FLIP = 0
    HORIZENTAL = 1
    VIRTICAL = 2


# -- DEFINE CONFIG -- #
class Augment_Module_Config():
    @dataclass
    class Base(Utils.Config):
        def _name_check(self, name: str):
            return self.__class__.__name__ == name

    @dataclass
    class Convert_to_Tensor(Base):
        ...

    @dataclass
    class Normalization(Base):
        _Mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.40])
        _Std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
        _Inplace: bool = False

        def _get_parameter(self) -> Dict[str, Any]:
            return {
                "mean": self._Mean,
                "std": self._Std,
                "inplace": self._Inplace}

    @dataclass
    class Resize(Base):
        _Size: List[int]
        _Interpolation: InterpolationMode = InterpolationMode.BILINEAR
        _Max_size: Optional[int] = None
        _Antialias: Optional[bool] = None

        def _get_parameter(self) -> Dict[str, Any]:
            return {
                "size": self._Size,
                "interpolation": self._Interpolation,
                "max_size": self._Max_size,
                "antialias": self._Antialias
            }

        def _convert_to_dict(self) -> Dict[str, JSON_WRITEABLE]:
            _dict = super()._convert_to_dict()
            _dict["_Interpolation"] = self._Interpolation.value
            return _dict

    @dataclass
    class Image_Rotate(Base):
        _Degrees: Union[int, List[int]]  # [-_Degrees, _Degrees] or [min, max]
        _Interpolation: InterpolationMode = InterpolationMode.NEAREST
        _CENTER: Optional[bool] = None
        _Expand: Optional[bool] = None
        _FILL: int = 0

        def _get_parameter(self) -> Dict[str, Any]:
            return {
                "degrees": self._Degrees,
                "interpolation": self._Interpolation,
                "expand": self._Expand,
                "center": self._CENTER,
                "fill": self._FILL}

        def _convert_to_dict(self) -> Dict[str, JSON_WRITEABLE]:
            _dict = super()._convert_to_dict()
            _dict["_Interpolation"] = self._Interpolation.value
            return _dict

    @dataclass
    class Flip(Base):
        _Direction: List[int]

    @dataclass
    class Center_Crop(Base):
        _Size: List[int]

        def _get_parameter(self) -> Dict[str, Any]:
            return {
                "size": self._Size}


@dataclass
class Augment_Config(Utils.Config):
    _Input_style: Input_Style
    _Label_style: Label_Style

    _Data_size: Tuple[int, int]
    _Normalization: Optional[Tuple[List[float], List[float]]] = None
    _Rotate_angle: Optional[Union[int, List[int]]] = None  # [-_Degrees, _Degrees] or [min, max]
    _Flip_direction: Optional[Flip_Direction] = None
    _Interpolation: Optional[InterpolationMode] = None

    def _get_parameter(self):
        _image_size = self._Data_size if self._Rotate_angle is None else self._get_data_size(self._Rotate_angle)
        _sq_dict: Dict[Augment_Mode, List[Augment_Module_Config.Base]] = {Augment_Mode.INPUT: [], Augment_Mode.LABEL: []}
        _inter_mode = self._Interpolation

        # if self._Input_style == Input_Style.IMAGE:

        for _mode, _sq_list in _sq_dict.items():
            _this_interpolation = DEFUALT_INTERPOLATION[_mode.value] if _inter_mode is None else _inter_mode
            _sq_list.append(Augment_Module_Config.Convert_to_Tensor())
            _sq_list.append(Augment_Module_Config.Resize(list(_image_size), _this_interpolation))
            if self._Normalization is not None and _mode is Augment_Mode.INPUT:
                _sq_list.append(Augment_Module_Config.Normalization(self._Normalization[0], self._Normalization[1]))
            if self._Rotate_angle is not None:
                _sq_list.append(Augment_Module_Config.Image_Rotate(self._Rotate_angle, _this_interpolation))
                _sq_list.append(Augment_Module_Config.Convert_to_Tensor())

        return _sq_dict

    def _get_data_size(self, angle: Union[int, List[int]]):
        _degree = angle if isinstance(angle, int) else max(angle)
        _rad = pi * _degree / 180
        _h_dot = ceil(self._Data_size[1] * sin(_rad) + self._Data_size[0] * cos(_rad))
        _w_dot = ceil(self._Data_size[0] * sin(_rad) + self._Data_size[1] * cos(_rad))

        return _h_dot, _w_dot

    def _convert_to_dict(self) -> Dict[str, JSON_WRITEABLE]:
        _dict = super()._convert_to_dict()

        _dict.update({
            "_Interpolation": self._Interpolation if self._Interpolation is None else self._Interpolation.value,
            "_Flip_direction": self._Flip_direction if self._Flip_direction is None else self._Flip_direction.value})

        return _dict


@dataclass
class Dataset_Config(Utils.Config):
    """

    """
    # Parameter for make Label_process
    _Label_config: Label_Process_Config

    _Input_style: Input_Style
    _Input_IO: IO_Style

    _Label_style: Label_Style
    _Label_IO: IO_Style

    _Augmentation: Dict[Learning_Mode, Augment_Config]
    _Amplitude: Dict[Learning_Mode, int]

    def _get_parameter(self, mode: Learning_Mode) -> Dict[str, Any]:
        _label_process = Label_Process._build(**self._Label_config._get_parameter())
        _label_process._set_learning_mode(mode)

        _aug_config_dict = self._Augmentation[mode]._get_parameter()
        _for_rotate = {"angle_holder": [0.0, ]}
        _augment_module = dict((
            _mode,
            Augment_Module.Transform([
                Augment_Module._build(_config, _for_rotate) if _config._name_check("Image_Rotate") else Augment_Module._build(_config) for _config in _configs]
            )
        ) for _mode, _configs in _aug_config_dict.items())

        return {
            "label_process": _label_process,
            "label_style": self._Label_style,
            "label_io": self._Input_IO,
            "input_style": self._Input_style,
            "input_io": self._Label_IO,

            "amplification": self._Amplitude[mode],
            "augmentation": _augment_module}

    def _convert_to_dict(self) -> Dict[str, JSON_WRITEABLE]:
        return {
            "_Label_opt": self._Label_config._convert_to_dict(),
            "_Input_style": self._Input_style.value,
            "_Input_IO": self._Input_IO.value,
            "_Label_style": self._Label_style.value,
            "_Label_IO": self._Label_IO.value,
            "_Amplitude": {learning_key.value: data for learning_key, data in self._Amplitude.items()},
            "_Augmentation": {
                learning_key.value: aug_config._convert_to_dict() for learning_key, aug_config in self._Augmentation.items()}}


# -- Mation Function -- #
class Augment_Module():
    class Convert_to_Tensor(ToTensor):
        ...

    class Normalization(Normalize):
        def __init__(self, mean, std, inplace=False):
            super().__init__(mean, std, inplace)

        def forward(self, tensor: Tensor) -> Tensor:
            if self.mean is None or self.std is None:
                _mean = tensor.mean(dim=list(range(1, len(tensor.shape))))
                _std = tensor.std(dim=list(range(1, len(tensor.shape))))

                return TF.normalize(tensor, _mean.tolist(), _std.tolist(), self.inplace)
            else:
                return super().forward(tensor)
        ...

    class Resize(Resize):
        ...

    class Image_Rotate(RandomRotation):
        def __init__(self, angle_holder: List[float], degrees, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None):
            super().__init__(degrees, interpolation, expand, center, fill, resample)

            self._Angle: List[float] = angle_holder

        def _set_angle(self):
            """Get parameters for ``rotate`` for a random rotation.

            Returns:
                float: angle parameter to be passed to ``rotate`` for random rotation.
            """
            self._Angle[0] = float(empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())

        def forward(self, img):
            """
            Args:
                img (PIL Image or Tensor): Image to be rotated.

            Returns:
                PIL Image or Tensor: Rotated image.
            """
            fill = self.fill
            if isinstance(img, Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * TF.get_image_num_channels(img)
                else:
                    fill = [float(f) for f in fill]   # type: ignore
            return TF.rotate(img, self._Angle[0], self.resample, self.expand, self.center, fill)   # type: ignore

    class Flip():
        ...

    class Center_Crop(CenterCrop):
        ...

    class Transform(Compose):
        def _set_target(self, target: Augment_Mode):
            self._Target = target

        def __call__(self, data: ndarray) -> Tensor:
            _target = self._Target
            _data = data

            for _t in self.transforms:
                if isinstance(_t, Augment_Module.Image_Rotate) and _target == Augment_Mode.INPUT:
                    _t._set_angle()
                _data = _t(_data)
            return _data  # type: ignore

    @staticmethod
    def _build(augment_config: Utils.Config, extra_parameter: Optional[Dict[str, Any]] = None):
        _name = augment_config.__class__.__name__
        if extra_parameter is None:
            return Augment_Module.__dict__[_name](**augment_config._get_parameter())
        else:
            return Augment_Module.__dict__[_name](**augment_config._get_parameter(), **extra_parameter)


class Custom_Dataset(Dataset):
    def __init__(
            self, label_process: Label_Process.Basement, label_style: Label_Style, label_io: IO_Style, input_style: Input_Style, input_io: IO_Style,
            amplification: int, augmentation: Dict[Augment_Mode, Augment_Module.Transform]):
        self._Data_process = label_process
        self._Work_profile = label_process._get_work_profile(label_style, label_io, input_style, input_io)

        self._Activate_class_info = label_process._Activate_label[label_style]
        self._Amplification = amplification
        self._Transform = augmentation

    def _data_transform(self, data_dict: Dict[str, Union[ndarray, List[ndarray]]]) -> Tuple[DATA_TYPING, DATA_TYPING, INFO_TYPING]:
        _info = data_dict["info"]

        _input = self._transform(Augment_Mode.INPUT, data_dict["input"], _info)
        _label = self._transform(Augment_Mode.LABEL, data_dict["label"], _info)

        return _input, _label, _info

    def _transform(self, target: Augment_Mode, data: Union[ndarray, List[ndarray]], info: Union[ndarray, List[ndarray]]):
        if isinstance(data, list):
            _transform_data = [self._Transform[target](_data) for _data in data]
        else:
            _transform_data = self._Transform[target](data)

        return _transform_data

    def __len__(self):
        return len(self._Work_profile._Data_list) * self._Amplification

    def __getitem__(self, index) -> Tuple[DATA_TYPING, DATA_TYPING, INFO_TYPING]:
        _source_index = index // self._Amplification
        return self._data_transform(self._Data_process._work(self._Work_profile, _source_index))
