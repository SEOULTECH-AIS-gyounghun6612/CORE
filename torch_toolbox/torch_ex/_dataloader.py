from dataclasses import dataclass, field
from typing import Dict, List, Any, Union
from enum import Enum
from math import pi, cos, sin, ceil

from torch import Tensor, empty
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, RandomRotation, InterpolationMode, Normalize, Resize, CenterCrop
import torchvision.transforms.functional as TF

from python_ex._base import Utils
from python_ex._label import Label_Process_Config, Input_Style, Label_Style, IO_Style, Label_Process


if __package__ == "":
    # if this file in local project
    from torch_ex._torch_base import Learning_Mode
else:
    # if this file in package folder
    from ._torch_base import Learning_Mode


# -- DEFINE CONSTNAT -- #
class Augmentation_Target(Enum):
    INPUT = "input"
    LABEL = "label"
    COMMON = "common"


# -- DEFINE CONFIG -- #
class Augmentation_Config():
    @dataclass
    class Convert_to_Tensor(Utils.Config):
        ...

    @dataclass
    class Normalization(Utils.Config):
        _Mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.40])
        _Std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
        _Inplace: bool = False

        def _get_parameter(self) -> Dict[str, Any]:
            return {
                "mean": self._Mean,
                "std": self._Std,
                "inplace": self._Inplace}

    @dataclass
    class Resize(Utils.Config):
        _Size: List[int]
        _Interpolation: InterpolationMode = InterpolationMode.BILINEAR
        _Max_size: int = None
        _Antialias: bool = None

        def _get_parameter(self) -> Dict[str, Any]:
            return {
                "size": self._Size,
                "interpolation": self._Interpolation,
                "max_size": self._Max_size,
                "antialias": self._Antialias
            }

        def _convert_to_dict(self) -> Dict[str, Union[Dict, str, int, float, bool, None]]:
            _dict = super()._convert_to_dict()
            _dict["_Interpolation"] = self._Interpolation.value
            return _dict

    @dataclass
    class Rotate(Utils.Config):
        _Degrees: Union[int, List[int]]  # [-_Degrees, _Degrees] or [min, max]
        _Interpolation: InterpolationMode = InterpolationMode.NEAREST
        _CENTER: bool = None
        _Expand: bool = None
        _FILL: int = 0

        def _get_parameter(self) -> Dict[str, Any]:
            return {
                "degrees": self._Degrees,
                "interpolation": self._Interpolation,
                "expand": self._Expand,
                "center": self._CENTER,
                "fill": self._FILL}

        def _convert_to_dict(self) -> Dict[str, Union[Dict, str, int, float, bool, None]]:
            _dict = super()._convert_to_dict()
            _dict["_Interpolation"] = self._Interpolation.value
            return _dict

    @dataclass
    class Flip(Utils.Config):
        _Direction: List[int]

    @dataclass
    class Center_Crop():
        _Size: List[int]

        def _get_parameter(self) -> Dict[str, Any]:
            return {
                "size": self._Size}


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

    _Data_size: List[int]
    _Amplitude: Dict[Learning_Mode, int] = field(default_factory=dict)
    _Augmentation: Dict[Learning_Mode, Dict[Augmentation_Target, List[Utils.Config]]] = field(default_factory=dict)

    def _get_parameter(self, mode: Learning_Mode) -> Dict[str, Any]:
        _label_process = Label_Process._build(**self._Label_config._get_parameter())
        _label_process._set_learning_mode(mode)

        if Augmentation_Target.COMMON in self._Augmentation[mode]:
            _use_rotate = False
            for _aug in self._Augmentation[mode][Augmentation_Target.COMMON]:
                if isinstance(_aug, Augmentation_Config.Rotate):
                    _use_rotate = True
                    _rad = pi * _aug._Degrees / 180
                    _h_dot = ceil(self._Data_size[1] * sin(_rad) + self._Data_size[0] * cos(_rad))
                    _w_dot = ceil(self._Data_size[0] * sin(_rad) + self._Data_size[1] * cos(_rad))
                    _resize = Augmentation_Config.Resize([_h_dot, _w_dot])
                    _crop = Augmentation_Config.Center_Crop(self._Data_size)

                    self._Augmentation[mode][Augmentation_Target.COMMON] = [_resize, ] + self._Augmentation[mode][Augmentation_Target.COMMON] + [_crop, ]
                    break

            if not _use_rotate:
                _resize = Augmentation_Config.Resize(self._Data_size)
                self._Augmentation[mode][Augmentation_Target.COMMON] = [_resize, ] + self._Augmentation[mode][Augmentation_Target.COMMON]
        return {
            "label_process": _label_process,
            "label_style": self._Label_style,
            "label_io": self._Input_IO,
            "input_style": self._Input_style,
            "input_io": self._Label_IO,

            "amplification": self._Amplitude[mode],
            "augmentation": self._Augmentation[mode]}

    def _convert_to_dict(self) -> Dict[str, Union[Dict, str, int, float, bool, None]]:
        return {
            "_Label_opt": self._Label_config._convert_to_dict(),
            "_Input_style": self._Input_style.value,
            "_Input_IO": self._Input_IO.value,
            "_Label_style": self._Label_style.value,
            "_Label_IO": self._Label_IO.value,
            "_Amplitude": {learning_key.value: data for learning_key, data in self._Amplitude.items()},
            "_Augmentation": {
                learning_key.value: {
                    target.value: {
                        config.__class__.__name__: config._convert_to_dict() for config in config_list} for target, config_list in data.items()
                } for learning_key, data in self._Augmentation.items()}}

    def _restore_from_dict(self, data: Dict[str, Union[Dict, str, int, float, bool, None]]):
        self._Label_config = self._Label_config._restore_from_dict(data["_Label_opt"])
        self._Label_style, = Label_Style(data["_Label_style"])
        self._IO_style = IO_Style(data["_IO_style"])


# -- Mation Function -- #
class Augmentation():
    class Convert_to_Tensor(ToTensor):
        ...

    class Normalization(Normalize):
        def __init__(self, mean, std, inplace=False):
            super().__init__(mean, std, inplace)

        def forward(self, tensor: Tensor) -> Tensor:
            if self.mean is None or self.std is None:
                _mean = tensor.mean(dim=list(range(1, len(tensor.shape))))
                _std = tensor.std(dim=list(range(1, len(tensor.shape))))

                return TF.normalize(tensor, _mean, _std, self.inplace)
            else:
                return super().forward(tensor)
        ...

    class Resize(Resize):
        ...

    class Rotate(RandomRotation):
        def _set_angle(self) -> float:
            """Get parameters for ``rotate`` for a random rotation.

            Returns:
                float: angle parameter to be passed to ``rotate`` for random rotation.
            """
            self.angle = float(empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())

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
                    fill = [float(f) for f in fill]
            return TF.rotate(img, self.angle, self.resample, self.expand, self.center, fill)

    class Flip():
        ...

    class Center_Crop(CenterCrop):
        ...

    @staticmethod
    def _build(Augmentation_list: List[Utils.Config]):
        _componant = []

        for _config in Augmentation_list:
            _name = _config.__class__.__name__
            _componant.append(Augmentation.__dict__[_name](**_config._get_parameter()))

        return _componant


class Custom_Dataset(Dataset):
    def __init__(
            self, label_process: Label_Process.Basement, label_style: Label_Style, label_io: IO_Style, input_style: Input_Style, input_io: IO_Style,
            amplification: int, augmentation: Dict[Augmentation_Target, List[Augmentation_Config]]):
        self._Data_process = label_process
        self._Work_profile = self._Data_process._get_work_profile(label_style, label_io, input_style, input_io)

        self._Activate_class_info = self._Data_process._Activate_label[label_style]
        self._Amplification = amplification
        self._Transform = {_process: Augmentation._build(config_list) for _process, config_list in augmentation.items()}

    def _transform(self, input_data: Union[List[Tensor], Tensor], label_data: Union[List[Tensor], Tensor], info: Dict):
        _input_data = input_data
        _label_data = label_data
        for _tr in self._Transform[Augmentation_Target.INPUT]:
            _input_data = [_tr(data) for data in _input_data] if isinstance(_input_data, list) else _tr(_input_data)

        for _tr in self._Transform[Augmentation_Target.LABEL]:
            _label_data = [_tr(data) for data in _label_data] if isinstance(_label_data, list) else _tr(_label_data)

        for _tr in self._Transform[Augmentation_Target.COMMON]:
            if isinstance(_tr, (Augmentation.Rotate, Augmentation.Resize)):
                if isinstance(_tr, Augmentation.Rotate):
                    _tr._set_angle()
                _tr.interpolation = InterpolationMode.NEAREST
                _input_data = [_tr(data) for data in _input_data] if isinstance(_input_data, list) else _tr(_input_data)
                _tr.interpolation = InterpolationMode.BILINEAR
                _label_data = [_tr(data) for data in _label_data] if isinstance(_label_data, list) else _tr(_label_data)
            else:
                _input_data = [_tr(data) for data in _input_data] if isinstance(_input_data, list) else _tr(_input_data)
                _label_data = [_tr(data) for data in _label_data] if isinstance(_label_data, list) else _tr(_label_data)

        return _input_data, _label_data, info

    def _input_process(self, input_data: Union[List[Tensor], Tensor]):
        if isinstance(input_data, list):
            return [self._Transform[Augmentation_Target.INPUT](data) for data in input_data]
        else:
            return self._Transform[Augmentation_Target.INPUT](input_data)

    def _label_process(self, label_data: Union[List[Tensor], Tensor]):
        if isinstance(label_data, list):
            return [self._Transform[Augmentation_Target.LABEL](data) for data in label_data]
        else:
            return self._Transform[Augmentation_Target.LABEL](label_data)

    def __len__(self):
        return len(self._Work_profile._Data_list) * self._Amplification

    def __getitem__(self, index):
        _source_index = index // self._Amplification
        _data: Dict = self._Data_process._work(self._Work_profile, _source_index)

        return self._transform(_data["input"], _data["label"], _data["info"])
