from dataclasses import dataclass, field
from typing import Dict, List, Any, Union
from enum import Enum
from math import pi, cos, sin, ceil

from torch import Tensor, empty, float64
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, RandomRotation, Normalize, Resize, CenterCrop, Compose, InterpolationMode
import torchvision.transforms.functional as TF

from python_ex._base import Utils
from python_ex._numpy import ndarray
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
class Augmentation_Module_Config():
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
class Augmentation_Config(Utils.Config):
    _Input_Augmentation: List[Utils.Config] = field(
        default_factory=lambda: [Augmentation_Module_Config.Convert_to_Tensor(), Augmentation_Module_Config.Normalization()])
    _Label_Augmentation: List[Utils.Config] = field(
        default_factory=lambda: [Augmentation_Module_Config.Convert_to_Tensor(), ])
    _Common_Augmentation: List[Utils.Config] = field(
        default_factory=lambda: [])

    def _get_parameter(self) -> Dict[Augmentation_Target, List[Utils.Config]]:
        return {
            Augmentation_Target.INPUT: self._Input_Augmentation,
            Augmentation_Target.LABEL: self._Label_Augmentation,
            Augmentation_Target.COMMON: self._Common_Augmentation}

    def _convert_to_dict(self) -> Dict[str, Union[Dict, str, int, float, bool, None]]:
        return {
            "_Input_Augmentation": {config.__class__.__name__: config._convert_to_dict() for config in self._Input_Augmentation},
            "_Label_Augmentation": {config.__class__.__name__: config._convert_to_dict() for config in self._Label_Augmentation},
            "_Common_Augmentation": {config.__class__.__name__: config._convert_to_dict() for config in self._Common_Augmentation}}

    def _restore_from_dict(self, data: Dict[str, Union[Dict, str, int, float, bool, None]]):
        return super()._restore_from_dict(data)


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
    _Augmentation: Dict[Learning_Mode, Augmentation_Config] = field(
        default_factory=lambda: {
            Learning_Mode.TRAIN: Augmentation_Config(),
            Learning_Mode.VALIDATION: Augmentation_Config(),
            Learning_Mode.TEST: Augmentation_Config(_Label_Augmentation=[])})

    def _get_parameter(self, mode: Learning_Mode) -> Dict[str, Any]:
        _label_process = Label_Process._build(**self._Label_config._get_parameter())
        _label_process._set_learning_mode(mode)

        _aug_config = self._Augmentation[mode]._get_parameter()
        _aug = {}

        for _target in _aug_config.keys():
            if _target == Augmentation_Target.COMMON:
                _rotate_check = [
                    _config.__dict__["_Degrees"] if "_Degrees" in _config.__dict__.keys() else 0 for _config in _aug_config[_target]]
                _resize_config = Augmentation_Module_Config.Resize(self._get_data_size(sum(_rotate_check))) if sum(_rotate_check) \
                    else Augmentation_Module_Config.Resize(self._Data_size)
                _aug_config[_target] = [_resize_config, ] + _aug_config[_target]
                if sum(_rotate_check):
                    _aug_config[_target].append(Augmentation_Module_Config.Center_Crop(self._Data_size))

            elif isinstance(_aug_config[_target], list):
                if not isinstance(_aug_config[_target][0], Augmentation_Module_Config.Convert_to_Tensor):
                    _aug_config[_target] = [Augmentation_Module_Config.Convert_to_Tensor(), ] + _aug_config[_target]

            else:
                _aug_config[_target] = [Augmentation_Module_Config.Convert_to_Tensor()]

            _aug[_target] = Augmentation_Module._build(_aug_config[_target])

        return {
            "label_process": _label_process,
            "label_style": self._Label_style,
            "label_io": self._Input_IO,
            "input_style": self._Input_style,
            "input_io": self._Label_IO,

            "amplification": self._Amplitude[mode],
            "augmentation": _aug}

    def _convert_to_dict(self) -> Dict[str, Union[Dict, str, int, float, bool, None]]:
        return {
            "_Label_opt": self._Label_config._convert_to_dict(),
            "_Input_style": self._Input_style.value,
            "_Input_IO": self._Input_IO.value,
            "_Label_style": self._Label_style.value,
            "_Label_IO": self._Label_IO.value,
            "_Amplitude": {learning_key.value: data for learning_key, data in self._Amplitude.items()},
            "_Augmentation": {
                learning_key.value: aug_config._convert_to_dict() for learning_key, aug_config in self._Augmentation.items()}}

    def _restore_from_dict(self, data: Dict[str, Union[Dict, str, int, float, bool, None]]):
        self._Label_config = self._Label_config._restore_from_dict(data["_Label_opt"])
        self._Label_style, = Label_Style(data["_Label_style"])
        self._IO_style = IO_Style(data["_IO_style"])

    def _get_data_size(self, degrees: float):
        _rad = pi * degrees / 180
        _h_dot = ceil(self._Data_size[1] * sin(_rad) + self._Data_size[0] * cos(_rad))
        _w_dot = ceil(self._Data_size[0] * sin(_rad) + self._Data_size[1] * cos(_rad))

        return [_h_dot, _w_dot]


# -- Mation Function -- #
class Augmentation_Module():
    _Defualt_interpolation: Dict = {
        Augmentation_Target.INPUT: InterpolationMode.NEAREST,
        Augmentation_Target.LABEL: InterpolationMode.BILINEAR
    }

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
        def _set_angle(self):
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

    class Transform(Compose):
        def _set_target(self, target: Augmentation_Target):
            self._Target = target

        def __call__(self, img, use_auto: bool = False):
            _target = self._Target

            for _t in self.transforms:
                if isinstance(_t, Augmentation_Module.Resize):
                    _t.interpolation = Augmentation_Module._Defualt_interpolation[_target] if use_auto else _t.interpolation
                elif isinstance(_t, Augmentation_Module.Rotate):
                    _t.interpolation = Augmentation_Module._Defualt_interpolation[_target] if use_auto else _t.interpolation
                    if _target == Augmentation_Target.INPUT:
                        _t._set_angle()

                img = _t(img)
            return img

    @staticmethod
    def _build(Augmentation_config_list: List[Utils.Config]):
        _componant = []

        for _config in Augmentation_config_list:
            _name = _config.__class__.__name__
            _componant.append(Augmentation_Module.__dict__[_name](**_config._get_parameter()))

        return Augmentation_Module.Transform(_componant)


class Custom_Dataset(Dataset):
    def __init__(
            self, label_process: Label_Process.Basement, label_style: Label_Style, label_io: IO_Style, input_style: Input_Style, input_io: IO_Style,
            amplification: int, augmentation: Dict[Augmentation_Target, Augmentation_Module.Transform]):
        self._Data_process = label_process
        self._Work_profile = label_process._get_work_profile(label_style, label_io, input_style, input_io)

        self._Activate_class_info = label_process._Activate_label[label_style]
        self._Amplification = amplification
        self._Transform = augmentation

    def _transform(self, data_dict: Dict[str, Union[ndarray, List[ndarray]]]):
        _holder: Dict[str, Union[Tensor, ndarray, List[Tensor], List[ndarray]]] = {}
        for _target, _data in data_dict.items():
            if _target == "info":
                _holder[_target] = _data

            else:
                _holder[_target] = self._transform_process(Augmentation_Target(_target), _data)

        return tuple([_data.astype(float64) for _data in _holder.values()])

    def _transform_process(self, target: Augmentation_Target, data: Union[Tensor, ndarray, List[Tensor], List[ndarray]]) -> Union[Tensor, List[Tensor]]:
        _data = data
        _process = self._Transform[target]
        _process._set_target(target)

        _data = [_process(_mem) for _mem in _data] if isinstance(_data, list) else _process(_data)

        _common_process = self._Transform[Augmentation_Target.COMMON]
        _common_process._set_target(target)

        return [_common_process(_mem, True) for _mem in _data] if isinstance(_data, list) else _common_process(_data, True)

    def __len__(self):
        return len(self._Work_profile._Data_list) * self._Amplification

    def __getitem__(self, index):
        _source_index = index // self._Amplification
        return self._transform(self._Data_process._work(self._Work_profile, _source_index))
