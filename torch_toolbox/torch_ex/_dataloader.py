from dataclasses import dataclass, field
from typing import Dict, List, Any, Union

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomRotation, InterpolationMode, Normalize, Resize
import torchvision.transforms.functional as TF

from python_ex._base import Utils
from python_ex._label import Label_Process_Config, Input_Style, Label_Style, IO_Style, Label_Process


if __package__ == "":
    # if this file in local project
    from torch_ex._torch_base import Learning_Mode
else:
    # if this file in package folder
    from ._torch_base import Learning_Mode


# -- DEFINE CONFIG -- #
class Augmentation_Config():
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

    @dataclass
    class Flip(Utils.Config):
        _Direction: List[int]

    @dataclass
    class Crop():
        # _Mode:
        ...


@dataclass
class Dataset_Config(Utils.Config):
    """

    """
    # Parameter for make Label_process
    _Label_opt: Label_Process_Config

    _Input_style: Input_Style
    _Input_IO: IO_Style

    _Label_style: Label_Style
    _Label_IO: IO_Style

    _Amplitude: int = 1
    _Augmentaion: Dict[Learning_Mode, Dict[str, List[Utils.Config]]] = field(default_factory=dict)

    def _get_parameter(self) -> Dict[str, Any]:
        return {
            "label_config": self._Label_opt,
            "label_style": self._Input_style,
            "label_io": self._Input_IO,
            "input_style": self._Label_style,
            "input_io": self._Label_IO,

            "amplification": self._Amplitude,
            "augmentation": self._Augmentaion}

    def _convert_to_dict(self) -> Dict[str, Union[Dict, str, int, float, bool, None]]:
        return {
            "_Label_opt": self._Label_opt._convert_to_dict(),
            "_Input_style": self._Input_style.value,
            "_Input_IO": self._Input_IO.value,
            "_Label_style": self._Label_style.value,
            "_Label_IO": self._Label_IO.value,
            "_Amplitude": self._Amplitude,
            "_Augmentaion": {
                learning_key.value: {
                    type_key: {
                        config.__class__.__name__: config._convert_to_dict() for config in config_list} for type_key, config_list in data.items()
                } for learning_key, data in self._Augmentaion.items()}}

    def _restore_from_dict(self, data: Dict[str, Union[Dict, str, int, float, bool, None]]):
        self._Label_opt = self._Label_opt._restore_from_dict(data["_Label_opt"])
        self._Label_style, = Label_Style(data["_Label_style"])
        self._IO_style = IO_Style(data["_IO_style"])


# -- Mation Function -- #
class Augmentation():
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
        ...

    class Flip():
        ...

    @staticmethod
    def _build(Augmentation_list: List[Utils.Config]):
        _componant = [ToTensor(), ]

        for _config in Augmentation_list:
            _name = _config.__class__.__name__
            _componant.append(Augmentation.__dict__[_name](**_config._get_parameter()))

        return Compose(_componant)


class Custom_Dataset(Dataset):
    def __init__(
            self, label_process: Label_Process.Basement, label_style: Label_Style, label_io: IO_Style, input_style: Input_Style, input_io: IO_Style,
            amplification: int, augmentation: Dict[str, List[Augmentation_Config]]):
        self._Data_process = label_process
        self._Work_profile = self._Data_process._get_work_profile(label_style, label_io, input_style, input_io)

        self._Amplification = amplification
        self._Transform = {_process: Augmentation._build(config_list) for _process, config_list in augmentation.items()}

    def _input_process(self, input_data: Union[List[Tensor], Tensor]):
        if isinstance(input_data, list):
            return [self._Transform["input"](data) for data in input_data]
        else:
            return self._Transform["input"](input_data)

    def _label_process(self, label_data: Union[List[Tensor], Tensor]):
        if isinstance(label_data, list):
            return [self._Transform["label"](data) for data in label_data]
        else:
            return self._Transform["label"](label_data)

    def __len__(self):
        return len(self._Work_profile._Data_list * self._Amplification)

    def __getitem__(self, index):
        _source_index = index // self._Amplification
        _data: Dict = self._Data_process._work(self._Work_profile, _source_index)

        _output = ()
        _output += tuple(self.__dict__[f"_{key}_process"](_data[key]) for key in self._Transform)
        _output += (_data["index"], )

        return _output
