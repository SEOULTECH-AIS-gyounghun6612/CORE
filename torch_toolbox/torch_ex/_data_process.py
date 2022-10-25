from dataclasses import dataclass
from typing import Dict, Any
from torch.utils.data import DataLoader, Dataset

from python_ex._base import Utils
from python_ex._label import Label_Config, Label_Style, IO_Style, Label_Process
from python_ex._numpy import image_process


if __package__ == "":
    from _torch_base import Learning_Mode, Torch_Utils

else:
    from ._torch_base import Learning_Mode, Torch_Utils


# -- DEFINE CONFIG -- #
@dataclass
class Augmentation_Config(Utils.Config):
    def _convert_to_dict(self) -> Dict[str, Any]:
        return super()._convert_to_dict()

    def _restore_from_dict(self, data: Dict[str, Any]):
        return super()._restore_from_dict(data)


@dataclass
class Dataset_Config(Utils.Config):
    """

    """
    # Parameter for make Dataset_function
    _Name: str

    # Parameter for make Label_process
    _Label_opt: Label_Config
    _Label_style: Label_Style
    _IO_style: IO_Style

    _Amplitude: int
    _Augmentation: Augmentation_Config

    def _convert_to_dict(self):
        return {
            "_Label_opt": self._Label_opt._convert_to_dict(),
            "_Label_style": self._Label_style.value,
            "_IO_style": self._IO_style.value,
            "_Amplitude": self._Amplitude,
            "_Augmentation": self._Augmentation._convert_to_dict()}

    def _restore_from_dict(self, data: Dict[str, Any]):
        self._Label_opt = self._Label_opt._restore_from_dict(data["_Label_opt"])
        self._Label_style, = Label_Style(data["_Label_style"])
        self._IO_style = IO_Style(data["_IO_style"])
        self._Amplitude = data["_Amplitude"]
        self._Augmentation = self._Augmentation._restore_from_dict(data["_Augmentation"])

    def _make_label_process(self) -> Label_Process.Basement:
        return Label_Process._build(self._Label_opt)


@dataclass
class Dataloder_Config(Utils.Config):
    """

    """
    # parameter for build Dataset
    _Dataset_opt: Dataset_Config

    # parameter for build Dataloader
    _Batch_size: int
    _Num_workers: int

    def _convert_to_dict(self):
        ...

    def _restore_from_dict(self, data: Dict[str, Any]):
        ...

    def _make_dataloader(self, mode: Learning_Mode):
        return DataLoader(
            dataset=Custom_Dataset(self._Dataset_opt, mode),
            batch_size=self._Batch_size,
            num_workers=self._Num_workers,
            shuffle=(mode == Learning_Mode.TRAIN))


# -- Mation Function -- #
class Augmentation():
    class Resize():
        ...

    class Rotate():
        ...

    @staticmethod
    def build(config: Augmentation_Config):
        ...


class Custom_Dataset(Dataset):
    def __init__(self, config: Dataset_Config, mode: Learning_Mode):
        super().__init__()
        self.data_process = config._make_label_process()
        self.data_process.set_learning_mode(mode.value)
        self.data_profile = self.data_process.get_data_profile(config._Label_style, config._IO_style)

    def __len__(self):
        return len(self.data_profile._Input)

    def __getitem__(self, index):
        __data = self.data_process.work(self.data_profile, index)
        __input = image_process.image_normalization(__data["input"])
        __input = image_process.conver_to_first_channel(__input)
        __input = Torch_Utils.Tensor._from_numpy(__input)

        __label = image_process.conver_to_first_channel(__data["label"])
        __info = __data["info"]

        return __input, __label, __info
