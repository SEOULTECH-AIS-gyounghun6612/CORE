from dataclasses import dataclass
from typing import Dict, Any
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from python_ex._base import Utils
from python_ex._label import Label_Config, Label_Style, IO_Style, Label_Process
from python_ex._numpy import image_process


if __package__ == "":
    from _torch_base import Learning_Mode, Torch_Utils

else:
    from ._torch_base import Learning_Mode, Torch_Utils


# -- DEFINE CONFIG -- #
@dataclass
class Dataset_Config(Utils.Config):
    """

    """
    # Parameter for make Label_process
    _Label_opt: Label_Config
    _Label_style: Label_Style
    _IO_style: IO_Style

    def _convert_to_dict(self):
        return {
            "_Label_opt": self._Label_opt._convert_to_dict(),
            "_Label_style": self._Label_style.value,
            "_IO_style": self._IO_style.value}

    def _restore_from_dict(self, data: Dict[str, Any]):
        self._Label_opt = self._Label_opt._restore_from_dict(data["_Label_opt"])
        self._Label_style, = Label_Style(data["_Label_style"])
        self._IO_style = IO_Style(data["_IO_style"])

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

    def _convert_to_dict(self) -> Dict[str, Any]:
        _dict = {
            "_Dataset_opt": self._Dataset_opt._convert_to_dict(),
            "_Batch_size": self._Batch_size,
            "_Num_workers": self._Num_workers}
        return _dict

    def _restore_from_dict(self, data: Dict[str, Any]):
        self._Dataset_opt._restore_from_dict(data["_Dataset_opt"])
        self._Batch_size = data["_Batch_size"]
        self._Num_workers = data["_Num_workers"]

    def _make_dataloader(self, mode: Learning_Mode):
        return DataLoader(
            dataset=Custom_Dataset(self._Dataset_opt, mode),
            batch_size=self._Batch_size,
            num_workers=self._Num_workers,
            shuffle=(mode == Learning_Mode.TRAIN))


# -- Mation Function -- #
class Custom_Dataset(Dataset):
    def __init__(self, mode: Learning_Mode, label_process: Label_Process.Basement, label_style: Label_Style, file_style: IO_Style):
        super().__init__()
        self.data_process = label_process
        self.data_process.set_learning_mode(mode.value)
        self.data_profile = self.data_process.get_data_profile(label_style, file_style)

    def _len_(self):
        return len(self.data_profile._Input)

    def _getitem_(self, index):
        _data = self.data_process.work(self.data_profile, index)
        _input = image_process.image_normalization(_data["input"])
        _input = image_process.conver_to_first_channel(_input)
        _input = Torch_Utils.Tensor._from_numpy(_input)

        _label = image_process.conver_to_first_channel(_data["label"])
        _info = _data["info"]

        return _input, _label, _info


class Custom_Dataloader():
    def __init__(self) -> None:
        pass

    def _set_activate_mode(self, mode: Learning_Mode):
        self._Activate_mode = mode
    
    def _set_dataset(self, label_process: Label_Process.Basement, label_style: Label_Style, file_style: IO_Style, is_distribute: bool = False):
        self._Dataset = Custom_Dataset(self._Activate_mode, label_process, label_style, file_style)

        if is_distribute:
            self._Smapler = DistributedSampler(self._Dataset, )

    def _make(self):
        ...

def _make_dataloader(mode: Learning_Mode, label_process: Label_Process.Basement, label_style: Label_Style, file_style: IO_Style, batch_size: int, ):
    _dataset = Custom_Dataset(mode, label_process, label_style, file_style)


    return DataLoader(
            dataset=Custom_Dataset(self._Dataset_opt, mode),
            batch_size=self._Batch_size,
            num_workers=self._Num_workers,
            shuffle=(mode == Learning_Mode.TRAIN))
