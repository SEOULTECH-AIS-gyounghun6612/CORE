from typing import Dict, List
from torch.utils.data import DataLoader, Dataset

# fron python_ex import _label

if __package__ == "":
    import _base

else:
    from . import _base


class dataloader():
    def __init__(self) -> None:
        pass


class data_worker():
    def __init__(self, opt: _base.opt._data) -> None:
        self.opt = opt
        self.dataset: Dataset = None
        self.dataloaders: Dict[str, DataLoader] = {}

        self.info = {
            "data_size": self.opt.Data_size,
            "minibatch": self.opt.Batch_size,
        }

        self.make_dataset()

    def make_dataset(self):
        self.dataset = self.opt.Dataset
        # self.log.info_update("label", # about label info; like bdd-100k, seg etc...)

    def get_dataloader(self, learning_modes: List[str]) -> Dict[str, DataLoader]:
        for _mode in learning_modes:
            _data = self.dataset(self.opt.Data_root, self.opt.Data_size, _mode)
            _dataloader = DataLoader(_data, batch_size=self.opt.Batch_size, num_workers=self.opt.Num_worker, shuffle=_mode == "train")
            self.dataloaders[_mode] = _dataloader
            self.info[_mode] = {"data_length": _dataloader.dataset.__len__()}

        return self.dataloaders


# dataset template
class BDD_100K():
    pass
