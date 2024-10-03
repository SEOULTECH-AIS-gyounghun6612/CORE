""" ###

------------------------------------------------------------------------
### Requirement
    -

### Structure
    -

"""
from __future__ import annotations
from typing import Any, Callable
from dataclasses import dataclass, field

from torch.utils.data import Dataset, DataLoader

from python_ex.project import Config


class Parser():
    def __init__(
        self,
        data_dir: str,
        learning_mode: str,
        **kwarg
    ) -> None:
        self.data_info: dict[str, Any] = {
            "learning_mode": learning_mode
        }
        self.data_block: dict[str, list] = {}

        self.Get_data_from(data_dir, **kwarg)

    def Get_data_from(self, data_dir: str, **kwarg):
        raise NotImplementedError


class Dataset_Basement(Dataset):
    def __init__(
        self,
        data_dir: str,
        learning_mode: str,
        data_parser: type[Parser] = Parser,
        **parser_kwarg
    ) -> None:
        _parser = data_parser(data_dir, learning_mode, **parser_kwarg)
        self.data_info = _parser.data_info
        self.data_block = _parser.data_block

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index) -> Any:
        raise NotImplementedError


@dataclass
class Dataset_Config(Config):
    name: str = "no_data"
    data_dir: str = "./datasets"
    additional: dict = field(default_factory=dict)

    def Build_dataset(
        self, learning_type: str
    ) -> Dataset_Basement:
        raise NotImplementedError

    def Get_summation(self):
        return [self.name]


@dataclass
class Dataloader_Config(Config):
    dataset_config: Dataset_Config = field(default_factory=Dataset_Config)

    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 0
    collate_fn: str | None = None
    drop_last: bool = False

    def _Build_collate_fn(self) -> Callable | None:
        raise NotImplementedError

    def Builde_dataloder(self, learning_type: str) -> tuple[int, DataLoader]:
        _dataset = self.dataset_config.Build_dataset(learning_type)
        try:
            _collate_fn = self._Build_collate_fn()
        except NotImplementedError:
            _collate_fn = None

        return len(_dataset), DataLoader(
            _dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
            drop_last=self.drop_last)

    def Set_from(self, source: dict[str, Any]):
        for _k, _v in source.items():
            if _k in self.__dict__:
                if _k == "dataset_config":
                    self.dataset_config.Set_from(_v)
                else:
                    self.__dict__[_k] = _v

    def Get_summation(self):
        return self.dataset_config.Get_summation()
