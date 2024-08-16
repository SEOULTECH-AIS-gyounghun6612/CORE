""" ###

------------------------------------------------------------------------
### Requirement
    -

### Structure
    -

"""
from __future__ import annotations
from typing import (
    Any, TypeVar, Callable
)
from dataclasses import dataclass, field

from torch.utils.data import Dataset


class Parser():
    def __init__(self, data_dir: str, **kwarg) -> None:
        self.data_info: dict[str, Any]
        self.data_block: dict[str, list] = {}

        self.Get_data_from(data_dir, **kwarg)

    def Get_data_from(self, data_dir: str, **kwarg):
        raise NotImplementedError


PARSER = TypeVar("PARSER", bound=Parser)


class Dataset_Basement(Dataset):
    def __init__(
        self,
        data_dir: str, data_parser: type[PARSER] = Parser,
        **parser_kwarg
    ) -> None:
        _parser = data_parser(data_dir, **parser_kwarg)
        self.data_info = _parser.data_info
        self.data_block = _parser.data_block

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index) -> Any:
        raise NotImplementedError


class Config():
    @dataclass
    class Dataset():
        name: str = "no_data"
        data_dir: str = "./datasets"

        def Build_dataset(self, mode: str) -> Dataset:
            raise NotImplementedError

    CONFIG_DATASET = TypeVar(
        "CONFIG_DATASET",
        bound=Dataset
    )

    @dataclass
    class Dataloader():
        dataset_config: Config.CONFIG_DATASET = field(default_factory=dict)

        batch_size: int = 1
        shuffle: bool = True
        num_workers: int = 0
        collate_fn: str | None = None
        drop_last: bool = False

        def _Build_collate_fn(self) -> Callable:
            raise NotImplementedError

        def Get_Dataloader_params(self) -> dict[str, Any]:
            _collate_fn = self._Build_collate_fn(
            ) if self.collate_fn is not None else None

            return {
                "batch_size": self.batch_size,
                "shuffle": self.shuffle,
                "num_workers": self.num_workers,
                "collate_fn": _collate_fn,
                "drop_last": self.drop_last

            }

    CONFIG_DATALOADER = TypeVar(
        "CONFIG_DATALOADER",
        bound=Dataloader
    )
