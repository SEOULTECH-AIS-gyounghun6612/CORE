""" ###

------------------------------------------------------------------------
### Requirement
    -

### Structure
    -

"""
from __future__ import annotations
from typing import Any, Callable, Literal
from dataclasses import dataclass, field

from torch.utils.data import Dataset, DataLoader

from python_ex.project import Config


@dataclass
class Data_Config(Config.Basement):
    name: str = "no_data"
    process: str = "custom"
    data_dir: str = "./datasets"
    additional: dict = field(default_factory=dict)


class Custom_Dataset(Dataset):
    def __init__(
        self,
        mode: Literal["train", "validation", "test"], cfg: Data_Config
    ):
        self.dataset_cfg = cfg
        self.dataset_block = self.Get_data_block(
            mode, cfg.data_dir, cfg.process, **cfg.additional)

    def Get_data_block(
        self,
        mode: Literal["train", "validation", "test"],
        data_dir: str, process: str, **kwarg
    ) -> dict[int, Any]:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index) -> Any:
        raise NotImplementedError


@dataclass
class Dataloader_Config(Config.Basement):
    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 0
    drop_last: bool = False


def Build_loader(
    dataloader_cfg: Dataloader_Config,
    dataset: Dataset,
    collect_func: Callable | None = None
):
    return DataLoader(
        dataset, collate_fn=collect_func, **dataloader_cfg.Config_to_dict())
