from typing import Any, Callable

from python_toolbox.registry import Registry

from .definition import Dataset_Config, Dataloader_Config, Custom_Dataset

DATASETS = Registry[type[Custom_Dataset]]("dataset", Custom_Dataset)
DATALOADER_FN = Registry[Callable[[Any], Any]]("dataloader_collect_fn", Callable[[Any], Any])
