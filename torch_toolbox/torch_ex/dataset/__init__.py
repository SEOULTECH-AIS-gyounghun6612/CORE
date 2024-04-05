from typing import Dict, Any
import sys
import importlib

from .basement import __Basement__
from python_ex.system import Path


def Get_support_dataset():
    _, _this_path, this_file_name = Path.Get_file_directory(__file__)
    sys.path.append(_this_path)  # add this path for import
    _not_dataset = [this_file_name, "basement.py"]

    _datasets = []

    for _path in Path.Search(_this_path, ext_filter=".py"):
        _name = Path.Devide(_path)[-1]
        if _name not in _not_dataset:
            _datasets.append(_name)

    return _datasets


def Build(dataset_config: Dict[str, Any]):
    _support_datasets = Get_support_dataset()
    _dataset_name: str = dataset_config["dataset"]

    for _support_name in _support_datasets:
        if _dataset_name in _support_name:
            try:
                _dataset_module = importlib.import_module(_dataset_name)
            except ImportError:
                pass  # in later raise error in here

            _dataset: __Basement__ = _dataset_module.CustomDataset(
                **_dataset_name
            )

            return _dataset

    raise ValueError(f"dataset {_dataset_name} is not support")