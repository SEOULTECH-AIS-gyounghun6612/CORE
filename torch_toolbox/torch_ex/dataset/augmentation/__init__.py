from typing import Dict, Any
import sys
import importlib

from python_ex.system import Path


def Get_support_augmentation():
    _, _this_path, this_file_name = Path.Get_file_directory(__file__)
    sys.path.append(_this_path)  # add this path for import
    _not_dataset = [this_file_name, "basement.py"]

    _datasets = []

    for _path in Path.Search(_this_path, ext_filter=".py"):
        _name = Path.Devide(_path)[-1]
        if _name not in _not_dataset:
            _datasets.append(_name)

    return _datasets


def Build(augment_name: str, process_config: Dict[str, Dict[str, Any]]):
    if augment_name.lower() == "torchvision":
        from .from_torchvision import FromTorchvision
        return FromTorchvision().Config_to_compose(process_config)

    _error_text = f"This augment process, that from {augment_name.lower()}, is not suport in this module version.\n"
    _error_text = f"{_error_text}Please change the augment process or module version\n"
    raise ValueError(_error_text)
