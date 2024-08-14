""" ### 데이터 증폭 처리 모듈

------------------------------------------------------------------------
### Requirement
    - python_ex

### Structure
    - Get_support_augmentation:
    - Build:

"""
from typing import Dict, Any
import sys
# import importlib

from python_ex.system import Path


def Get_support_augmentation():
    """ ###

    ------------------------------------------------------------------
    ### Args
    - None

    ### Returns
    - `Dataset`: 학습에 사용하고자 하는 데이터 셋

    ### Raises
    - None

    """
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
    """ ###

    ------------------------------------------------------------------
    ### Args
    - `augment_name`:
    - `process_config`:

    ### Returns
    - `Dataset`: 학습에 사용하고자 하는 데이터 셋

    ### Raises
    - None

    """
    if augment_name.lower() == "torchvision":
        from .by_torchvision import FromTorchvision
        return FromTorchvision().Config_to_compose(process_config)

    _error_text = "\n".join(
        [
            ", ".join([
                "This augment process",
                f"that from {augment_name.lower()}",
                "is not suport in this module version."
            ]),
            "Please change the augment process or module version"
        ]
    )

    raise ValueError(_error_text)
