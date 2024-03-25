from __future__ import annotations
from typing import Tuple, Dict, List, Any

from torch.utils.data import Dataset


class __Basement__(Dataset):
    def __init__(self, data_root: str, data_category: str, data_transform_config: Dict[str, Any], **kwarg) -> None:
        self.input_datas, self.target_datas = self.Make_data_list(data_root, data_category, **kwarg)
        self.transform = self.Make_data_transform(data_transform_config)

    def __len__(self):
        return len(self.input_datas)

    def Make_data_list(self, data_root: str, data_category: str, **kwarg) -> Tuple[List, List]:
        raise NotImplementedError

    def Make_data_transform(self, data_transform_config: Dict[str, Any]):
        raise NotImplementedError
