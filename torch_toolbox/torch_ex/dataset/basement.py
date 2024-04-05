from __future__ import annotations
from typing import Tuple, Dict, List, Any

from torch.utils.data import Dataset
from .augmentation import Build


class __Basement__(Dataset):
    def __init__(
        self,
        root: str,
        dataset_name: str, category: str,
        transform_config: Dict[str, Any],
        **kwarg
    ) -> None:
        self.dataset_name = dataset_name
        self.inputs, self.targets = self.Make_datalist(root, category, **kwarg)
        self.transform = Build(**transform_config)

    def __len__(self):
        return len(self.inputs)

    def Make_datalist(
        self, root: str, category: str, **kwarg
    ) -> Tuple[List, List]:
        raise NotImplementedError
