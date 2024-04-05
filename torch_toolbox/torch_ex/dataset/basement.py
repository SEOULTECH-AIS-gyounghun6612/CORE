from __future__ import annotations
from typing import Tuple, Dict, List, Any

from torch.utils.data import Dataset

from python_ex.system import Path

from .augmentation import Build


class __Basement__(Dataset):
    def __init__(
        self,
        root: str,
        dataset: str, category: str, mode: str,
        transform_config: Dict[str, Any],
        **kwarg
    ) -> None:
        self.dataset = dataset
        _root = Path.Join([dataset, category], root)

        self.inputs, self.targets = self.Make_datalist(
            _root,
            mode,
            **kwarg
        )
        self.transform = Build(**transform_config)

    def __len__(self):
        return len(self.inputs)

    def Make_datalist(
        self, root: str, mode: str, **kwarg
    ) -> Tuple[List, List]:
        raise NotImplementedError
