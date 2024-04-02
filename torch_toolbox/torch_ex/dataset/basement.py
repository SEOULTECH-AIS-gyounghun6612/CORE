from __future__ import annotations
from typing import Tuple, Dict, List, Any

from torch.utils.data import Dataset


class __Basement__(Dataset):
    def __init__(
        self, root: str, category: str, trans_config: Dict[str, Any], **kwarg
    ) -> None:
        self.inputs, self.targets = self.Make_datalist(root, category, **kwarg)
        self.transform = self.Make_transform(trans_config)

    def __len__(self):
        return len(self.inputs)

    def Make_datalist(
        self, root: str, category: str, **kwarg
    ) -> Tuple[List, List]:
        raise NotImplementedError

    def Make_transform(self, trans_config: Dict[str, Any]):
        raise NotImplementedError
