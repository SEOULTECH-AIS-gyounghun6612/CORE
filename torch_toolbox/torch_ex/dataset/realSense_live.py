from typing import Dict, Any
from .basement import __Basement__


class Realsense_Dataset(__Basement__):
    def __init__(
        self, root: str, category: str, trans_config: Dict[str, Any], **kwarg
    ) -> None:
        super().__init__(root, category, trans_config, **kwarg)

    def Make_datalist(self, data_root: str, data_category: str, **kwarg):
        ...

    def Make_transform(self, data_transform_config: Dict[str, Any]):
        return super().Make_transform(data_transform_config)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        ...
