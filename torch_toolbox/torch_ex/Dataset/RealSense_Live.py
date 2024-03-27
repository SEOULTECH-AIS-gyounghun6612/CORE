from typing import Dict, Any
from .Basement import __Basement__


class Realsense_Dataset(__Basement__):
    def __init__(self, data_root: str, data_category: str, data_transform_config: Dict[str, Any], **kwarg) -> None:
        super().__init__(data_root, data_category, data_transform_config, **kwarg)

    def Make_data_list(self, data_root: str, data_category: str, **kwarg):
        ...

    def Make_data_transform(self, data_transform_config: Dict[str, Any]):
        return super().Make_data_transform(data_transform_config)

    def __len__(self):
        return len(self.input_datas)

    def __getitem__(self, index):
        ...
