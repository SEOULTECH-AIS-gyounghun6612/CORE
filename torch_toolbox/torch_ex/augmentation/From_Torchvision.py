from typing import Dict, Any

try:
    from torchvision.transforms import Compose
except Exception:
    raise ValueError("This augment process, that from Torchvision, is not suport in this env. Please change the augment process")

from .Basement import __Basement__


class FromTorchvision(__Basement__):
    def Resize(self):
        return super().Resize()

    def Config_to_compose(self, process_config: Dict[str, Dict[str, Any]]):
        return Compose([self.__class__.__dict__[_process_name](**_kwarg) for _process_name, _kwarg in process_config.items()])
