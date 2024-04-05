from __future__ import annotations
from typing import Dict, Any


class __Basement__():
    def __init__(self) -> None:
        pass

    def Resize(self):
        raise NotImplementedError

    def Config_to_compose(self, process_config: Dict[str, Dict[str, Any]]):
        raise NotImplementedError
