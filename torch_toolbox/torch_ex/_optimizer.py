from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Tuple
from python_ex._base import Utils

# optimizer
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler


if __package__ == "":
    # if this file in local project
    from torch_ex._layer import Custom_Module
else:
    # if this file in package folder
    from ._layer import Custom_Module


# -- DEFINE CONSTNAT -- #
class Suported_Optimizer(Enum):
    Adam = 0


class Suported_Schedule(Enum):
    Adam = 0


# -- DEFINE CONFIG -- #
@dataclass
class Scheduler_Config(Utils.Config):
    _Optim_name: Suported_Optimizer

    _Schedule_name: Suported_Schedule

    _LR_initail: float = 0.005

    def _convert_to_dict(self) -> Dict[str, Any]:
        return super()._convert_to_dict()

    def _restore_from_dict(self, data: Dict[str, Any]):
        return super()._restore_from_dict(data)

    def _make_schedule(self, model: Custom_Module) -> Tuple[Optimizer, _LRScheduler]:
        # make optim
        if self._Optim_name == Suported_Optimizer.Adam:
            __optim = Adam(model.parameters(), self._LR_initail)

        # make schedule
        if self._Schedule_name == Suported_Schedule.Adam:
            __schedule = None

        return __optim, __schedule
