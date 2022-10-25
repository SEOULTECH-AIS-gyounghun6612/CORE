from dataclasses import dataclass
from enum import Enum
from math import cos, pi
from typing import Dict, Any, List, Union
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
    Cosin_Annealing = 0


# -- DEFINE CONFIG -- #
@dataclass
class Optimizer_Config(Utils.Config):
    _Optim_name: Suported_Optimizer
    _LR_rate: float = 0.005

    def _convert_to_dict(self) -> Dict[str, Any]:
        return super()._convert_to_dict()

    def _restore_from_dict(self, data: Dict[str, Any]):
        return super()._restore_from_dict(data)

    def _make_optim(self, model: Custom_Module.Model) -> _LRScheduler:
        # make optim
        if self._Optim_name == Suported_Optimizer.Adam:
            __optim = Adam(model.parameters(), self._LR_rate)

        return __optim


@dataclass
class Scheduler_Config(Utils.Config):
    _Schedule_name: Suported_Schedule

    _Maximum: float = 0.005
    _Minimum: float = 0.0001
    _Decay: float = 1.0

    _Term: Union[List[int], int] = 50
    _Term_amp: int = 1

    def _convert_to_dict(self) -> Dict[str, Any]:
        return super()._convert_to_dict()

    def _restore_from_dict(self, data: Dict[str, Any]):
        return super()._restore_from_dict(data)

    def _get_next_term(self, cycle: int = 0):
        return self._Term[cycle] if isinstance(self._Term, list) else (self._Term * (self._Term_amp ** cycle))


# -- Mation Function -- #
class Custom_Scheduler():
    class Base(_LRScheduler):
        def __init__(self, config: Scheduler_Config, optimizer: Optimizer, last_epoch: int = -1) -> None:
            self._Option = config

            self._Cycle: int = 0
            self._This_count: int = last_epoch
            self._This_term: int = self._Option._get_next_term()

            super().__init__(optimizer, last_epoch)

        def step(self, epoch: int = None):
            if epoch is None:  # go to next epoch
                self.last_epoch = self.last_epoch + 1
                self._This_count = self._This_count + 1

                if self._This_count >= self._This_term:
                    self._This_count = 0
                    self._Cycle += 1
                    self._This_term = self._Option._get_next_term(self._Cycle)

            else:  # restore session
                while self._This_count >= self._This_term:
                    self._This_count = self._This_count - self._This_term
                    self._Cycle += 1
                    self._This_term = self._Option._get_next_term(self._Cycle)

            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

    class Cosin_Annealing_Schedule(Base):
        def __init__(self, config: Scheduler_Config, optimizer: Optimizer, last_epoch: int = -1) -> None:
            super().__init__(config, optimizer, last_epoch)

        def get_lr(self):
            __amp = (1 + cos(pi * (self._This_count) / (self._This_term))) / 2
            return [
                self._Option._Minimum + (self._Option._Maximum - self._Option._Minimum) * __amp for _ in self.base_lrs]

    @staticmethod
    def build(config: Scheduler_Config, optimizer: Optimizer, last_epoch: int = -1):
        if config._Schedule_name == Suported_Schedule.Cosin_Annealing:
            return Custom_Scheduler.Cosin_Annealing_Schedule(config, optimizer, last_epoch)
        else:
            return None
