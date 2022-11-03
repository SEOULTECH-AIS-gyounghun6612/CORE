from dataclasses import dataclass
from enum import Enum
from math import cos, pi
from typing import Dict, List, Union, Any
from python_ex._base import Utils

# optimizer
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler


if __package__ == "":
    # if this file in local project
    from torch_ex._layer import Custom_Model
else:
    # if this file in package folder
    from ._layer import Custom_Model


# -- DEFINE CONSTNAT -- #
class Suported_Optimizer(Enum):
    Adam = "Adam"


class Suported_Schedule(Enum):
    Cosin_Annealing = "Cosin_Annealing"


# -- DEFINE CONFIG -- #
@dataclass
class Scheduler_Config(Utils.Config):
    _Optim_name: Suported_Optimizer
    _Schedule_name: Suported_Schedule

    _LR_Maximum: float = 0.005
    _LR_Minimum: float = 0.0001
    _LR_Decay: float = 1.0

    _Term: Union[List[int], int] = 50
    _Term_amp: float = 1.0

    def _get_parameter(self, model: Custom_Model) -> Dict[str, Any]:
        return {
            "optimizer": optim.__dict__[self._Optim_name.value](model.parameters(), self._LR_Maximum),
            "schedule_name": self._Schedule_name,
            "term": self._Term,
            "term_amp": self._Term_amp,
            "maximum": self._LR_Maximum,
            "minimum": self._LR_Minimum,
            "decay": self._LR_Decay}

    def _convert_to_dict(self) -> Dict[str, Union[Dict, str, int, float, bool, None]]:
        return {
            "_Optim_name": self._Optim_name.value,
            "_Schedule_name": self._Schedule_name.value,
            "_LR_Maximum": self._LR_Maximum,
            "_LR_Minimum": self._LR_Minimum,
            "_LR_Decay": self._LR_Decay,
            "_Term": self._Term,
            "_Term_amp": self._Term_amp}

    def _restore_from_dict(self, data: Dict[str, Union[Dict, str, int, float, bool, None]]):
        self._Optim_name = Suported_Optimizer(data["_Optim_name"])
        self._Schedule_name = Suported_Schedule(data["_Schedule_name"])
        self._LR_Maximum = data["_LR_Maximum"]
        self._LR_Minimum = data["_LR_Minimum"]
        self._LR_Decay = data["_LR_Decay"]
        self._Term = data["_Term"]
        self._Term_amp = data["_Term_amp"]


# -- Mation Function -- #
class Custom_Scheduler():
    class Base(_LRScheduler):
        def __init__(
                self, optimizer: optim.Optimizer,
                term: Union[List[int], int], term_amp: float,
                maximum: float, minimum: float, decay: float,
                last_epoch: int = -1) -> None:

            self._Cycle: int = 0
            self._Term = term
            self._Term_amp = term_amp

            self._Maximum = maximum
            self._Minimum = minimum
            self._Decay = decay

            self._This_count: int = last_epoch
            self._This_term: int = self._get_next_term()

            super().__init__(optimizer, last_epoch)

        # Freeze function
        def _get_next_term(self):
            if isinstance(self._Term, list):
                return self._Term[-1] if self._Cycle >= len(self._Term) else self._Term[self._Cycle]
            else:
                return round(self._Term * (self._Term_amp ** self._Cycle))

        def step(self, epoch: int = None):
            if epoch is None:  # go to next epoch
                self.last_epoch = self.last_epoch + 1
                self._This_count = self._This_count + 1

                if self._This_count >= self._This_term:
                    self._This_count = 0
                    self._Cycle += 1
                    self._This_term = self._get_next_term()

            else:  # restore session
                while epoch >= self._This_term:
                    epoch = epoch - self._This_term
                    self._Cycle += 1
                    self._This_term = self._get_next_term()

                self._This_count = epoch

            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        # Un-Freeze function
        def get_lr(self):
            return [self._Maximum for _ in self.base_lrs]

    class Cosin_Annealing(Base):
        def get_lr(self):
            _amp = (1 + cos(pi * (self._This_count) / (self._This_term))) / 2
            _value = self._Minimum + (self._Maximum - self._Minimum) * _amp
            return [_value for _ in self.base_lrs]

    @staticmethod
    def _build(
            optimizer: optim.Optimizer, schedule_name: Suported_Schedule,
            term: Union[List[int], int], term_amp: float,
            maximum: float, minimum: float, decay: float,
            last_epoch: int = -1) -> _LRScheduler:

        return optimizer, Custom_Scheduler.__dict__[schedule_name.value](optimizer, term, term_amp, maximum, minimum, decay, last_epoch)
