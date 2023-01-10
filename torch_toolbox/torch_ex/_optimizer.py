from enum import Enum
from math import cos, pi
from typing import List, Union, Optional, Tuple

# optimizer
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import Module

if __package__ == "":
    # if this file in local project
    ...
else:
    # if this file in package folder
    ...


# -- DEFINE CONSTNAT -- #
class Suport_Optimizer(Enum):
    Adam = "Adam"


class Suport_Schedule(Enum):
    Cosin_Annealing = "Cosin_Annealing"


# -- Mation Function -- #
class Scheduler():
    class Basement(_LRScheduler):
        def __init__(
                self,
                optimizer: optim.Optimizer,
                term: Union[List[int], int],
                term_amp: float,
                maximum: float,
                minimum: float,
                decay: float,
                last_epoch: int = -1) -> None:
            self._Cycle: int = 0
            self._Term = term  # int -> fixed term list[int] -> milestone
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

        def step(self, epoch: Optional[int] = None):
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

            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):  # type: ignore
                param_group['lr'] = lr

        # Un-Freeze function
        def get_lr(self):
            return [self._Maximum for _ in self.base_lrs]  # type: ignore

    class Cosin_Annealing(Basement):
        def get_lr(self):
            _amp = (1 + cos(pi * (self._This_count) / (self._This_term))) / 2
            _value = self._Minimum + (self._Maximum - self._Minimum) * _amp
            return [_value for _ in self.base_lrs]  # type: ignore


def _Optimizer_build(
        optim_name: Suport_Optimizer,
        model: Module,
        initial_lr: float,
        schedule_name: Optional[Suport_Schedule],
        last_epoch: int = -1,
        **additional_parameter) -> Tuple[optim.Optimizer, Optional[_LRScheduler]]:

    _optim = optim.__dict__[optim_name.value](model.parameters(), initial_lr)
    _scheduler = Scheduler.__dict__[schedule_name.value](_optim, last_epoch=last_epoch, **additional_parameter) if schedule_name is not None else None

    return _optim, _scheduler
