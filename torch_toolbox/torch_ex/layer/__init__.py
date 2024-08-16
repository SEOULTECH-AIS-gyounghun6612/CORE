from __future__ import annotations
from typing import (
    TypeVar, Callable
)
from dataclasses import dataclass
from torch.nn import Module


class Config():
    @dataclass
    class Model_n_Loss():
        def Build_model_n_loss(self) -> tuple[Module, Callable]:
            raise NotImplementedError

    CONFIG_MODEL_N_LOSS = TypeVar(
        "CONFIG_MODEL_N_LOSS",
        bound=Model_n_Loss
    )
