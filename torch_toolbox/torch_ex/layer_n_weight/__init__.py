from __future__ import annotations
from typing import (
    Generic, TypeVar, Callable
)
from dataclasses import dataclass

from abc import ABC, abstractmethod
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


class Model_Basement(Module, ABC, Generic[Config.CONFIG_MODEL_N_LOSS]):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError

    @abstractmethod
    def forward(self):
        raise NotImplementedError
