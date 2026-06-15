from typing import Callable
from enum import StrEnum

from torch import Tensor


FUNC_LOSS = Callable[
    [
        dict[str, Tensor], dict[str, Tensor]
    ],
    Tensor | tuple[Tensor, dict[str, Tensor]]
]


class Mode(StrEnum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"
