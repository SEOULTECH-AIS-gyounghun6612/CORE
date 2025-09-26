from __future__ import annotations
from typing import Any
from dataclasses import dataclass, field, InitVar
from pathlib import Path

from torch.nn import Module
from torch import load, save

from python_ex.project import Config


@dataclass
class Model_Config(Config.Basement):
    sub_model_arg: InitVar[list[dict[str, Any]]] | None = None

    model_name: str = "default_name"
    model_arg: dict[str, Any] = field(default_factory=dict)

    sub_model_cfg: list[Model_Config] = field(init=False)

    def __post_init__(self, sub_model_arg: list[dict[str, Any]] | None = None):
        self.sub_model_cfg = [] if sub_model_arg is None else [
            Model_Config(**_kwarg) for _kwarg in sub_model_arg
        ]

    def Config_to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_arg": self.model_arg,
            "sub_model_arg": [
                _cfg.Config_to_dict() for _cfg in self.sub_model_cfg]
        }


class Custom_Model(Module):
    def __init__(self, model_cfg: Model_Config) -> None:
        super().__init__()
        self.model_name = model_cfg.model_name

    def forward(self, **kwarg):
        raise NotImplementedError

    def Save_weight(self, file_dir: Path):
        save({"model": self.state_dict()}, file_dir / f"{self.model_name}.h5")

    def Load_weight(self, file_dir: Path):
        self.load_state_dict(load(file_dir / f"{self.model_name}.h5")["model"])
