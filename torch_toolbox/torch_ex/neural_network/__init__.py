from __future__ import annotations
from dataclasses import dataclass

from abc import ABC, abstractmethod
from torch.nn import Module
from torch import load, save, device

from python_ex.system import Path
from python_ex.project import Config


class Model_Basement(Module, ABC):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name

    @abstractmethod
    def forward(self, **kwarg):
        raise NotImplementedError

    def Save_weight(
        self,
        file_dir: str | None = None
    ):
        """ ### 학습 과정의 주요 인자값을 저장하는 함수

        ------------------------------------------------------------------
        ### Args
        - `file_dir`: 저장되는 파일이 위치할 디렉토리

        ### Returns
        - None

        ### Raises
        - None

        """
        save(
            {"model": self.state_dict()},
            Path.Join(f"{self.model_name}.h5", file_dir))

    def Load_weight(
        self,
        file_dir: str | None = None
    ):
        """ ### 이전 학습 과정에서 생성된 주요 인자값을 불러오는 함수

        ------------------------------------------------------------------
        ### Args
        - `file_dir`: 저장되어 있는 파일이 위치한 디렉토리

        ### Returns or Yields
        None

        ### Raises
        - None

        """
        _state_dict = load(Path.Join(f"{self.model_name}.h5", file_dir))

        self.load_state_dict(_state_dict["model"])


@dataclass
class Neural_Network_Config(ABC, Config):
    model_name: str = "default_name"

    @abstractmethod
    def Build_model_n_loss(
        self, device_info: device
    ) -> tuple[Model_Basement, list[Module]]:
        raise NotImplementedError

    def Get_summation(self):
        return [self.model_name]
