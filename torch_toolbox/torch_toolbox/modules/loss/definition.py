from __future__ import annotations
from typing import Any
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import ModuleDict

from ... import CFGS
from .. import LOSSES
from ..definition import Composable_Config, Composable_Module


LOSS_NAME = "assemble_loss"
CONFIG_NAME = f"{LOSS_NAME}_config"


@CFGS.Register_module(CONFIG_NAME)
@dataclass
class Assemble_Loss_Config(Composable_Config):
    """여러 Loss를 가중합으로 조합하기 위한 설정.

    Attributes:
        sub_loss_coefs: 서브 Loss 이름 → 가중치 계수 매핑.
            미지정 항목은 forward 시 1.0으로 자동 할당된다.
    """

    config_type: str = CONFIG_NAME
    object_type: str = LOSS_NAME

    sub_loss_coefs: dict[str, float] = field(default_factory=dict)


@LOSSES.Register_module(LOSS_NAME)
class Assemble_Loss(Composable_Module):
    """등록된 여러 Loss를 가중합으로 결합하는 컴포저블 모듈.

    pred에 존재하는 키만 계산에 참여한다. pred에 없는 서브 Loss는 silently skip되며,
    pred에 있지만 target에 없으면 KeyError를 발생시킨다.
    """

    def Build(
        self,
        sub_loss_coefs: dict[str, float],
        **sub_modules: Any
    ):
        """서브 Loss 모듈과 가중치를 초기화한다.

        ModuleDict에 등록해 PyTorch가 파라미터를 추적하도록 하고,
        _cached_func에 (coef, module) 쌍을 저장해 forward에서 빠르게 접근한다.

        Args:
            sub_loss_coefs: 서브 Loss 이름 → 가중치 계수.
            **sub_modules: __Build_from_registry__가 주입한 서브 Loss 인스턴스.
        """
        self.loss_modules = ModuleDict()
        self._cached_func: dict[str, tuple[float, Composable_Module]] = {}

        for _name, _module in sub_modules.items():
            _coef = sub_loss_coefs.get(_name, 1.0)
            # ModuleDict: PyTorch 파라미터 추적용 / _cached_func: forward 빠른 접근용
            self.loss_modules[_name] = _module
            self._cached_func[_name] = (_coef, _module)

    def forward(
        self, pred: dict[str, Tensor], target: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, float]]:
        """가중합 loss를 계산한다.

        Args:
            pred: 모델 출력 딕셔너리.
            target: 정답 딕셔너리.

        Returns:
            tuple:
                - total_loss: 전체 가중합 loss 텐서.
                - loss_details: 서브 Loss별 raw 값과 가중 적용 값.
                  키 형식: ``{name}`` (raw), ``{name}_weighted`` (가중치 적용).

        Raises:
            KeyError: pred에 존재하는 키가 target에 없는 경우.
        """
        _loss_details: dict[str, float] = {}
        _device = next(iter(pred.values())).device
        _total_loss = torch.tensor(0.0, device=_device)

        for _k, (_coef, _func) in self._cached_func.items():
            if _k not in pred:
                # pred에 없는 출력 키는 해당 Loss 계산을 건너뜀
                continue
            if _k not in target:
                raise KeyError(
                    f"Assemble_Loss: '{_k}'가 target에 없음. "
                    f"target keys: {list(target.keys())}"
                )

            _raw_loss: Tensor = _func(pred[_k], target[_k])
            _weighted_loss = _coef * _raw_loss
            _total_loss = _total_loss + _weighted_loss

            _loss_details[_k] = _raw_loss.item()
            _loss_details[f"{_k}_weighted"] = _weighted_loss.item()

        return _total_loss, _loss_details
