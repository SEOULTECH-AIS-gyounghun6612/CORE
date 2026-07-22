from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn

from .... import CFGS
from ... import LOSSES
from ...definition import Module_Config_Template, Composable_Module


CE_LOSS_NAME = "cross_entropy"
CE_CONFIG_NAME = f"{CE_LOSS_NAME}_Config"


@CFGS.Register_module(CE_CONFIG_NAME)
@dataclass
class Cls_CrossEntropy_Config(Module_Config_Template):
    """분류용 CrossEntropy Loss 설정.

    Attributes:
        label_smoothing: 라벨 스무딩 비율.
        ignore_index: loss 에서 제외할 타깃 인덱스. 예약 슬롯(미분류 등)처럼
            **출력 공간에는 자리가 있으나 학습 타깃은 아닌** 클래스에 쓴다.
            None 이면 torch 기본 sentinel(-100)을 써서 아무것도 제외하지 않는다
            (예약 슬롯이 없는 데이터셋에서 그대로 동작하게 하기 위함).
    """
    config_type: str = CE_CONFIG_NAME
    object_type: str = CE_LOSS_NAME

    label_smoothing: float = 0.0
    ignore_index: int | None = None


@LOSSES.Register_module(CE_LOSS_NAME)
class Cls_CrossEntropy(Composable_Module):
    """분류 로짓에 대한 CrossEntropy Loss."""

    def Build(
        self, label_smoothing: float = 0.0, ignore_index: int | None = None, **build_kwarg
    ) -> None:
        self.ce = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=-100 if ignore_index is None else int(ignore_index),
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, num_classes) 로짓.
            target: (N,) 클래스 인덱스.
        """
        return self.ce(pred, target)
