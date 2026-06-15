from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn

from ....registry import CFGS, LOSSES
from ...definition import Module_Config_Template, Composable_Module


CE_LOSS_NAME = "cross_entropy"
CE_CONFIG_NAME = f"{CE_LOSS_NAME}_Config"


@CFGS.Register_module(CE_CONFIG_NAME)
@dataclass
class Cls_CrossEntropy_Config(Module_Config_Template):
    """분류용 CrossEntropy Loss 설정.

    Attributes:
        label_smoothing: 라벨 스무딩 비율.
    """
    config_type: str = CE_CONFIG_NAME
    object_type: str = CE_LOSS_NAME

    label_smoothing: float = 0.0


@LOSSES.Register_module(CE_LOSS_NAME)
class Cls_CrossEntropy(Composable_Module):
    """분류 로짓에 대한 CrossEntropy Loss."""

    def Build(self, label_smoothing: float = 0.0, **build_kwarg) -> None:
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, num_classes) 로짓.
            target: (N,) 클래스 인덱스.
        """
        return self.ce(pred, target)
