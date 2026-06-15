from __future__ import annotations
from typing import Any, Literal
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .... import CFGS
from ... import MODELS
from ...build import Module_Config_Template
from ..definition import Trainable_Model
from .utils.from_timm import load_timm_backbone


MODEL_NAME = "resnet"
CONFIG_NAME = f"{MODEL_NAME}_Config"


# ResNet 버전에 따른 timm 모델명 매핑 딕셔너리
_RESNET_VARIANTS = {
    "resnet18": "resnet18.tv_in1k",
    "resnet34": "resnet34.tv_in1k",
    "resnet50": "resnet50.tv_in1k",
    "resnet101": "resnet101.tv_in1k",
    
    # 향상된 버전 (ResNet strikes back)
    "resnet50_v15": "resnet50.a1_in1k",
    "resnet101_v15": "resnet101.a1_in1k",
}

ResNetVariantType = Literal[
    "resnet18", "resnet34", "resnet50", "resnet101",
    "resnet50_v15", "resnet101_v15"
]


@CFGS.Register_module(CONFIG_NAME)
@dataclass
class ResNet_Config(Module_Config_Template):
    config_type: str = CONFIG_NAME
    object_type: str = MODEL_NAME
    trainable: bool = False
    
    variant: ResNetVariantType = "resnet50"
    out_indices: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    timm_kwargs: dict[str, Any] = field(default_factory=dict)


@MODELS.Register_module(MODEL_NAME)
class ResNet(Trainable_Model):
    """
    timm 라이브러리를 기반으로 ResNet 계열 모델을 불러오는 직관적인 백본 래퍼입니다.
    """
    backbone: nn.Module
    
    def Build(
        self,
        variant: str,
        out_indices: list[int] | None = None,
        timm_kwargs: dict[str, Any] | None = None,
        **build_kwarg
    ):
        if variant not in _RESNET_VARIANTS:
            raise ValueError(f"Unsupported ResNet variant '{variant}'")

        self.backbone = load_timm_backbone(
            model_name=_RESNET_VARIANTS[variant],
            out_indices=out_indices,
            **(timm_kwargs or {})
        )

    def forward(self, x: torch.Tensor, **kwarg) -> list[torch.Tensor]:
        return self.backbone(x)
