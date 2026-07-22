from __future__ import annotations
from typing import Any, Literal
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .... import CFGS
from ... import MODELS
from ...build import Module_Config_Template
from ..definition import Trainable_Model
from .utils.from_timm import Timm_Feature_Backbone, load_timm_backbone


MODEL_NAME = "convnext"
CONFIG_NAME = f"{MODEL_NAME}_Config"


_CONVNEXT_VARIANTS = {
    # ConvNeXt V1
    "convnext_atto": "convnext_atto.d2_in1k",
    "convnext_femto": "convnext_femto.d1_in1k",
    "convnext_pico": "convnext_pico.d1_in1k",
    "convnext_nano": "convnext_nano.in1k",
    "convnext_tiny": "convnext_tiny.in12k_ft_in1k",
    "convnext_small": "convnext_small.in12k_ft_in1k",
    "convnext_base": "convnext_base.in12k_ft_in1k",
    "convnext_large": "convnext_large.in12k_ft_in1k",
    "convnext_xlarge": "convnext_xlarge.in22k_ft_in1k",
    
    # ConvNeXt V2 (Self-supervised learning + Architectural upgrades)
    "convnextv2_atto": "convnextv2_atto.fcmae_ft_in1k",
    "convnextv2_femto": "convnextv2_femto.fcmae_ft_in1k",
    "convnextv2_pico": "convnextv2_pico.fcmae_ft_in1k",
    "convnextv2_nano": "convnextv2_nano.fcmae_ft_in1k",
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in1k",
    "convnextv2_base": "convnextv2_base.fcmae_ft_in1k",
    "convnextv2_large": "convnextv2_large.fcmae_ft_in1k",
    "convnextv2_huge": "convnextv2_huge.fcmae_ft_in1k",
}

ConvNeXtVariantType = Literal[
    "convnext_atto", "convnext_femto", "convnext_pico", "convnext_nano",
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large", "convnext_xlarge",
    "convnextv2_atto", "convnextv2_femto", "convnextv2_pico", "convnextv2_nano",
    "convnextv2_tiny", "convnextv2_base", "convnextv2_large", "convnextv2_huge"
]


@CFGS.Register_module(CONFIG_NAME)
@dataclass
class ConvNeXt_Config(Module_Config_Template):
    config_type: str = CONFIG_NAME
    object_type: str = MODEL_NAME
    trainable: bool = False
    
    variant: ConvNeXtVariantType = "convnextv2_base"
    out_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    timm_kwargs: dict[str, Any] = field(default_factory=dict)

@MODELS.Register_module(MODEL_NAME)
class ConvNeXt(Timm_Feature_Backbone, Trainable_Model):
    """
    timm 라이브러리를 기반으로 ConvNeXt 및 ConvNeXt V2 모델을 불러오는 직관적인 백본 래퍼입니다.
    """
    backbone: nn.Module
    
    def Build(
        self,
        variant: str,
        out_indices: list[int] | None = None,
        timm_kwargs: dict[str, Any] | None = None,
        **build_kwarg
    ):
        if variant not in _CONVNEXT_VARIANTS:
            raise ValueError(f"Unsupported ConvNeXt variant '{variant}'")

        self.backbone = load_timm_backbone(
            model_name=_CONVNEXT_VARIANTS[variant],
            out_indices=out_indices,
            **(timm_kwargs or {})
        )

    def forward(self, x: torch.Tensor, **kwarg) -> list[torch.Tensor]:
        return self.backbone(x)
