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


MODEL_NAME = "efficientnet"
CONFIG_NAME = f"{MODEL_NAME}_Config"


_EFFICIENT_NET_VARIANTS = {
    # EfficientNet (B0-B7)
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "efficientnet_b1": "efficientnet_b1.ft_in1k",
    "efficientnet_b2": "efficientnet_b2.ra_in1k",
    "efficientnet_b3": "efficientnet_b3.ra2_in1k",
    "efficientnet_b4": "efficientnet_b4.ra2_in1k",
    "efficientnet_b5": "efficientnet_b5.sw_in12k_ft_in1k",
    "efficientnet_b6": "efficientnet_b6.rw_weight_in1k",
    "efficientnet_b7": "efficientnet_b7.ra_in1k",
    
    # EfficientNet V2
    "efficientnetv2_s": "efficientnetv2_rw_s.ra2_in1k",
    "efficientnetv2_m": "efficientnetv2_rw_m.agc_in1k",
    "efficientnetv2_l": "efficientnetv2_rw_t.ra2_in1k", # timm uses rw_t, etc for variations
    "efficientnetv2_rw_t": "efficientnetv2_rw_t.ra2_in1k",
}

EfficientNetVariantType = Literal[
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    "efficientnetv2_s", "efficientnetv2_m", "efficientnetv2_l", "efficientnetv2_rw_t"
]


@CFGS.Register_module(CONFIG_NAME)
@dataclass
class EfficientNet_Config(Module_Config_Template):
    config_type: str = CONFIG_NAME
    object_type: str = MODEL_NAME
    trainable: bool = False
    
    variant: EfficientNetVariantType = "efficientnet_b0"
    out_indices: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    timm_kwargs: dict[str, Any] = field(default_factory=dict)


@MODELS.Register_module(MODEL_NAME)
class EfficientNet(Timm_Feature_Backbone, Trainable_Model):
    """
    timm 라이브러리를 기반으로 경량화/고효율의 EfficientNet 및 EfficientNet V2 모델을 불러오는 래퍼입니다.
    """
    backbone: nn.Module
    
    def Build(
        self,
        variant: str,
        out_indices: list[int] | None = None,
        timm_kwargs: dict[str, Any] | None = None,
        **build_kwarg
    ):
        if variant not in _EFFICIENT_NET_VARIANTS:
            raise ValueError(f"Unsupported EfficientNet variant '{variant}'")

        self.backbone = load_timm_backbone(
            model_name=_EFFICIENT_NET_VARIANTS[variant],
            out_indices=out_indices,
            **(timm_kwargs or {})
        )

    def forward(self, x: torch.Tensor, **kwarg) -> list[torch.Tensor]:
        return self.backbone(x)
