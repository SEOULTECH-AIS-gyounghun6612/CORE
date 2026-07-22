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


MODEL_NAME = "swin"
CONFIG_NAME = f"{MODEL_NAME}_Config"


_SWIN_VARIANTS = {
    # Swin Transformer V1
    "swin_tiny_224": "swin_tiny_patch4_window7_224.ms_in1k",
    "swin_small_224": "swin_small_patch4_window7_224.ms_in1k",
    "swin_base_224": "swin_base_patch4_window7_224.ms_in1k",
    "swin_base_384": "swin_base_patch4_window12_384.ms_in1k",
    "swin_large_224": "swin_large_patch4_window7_224.ms_in22k_in1k",
    "swin_large_384": "swin_large_patch4_window12_384.ms_in22k_in1k",
    
    # Swin Transformer V2
    "swinv2_tiny_256": "swinv2_tiny_window16_256.ms_in1k",
    "swinv2_small_256": "swinv2_small_window16_256.ms_in1k",
    "swinv2_base_256": "swinv2_base_window16_256.ms_in1k",
    "swinv2_large_256": "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
}

SwinVariantType = Literal[
    "swin_tiny_224", "swin_small_224", "swin_base_224", "swin_base_384", "swin_large_224", "swin_large_384",
    "swinv2_tiny_256", "swinv2_small_256", "swinv2_base_256", "swinv2_large_256"
]


@CFGS.Register_module(CONFIG_NAME)
@dataclass
class Swin_Config(Module_Config_Template):
    config_type: str = CONFIG_NAME
    object_type: str = MODEL_NAME
    trainable: bool = False

    variant: SwinVariantType = "swinv2_base_256"
    out_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    timm_kwargs: dict[str, Any] = field(default_factory=dict)

@MODELS.Register_module(MODEL_NAME)
class Swin(Timm_Feature_Backbone, Trainable_Model):
    """
    timm 라이브러리를 기반으로 Swin Transformer V1/V2 모델을 불러오는 직관적인 백본 래퍼입니다.
    """
    backbone: nn.Module
    
    def Build(
        self,
        variant: str,
        out_indices: list[int] | None = None,
        timm_kwargs: dict[str, Any] | None = None,
        **build_kwarg
    ):
        if variant not in _SWIN_VARIANTS:
            raise ValueError(f"Unsupported Swin variant '{variant}'")

        self.backbone = load_timm_backbone(
            model_name=_SWIN_VARIANTS[variant],
            out_indices=out_indices,
            **(timm_kwargs or {})
        )

    def forward(self, x: torch.Tensor, **kwarg) -> list[torch.Tensor]:
        return self.backbone(x)
