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


MODEL_NAME = "clip_vision"
CONFIG_NAME = f"{MODEL_NAME}_Config"


_CLIP_VARIANTS = {
    # OpenAI CLIP Vision Encoders (ViT)
    "clip_vit_base_patch32": "vit_base_patch32_clip_224.openai",
    "clip_vit_base_patch16": "vit_base_patch16_clip_224.openai",
    "clip_vit_large_patch14": "vit_large_patch14_clip_224.openai",
    "clip_vit_large_patch14_336": "vit_large_patch14_clip_336.openai",
    
    # OpenAI CLIP Vision Encoders (ResNet)
    "clip_resnet50": "resnet50_clip_224.openai",
    "clip_resnet101": "resnet101_clip_224.openai",
    "clip_resnet50x4": "resnet50x4_clip_224.openai",
    
    # LAION CLIP Vision Encoders (OpenCLIP)
    "laion_vit_base_patch32": "vit_base_patch32_clip_224.laion2b",
    "laion_vit_huge_patch14": "vit_huge_patch14_clip_224.laion2b",
}

ClipVariantType = Literal[
    "clip_vit_base_patch32", "clip_vit_base_patch16", "clip_vit_large_patch14", "clip_vit_large_patch14_336",
    "clip_resnet50", "clip_resnet101", "clip_resnet50x4",
    "laion_vit_base_patch32", "laion_vit_huge_patch14"
]


@CFGS.Register_module(CONFIG_NAME)
@dataclass
class CLIP_Vision_Config(Module_Config_Template):
    config_type: str = CONFIG_NAME
    object_type: str = MODEL_NAME
    trainable: bool = False

    variant: ClipVariantType = "clip_vit_base_patch16"
    out_indices: list[int] = field(default_factory=lambda: [-1])
    timm_kwargs: dict[str, Any] = field(default_factory=dict)


@MODELS.Register_module(MODEL_NAME)
class CLIP_Vision(Trainable_Model):
    """
    timm 라이브러리를 기반으로 CLIP 모델의 Vision Encoder 부분만 불러오는 래퍼입니다.
    Zero-shot 분류나 멀티모달 연구의 이미지 백본으로 활용하기 좋습니다.
    """
    backbone: nn.Module
    
    def Build(
        self,
        variant: str,
        out_indices: list[int] | None = None,
        timm_kwargs: dict[str, Any] | None = None,
        **build_kwarg
    ):
        if variant not in _CLIP_VARIANTS:
            raise ValueError(f"Unsupported CLIP Vision variant '{variant}'")

        self.backbone = load_timm_backbone(
            model_name=_CLIP_VARIANTS[variant],
            out_indices=out_indices,
            **(timm_kwargs or {})
        )

    def forward(self, x: torch.Tensor, **kwarg) -> list[torch.Tensor]:
        return self.backbone(x)
