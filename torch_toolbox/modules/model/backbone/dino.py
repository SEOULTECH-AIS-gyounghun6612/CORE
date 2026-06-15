from __future__ import annotations
from typing import Any, Literal
from dataclasses import dataclass, field

import timm
import torch.nn as nn

from ....registry import CFGS, MODELS
from ...build import Module_Config_Template
from ..definition import Trainable_Model


MODEL_NAME = "dino"
CONFIG_NAME = f"{MODEL_NAME}_Config"

_DINO_VARIANTS = {
    # DINOv2 (LVD-142M)
    "v2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "v2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "v2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "v2_vitg14": "vit_giant2_patch14_dinov2.lvd142m",

    # DINOv2 with registers
    "v2_vits14_reg": "vit_small_patch14_reg4_dinov2.lvd142m",
    "v2_vitb14_reg": "vit_base_patch14_reg4_dinov2.lvd142m",
    "v2_vitl14_reg": "vit_large_patch14_reg4_dinov2.lvd142m",
    "v2_vitg14_reg": "vit_giant2_patch14_reg4_dinov2.lvd142m",

    # DINOv3
    "v3_vitl16_sat": "vit_large_patch16_dinov3.sat493m",
    "v3_vithp16_lvd": "vit_huge_plus_patch16_dinov3.lvd1689m",
    "v3_vithp16_qkvb_lvd": "vit_huge_plus_patch16_dinov3_qkvb.lvd1689m",
    "v3_vit7b16_lvd": "vit_7b_patch16_dinov3.lvd1689m",
    "v3_vit7b16_sat": "vit_7b_patch16_dinov3.sat493m",
}

DinoVariantType = Literal[
    "v2_vits14", "v2_vitb14", "v2_vitl14", "v2_vitg14",
    "v2_vits14_reg", "v2_vitb14_reg", "v2_vitl14_reg", "v2_vitg14_reg",
    "v3_vitl16_sat", "v3_vithp16_lvd", "v3_vithp16_qkvb_lvd", "v3_vit7b16_lvd", "v3_vit7b16_sat"
]


@CFGS.Register_module(CONFIG_NAME)
@dataclass
class DINO_Config(Module_Config_Template):
    config_type: str = CONFIG_NAME
    object_type: str = MODEL_NAME
    trainable: bool = False

    variant: DinoVariantType = "v2_vits14"
    timm_kwargs: dict[str, Any] = field(default_factory=dict)
    # trainable=False일 때 전체 freeze 후 이 목록의 모듈만 reset + unfreeze
    trainable_modules: list[str] = field(default_factory=list)


@MODELS.Register_module(MODEL_NAME)
class DINO(Trainable_Model):
    """timm 라이브러리를 기반으로 DINO 모델을 불러오는 백본 래퍼."""

    backbone: nn.Module

    def __init__(
        self,
        name: str,
        trainable: bool = False,
        trainable_modules: list[str] | None = None,
        **build_kwarg,
    ) -> None:
        super().__init__(name, trainable, **build_kwarg)
        # Composable_Module이 전체 freeze를 적용한 뒤 지정 모듈만 reset + unfreeze
        if not trainable and trainable_modules:
            for mod_name, module in self.backbone.named_modules():
                for prefix in trainable_modules:
                    if mod_name == prefix or mod_name.startswith(f"{prefix}."):
                        if hasattr(module, "reset_parameters"):
                            module.reset_parameters()
                        for param in module.parameters(recurse=False):
                            param.requires_grad_(True)
                        break

    def Build(
        self,
        variant: str,
        timm_kwargs: dict[str, Any] | None = None,
        **build_kwarg
    ) -> None:
        if variant not in _DINO_VARIANTS:
            raise ValueError(f"Unsupported DINO variant '{variant}'")

        self.backbone = timm.create_model(
            _DINO_VARIANTS[variant],
            num_classes=0,
            **(timm_kwargs or {})
        )

    def forward(self, x, **kwarg):
        tokens = self.backbone.forward_features(x)      # (B, N+1, D)
        patches = tokens[:, 1:]                          # CLS 제거 → (B, N, D)
        B, N, D = patches.shape
        H = W = int(N ** 0.5)
        spatial = patches.permute(0, 2, 1).reshape(B, D, H, W)  # (B, D, H, W)
        return [spatial]
