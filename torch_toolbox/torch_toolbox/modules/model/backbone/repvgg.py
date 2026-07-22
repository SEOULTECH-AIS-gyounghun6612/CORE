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

MODEL_NAME = "repvgg"
CONFIG_NAME = f"{MODEL_NAME}_Config"

# 10ms 방어를 위한 RepVGG 경량 변형 모델 매핑
_REPVGG_VARIANTS = {
    "repvgg_a0": "repvgg_a0",
    "repvgg_a1": "repvgg_a1",
    "repvgg_a2": "repvgg_a2",
}

RepVGGVariantType = Literal["repvgg_a0", "repvgg_a1", "repvgg_a2"]

@CFGS.Register_module(CONFIG_NAME)
@dataclass
class RepVGG_Config(Module_Config_Template):
    config_type: str = CONFIG_NAME
    object_type: str = MODEL_NAME
    trainable: bool = False
    
    variant: RepVGGVariantType = "repvgg_a0"
    # 해상도를 줄여 Context를 얻기 위한 Feature Level (1/4, 1/8, 1/16)
    out_indices: list[int] = field(default_factory=lambda: [1, 2, 3])
    timm_kwargs: dict[str, Any] = field(default_factory=dict)

@MODELS.Register_module(MODEL_NAME)
class RepVGG(Timm_Feature_Backbone, Trainable_Model):
    """
    timm 기반 RepVGG 백본 래퍼. 
    10ms 제약 통과를 위해 배포 전 반드시 convert_for_inference() 호출 요망.
    """
    backbone: nn.Module
    
    def Build(
        self,
        variant: str,
        out_indices: list[int] | None = None,
        timm_kwargs: dict[str, Any] | None = None,
        **build_kwarg
    ):
        if variant not in _REPVGG_VARIANTS:
            raise ValueError(f"Unsupported RepVGG variant '{variant}'")

        self.backbone = load_timm_backbone(
            model_name=_REPVGG_VARIANTS[variant],
            out_indices=out_indices,
            **(timm_kwargs or {})
        )

        # 추론 모드(trainable=False)일 경우 자동으로 구조적 재매개변수화 수행
        if self.trainable:
            self.convert_for_inference()

    def convert_for_inference(self):
        """
        다중 분기 토폴로지를 단일 3x3 Conv로 병합 (Re-parameterization)
        시간 복잡도를 물리적으로 소거함.
        """
        for module in self.backbone.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        print(f"[{MODEL_NAME}] Structural re-parameterization completed for inference.")

    def forward(self, x: torch.Tensor, **kwarg) -> list[torch.Tensor]:
        return self.backbone(x)
