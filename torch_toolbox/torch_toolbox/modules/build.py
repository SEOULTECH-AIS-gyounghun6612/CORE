from __future__ import annotations
from typing import cast

from python_toolbox.registry import Registry

from .. import CFGS
from .definition import Composable_Config, Composable_Module, Module_Config_Template
from .model.definition import Trainable_Model, Trainable_Model_Config
from .model.backbone import BACKBONES, BACKBONE_CFGS
from .loss.definition import Assemble_Loss_Config, Assemble_Loss


MODULES_CONFIG = (
    Module_Config_Template
    | Assemble_Loss_Config
    | Trainable_Model_Config | BACKBONE_CFGS
)
MODULES = (
    Composable_Module
    | Assemble_Loss
    | Trainable_Model | BACKBONES
)


def Build_from_registry(config: MODULES_CONFIG, registry: Registry) -> MODULES:
    """Config 트리를 재귀적으로 순회하며 모듈을 조립한다.

    Composable_Config이면 sub_module_meta의 각 항목을 CFGS로 인스턴스화한 뒤
    재귀 빌드하고, 결과를 상위 모듈의 Build(**sub_modules)에 키워드 인자로 주입한다.
    리프 Config는 재귀 없이 바로 인스턴스화한다.

    Args:
        config: 빌드할 모듈의 Config.
        registry: 대상 도메인 레지스트리 (MODELS, LOSSES 등).

    Returns:
        조립 완료된 Composable_Module 인스턴스.
    """
    _sub_kwargs = {}

    if isinstance(config, Composable_Config):
        # 자식 먼저 빌드: 부모의 Build()가 서브모듈을 **kwargs로 받기 때문
        for _k, _meta in config.sub_module_meta.items():
            # sub_module_meta dict → Config 인스턴스화 후 재귀 빌드
            _sub_cfg = cast(Composable_Config, CFGS.Get(_meta["config_type"])(**_meta))
            _sub_kwargs[_k] = Build_from_registry(_sub_cfg, registry)

    # 평탄화된 하이퍼파라미터 + 재귀 빌드된 서브모듈을 함께 생성자에 주입
    return registry.Get(config.object_type)(**config.Extract(), **_sub_kwargs)