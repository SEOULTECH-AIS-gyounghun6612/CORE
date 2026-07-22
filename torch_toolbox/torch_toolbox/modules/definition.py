from __future__ import annotations
from typing import Any, ClassVar
from dataclasses import dataclass, field

import torch
from torch.nn import Module

from python_toolbox.project.config import Base_Config


@dataclass
class Module_Config_Template(Base_Config):
    """모든 PyTorch 모듈 Config의 최상위 뼈대.

    Attributes:
        name: 모듈 식별자.
        config_type: CFGS 레지스트리 조회 키.
        object_type: 도메인 레지스트리(MODELS, LOSSES 등) 조회 키.
        trainable: False이면 빌드 후 파라미터 grad 비활성화.
    """

    name: str = "default_name"
    config_type: str = "Config"
    object_type: str = "Model"
    trainable: bool = True


@dataclass
class Composable_Config(Module_Config_Template):
    """계층적 서브모듈 선언을 지원하는 Config.

    sub_module_meta를 raw dict로만 보유한다.
    CFGS 레지스트리를 통한 서브 Config 인스턴스화는 __Build_from_registry__에서 수행한다.

    Attributes:
        sub_module_meta: 서브모듈 키 → raw Config dict 매핑.
            Extract() 대상에서 제외되며 __Build_from_registry__가 소비한다.
    """

    __exclude_extract__: ClassVar[set[str]] = {"sub_module_meta"}

    sub_module_meta: dict[str, dict[str, Any]] = field(default_factory=dict)


class Composable_Module(Module):
    """레지스트리 기반 조립을 지원하는 PyTorch 모듈 추상 기반 클래스.

    생성 시 Build()를 즉시 호출하여 서브모듈을 구성한다.
    서브클래스는 Build()와 forward()만 구현하면 된다.

    Attributes:
        name: 모듈 식별자. Config의 name 필드와 일치.
        trainable: False이면 모든 파라미터의 requires_grad를 비활성화.
    """

    def __init__(
        self, name: str,
        trainable: bool = True,
        **build_kwarg
    ) -> None:
        super().__init__()
        self.name = name
        self.trainable = trainable
        self.Build(**build_kwarg)
        if not trainable:
            self.requires_grad_(trainable)

    def Build(self, *arg, **build_kwarg):
        """서브모듈 및 레이어를 초기화한다.

        __Build_from_registry__가 Config.Extract()로 추출한 하이퍼파라미터와
        재귀 빌드된 서브모듈을 **kwargs로 전달한다.

        Args:
            **build_kwarg: Config에서 추출된 하이퍼파라미터 및 빌드된 서브모듈.
        """
        raise NotImplementedError

    def forward(self, *args, **kwarg):
        raise NotImplementedError

    def Out_channels(self) -> list[int]:
        """출력 텐서별 채널 수. **출력 하나당 한 항목**이며 단일 출력도 리스트다.

        같은 계층의 다른 모듈이 자기 입력 차원을 도출할 때 참조한다
        (``Build_from_registry``의 동일 계층 해석). config에서
        ``in_channels: {sum: [backbone, $feat_dim]}`` 처럼 이름으로 가리킨다.

        기본 구현은 실패한다 — **참조당하는 모듈만** 구현하면 되고, 구현하지 않은 모듈을
        가리키면 조립 시점에 바로 드러난다.

        Returns:
            출력 텐서별 채널 수 리스트.
        """
        raise NotImplementedError(
            f"{type(self).__name__}에 Out_channels()가 없다. config에서 이 모듈을 "
            f"차원 출처로 참조하려면 구현해야 한다."
        )

    def Load_weights(self, weight_path: str) -> None:
        """사전학습 가중치를 부분 로드한다.

        strict=False로 로드하므로 키 불일치 시 해당 레이어는 건너뛴다.

        Args:
            weight_path: 가중치 파일 경로 (.pt / .pth).
        """
        _state = torch.load(weight_path, map_location="cpu")
        self.load_state_dict(_state, strict=False)
