from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from ..definition import Composable_Config, Composable_Module


@dataclass
class Trainable_Model_Config(Composable_Config):
    """학습 하이퍼파라미터가 결합된 모델 설정 노드.

    Attributes:
        lr: 모듈별 학습률 override. None이면 Assembler의 base_lr 사용.
        weight_decay: 모듈별 weight decay override. None이면 base_weight_decay 사용.
    """

    lr: float | None = None
    weight_decay: float | None = None


class Trainable_Model(Composable_Module):
    """모듈별 학습률·weight decay 재정의를 지원하는 학습용 모델 추상 클래스.

    Attributes:
        lr: 이 모듈의 학습률 override. None이면 Assembler의 base_lr 사용.
        weight_decay: 이 모듈의 weight decay override. None이면 base_weight_decay 사용.
    """

    def __init__(
        self,
        name: str,
        trainable: bool = True,
        lr: float | None = None,
        weight_decay: float | None = None,
        **build_kwarg
    ) -> None:
        # Build()가 super().__init__() 내부에서 호출되므로
        # 서브클래스의 Build()가 lr / weight_decay를 참조할 경우를 위해 먼저 할당한다.
        self.lr = lr
        self.weight_decay = weight_decay
        super().__init__(name, trainable, **build_kwarg)

    def Get_group_map(
        self,
        base_lr: float,
        base_weight_decay: float,
        group_map: dict[tuple[int, int], list[Any]],
        scaled_value: int = 10 ** 10
    ) -> None:
        """파라미터 그룹 맵을 재귀적으로 구성한다.

        모듈 트리를 DFS로 순회하며 (scaled_lr, scaled_wd) 키별로 파라미터를 분류한다.
        자식이 Trainable_Model이면 해당 모듈의 lr/wd로 재귀하고,
        일반 nn.Module이면 현재 lr/wd로 파라미터를 수집한다.

        float 비교 오차를 피하기 위해 lr·wd에 scaled_value를 곱한 정수를 키로 사용한다.

        Args:
            base_lr: 부모로부터 상속된 학습률.
            base_weight_decay: 부모로부터 상속된 weight decay.
            group_map: (scaled_lr, scaled_wd) → 파라미터 리스트 누적 맵.
                호출자가 빈 dict를 생성해 전달하고, 재귀 호출이 in-place로 채운다.
            scaled_value: lr·wd를 정수 키로 변환하는 배율. 기본값 10^10.
        """
        if (_lr := getattr(self, "lr", None)) is None:
            _lr = base_lr

        if (_wd := getattr(self, "weight_decay", None)) is None:
            _wd = base_weight_decay

        _key = (int(_lr * scaled_value), int(_wd * scaled_value))

        _local_params = [
            _p for _p in self.parameters(recurse=False) if _p.requires_grad
        ]
        if _local_params:
            group_map.setdefault(_key, []).extend(_local_params)

        for _module in self.children():
            if isinstance(_module, Trainable_Model):
                # Trainable_Model 자식: 자체 lr/wd로 재귀하여 별도 그룹 구성
                _module.Get_group_map(_lr, _wd, group_map, scaled_value)
            else:
                # 일반 nn.Module: 현재 lr/wd 그룹에 하위 파라미터 전부 포함
                _standard_params = [
                    _p for _p in _module.parameters(recurse=True) if _p.requires_grad
                ]
                if _standard_params:
                    group_map.setdefault(_key, []).extend(_standard_params)
