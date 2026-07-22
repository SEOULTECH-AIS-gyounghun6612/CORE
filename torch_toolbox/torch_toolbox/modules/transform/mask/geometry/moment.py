from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..... import CFGS
from .... import MODELS
from ....definition import Composable_Config
from ....model.definition import Trainable_Model

from .spec import Feature_Spec

"""정렬 프레임 비대칭·모멘트 — **FP16 안전 정식화**.

기존 구현의 두 결함을 여기서 고친다.

``_signed_moments`` — 오버플로::

    dx 최대 ~112 (224 캔버스) -> dx^3 ~= 1.4e6
    FP16 최대값 65504 -> **합산 이전에 단일 항이 21배 초과** -> inf -> slog(inf) = inf

    처방: 좌표를 [-1, 1] 로 정규화한 뒤 3제곱한다 (:class:`Frame_Coords` 가 이미 나눠서 준다).
          수학적으로 동일하고 결과가 [-1, 1] 유계다.

``_chirality`` — 상쇄 소거::

    lr = (n_right - n_left) * pa,  각 count ~1e4
    FP16 은 2048 까지만 정수를 정확히 표현 -> 1e4 부근 눈금 8
    align_pca 가 좌우 질량을 맞춰 두므로 lr 자체는 작다 -> 큰 수 두 개의 차로 작은 수를 얻는다

    처방: 카운트 차가 아니라 **비율**로 정식화한다. [-1, 1] 유계이고 상쇄가 사라진다.

두 처방 모두 결과가 유계라 ``Feature_Spec.value_range`` 선언과 그대로 맞물린다.
정규화가 곧 안전성이고, 안전성이 곧 선언 가능한 범위다.

좌표는 :class:`~torch_toolbox.modules.transform.mask.canonical.Frame_Coords` 가 주는 무차원 ``(u, v)`` 다.
원점이 centroid 이므로 1차 모멘트는 항상 0 이고, 형상 정보는 3차부터 나온다.
"""


_CHIRALITY_MOMENTS_NAME = "chirality_moments"
_CHIRALITY_MOMENTS_CFG  = f"{_CHIRALITY_MOMENTS_NAME}_Config"


@CFGS.Register_module(_CHIRALITY_MOMENTS_CFG)
@dataclass
class Chirality_Moments_Config(Composable_Config):
    """정렬 프레임 비대칭·모멘트 설정. 파라미터가 없다 — 좌표가 이미 무차원이다.
    """
    config_type: str = _CHIRALITY_MOMENTS_CFG
    object_type: str = _CHIRALITY_MOMENTS_NAME
    trainable: bool = False


@MODELS.Register_module(_CHIRALITY_MOMENTS_NAME)
class Chirality_Moments(Trainable_Model):
    """정렬 프레임 좌우/상하 불균형 + 3차 중심 모멘트.

    출력 6차원::

        chirality_lr  주축(+u) 방향 픽셀 수 불균형 비율      [-1, 1]
        chirality_ud  수직(+v) 방향 픽셀 수 불균형 비율      [-1, 1]  (카이랄 신호)
        mu30, mu03, mu21, mu12   3차 중심 모멘트             [-1, 1]

    상하 불균형이 카이랄(거울상 부호 반전) 신호다. 정렬이 **회전만** 하고 반사를 쓰지
    않으므로 이 정보가 지워지지 않는다.
    """

    dim: int = 6

    def Out_channels(self) -> list[int]:
        return [self.dim]

    def Build(self, **kwargs: Any) -> None:
        """상태가 없다."""

    def Spec(self) -> tuple[Feature_Spec, ...]:
        """범위 선언. 비율·정규화 좌표라 전부 [-1, 1] 유계다 (FP16 안전 정식화의 결과)."""
        return (
            Feature_Spec("chirality", 2, "identity", (-1.0, 1.0)),
            Feature_Spec("moments",   4, "identity", (-1.0, 1.0)),
        )

    def forward(self, mask: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        Args:
            mask: (B, 1, H, W) float — 전경 1 / 배경 0.
            u:    (B, H, W) float — 주축 방향 무차원 좌표.
            v:    (B, H, W) float — 수직 방향 무차원 좌표.

        Returns:
            (B, 6) float.
        """
        _m = mask[:, 0]
        _n = _m.sum(dim=(1, 2)).clamp_min(1.0)

        # 비율 정식화 — 카운트 차가 아니므로 상쇄 소거가 없다.
        _lr = (_m * torch.where(u >= 0, 1.0, -1.0)).sum(dim=(1, 2)) / _n
        _ud = (_m * torch.where(v >= 0, 1.0, -1.0)).sum(dim=(1, 2)) / _n

        # 좌표가 이미 [-1, 1] 이라 3제곱해도 유계다.
        _mu30 = (_m * u * u * u).sum(dim=(1, 2)) / _n
        _mu03 = (_m * v * v * v).sum(dim=(1, 2)) / _n
        _mu21 = (_m * u * u * v).sum(dim=(1, 2)) / _n
        _mu12 = (_m * u * v * v).sum(dim=(1, 2)) / _n

        return torch.stack([_lr, _ud, _mu30, _mu03, _mu21, _mu12], dim=1)
