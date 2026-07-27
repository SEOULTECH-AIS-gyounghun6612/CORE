from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from .... import CFGS
from ... import MODELS
from ...definition import Composable_Config
from ...model.definition import Trainable_Model
from typing import NamedTuple

from .geometry.spec import Feature_Spec

"""occupancy 관점의 영역 정보 — "그 좌표에 재료가 있는가"만 따진다.

fill 도 convex hull 도 쓰지 않는다. 관통 구멍은 메울 대상이 아니라 **비어 있는 셀**이고,
그 자체가 정보다.

핵심은 :class:`Occupancy` 가 내는 두 값이 **서로 다르다는 것**이다. 극좌표 셀의 물리적
크기는 ``r * dr * dtheta`` 로 반경에 비례해 커지므로, polar occupancy 는 사실상
**1/r 가중 면적**이 된다. Cartesian occupancy(= 실제 면적)와의 비는 "재료가 중심에
몰렸는가 바깥에 퍼졌는가"를 바로 준다. 셀 면적으로 가중해 둘을 일치시키면 이 정보가
사라지므로 **가중하지 않는다.**

occupancy 를 boolean 이 아니라 **분수**로 다루는 이유: 바깥쪽 셀은 폭이 픽셀보다 커서
(224 캔버스·512 bin 기준 r=157px 에서 약 1.9px) 얇은 구멍을 통째로 삼킨다. 분수로 두면
구멍이 사라지는 대신 값을 떨어뜨려 흔적을 남긴다.
"""


class Region_Profile(NamedTuple):
    """theta별 반경 프로파일.

    Attributes:
        r_outer:  (B, NT) float — theta별 재료가 있는 최대 반경(px). 없으면 0.
        r_inner:  (B, NT) float — theta별 재료가 있는 최소 반경(px). 없으면 0.
        coverage: (B, NT) float — theta별 occupancy 평균 [0, 1]. 재료가 차지한 비율.
    """

    r_outer:  Tensor
    r_inner:  Tensor
    coverage: Tensor


_OCCUPANCY_TOTALS_NAME = "occupancy_totals"
_OCCUPANCY_TOTALS_CFG  = f"{_OCCUPANCY_TOTALS_NAME}_Config"


@CFGS.Register_module(_OCCUPANCY_TOTALS_CFG)
@dataclass
class OccupancyTotals_Config(Composable_Config):
    """직교/극좌표 occupancy 총량 설정.

    Attributes:
        size: 캔버스 ``(H, W)``.
        num_radial / num_angular: 극좌표 격자. 셀 수 도출에 쓴다.
    """
    config_type: str = _OCCUPANCY_TOTALS_CFG
    object_type: str = _OCCUPANCY_TOTALS_NAME
    trainable: bool = False
    size: tuple[int, int] = (224, 224)
    num_radial: int = 224
    num_angular: int = 512


@MODELS.Register_module(_OCCUPANCY_TOTALS_NAME)
class Occupancy(Trainable_Model):
    """직교/극좌표 occupancy 총량.

    Returns 는 ``(area_cartesian, area_polar)`` 두 스칼라다. 둘의 차이가 곧 정보이므로
    같은 단위로 맞추지 않는다.

    - ``area_cartesian``: 전경 픽셀 합. 균일 격자이므로 실제 면적에 비례한다.
    - ``area_polar``: 극좌표 셀 occupancy 분수의 합. 셀 크기가 ``r`` 에 비례하므로
      중심부 재료가 과대·주변부가 과소 계상된다 (= 1/r 가중).

    Args:
        size: 캔버스 ``(H, W)``.
        cells: 극좌표 셀 수 ``NR * NT``.
    """

    dim: int = 3

    def Out_channels(self) -> list[int]:
        return [self.dim]

    def Build(
        self, size: tuple[int, int] = (224, 224),
        num_radial: int = 224, num_angular: int = 512, **kwargs: Any,
    ) -> None:
        self.size = (int(size[0]), int(size[1]))
        self.cells = int(num_radial) * int(num_angular)

    def Spec(self) -> tuple[Feature_Spec, ...]:
        """두 면적은 픽셀 수·셀 수에 유계. 비는 실용 상한으로 선언한다.

        **원본 스케일(선형)** — 형상 크기라 log 로 뭉개지 않는다.
        """
        return (
            Feature_Spec("area_cartesian", 1, "identity",
                         (0.0, float(self.size[0] * self.size[1]))),
            Feature_Spec("area_polar",     1, "identity", (0.0, float(self.cells))),
            Feature_Spec("area_ratio",     1, "identity", (0.0, 20.0)),
        )

    def forward(self, mask: Tensor, polar: Tensor) -> Tensor:
        """
        Args:
            mask:  (B, 1, H, W) float — 전경 1 / 배경 0.
            polar: (B, NR, NT) float — :class:`~torch_toolbox.modules.transform.mask.polar.Polar_Raster` 출력.

        Returns:
            (B, 3) float — ``area_cartesian``, ``area_polar``, 그 비.
        """
        _cart = mask.sum(dim=(1, 2, 3))
        _pol = polar.sum(dim=(1, 2))
        return torch.stack([_cart, _pol, _pol / _cart.clamp_min(1.0)], dim=1)


_RADIAL_PROFILE_NAME = "radial_profile"
_RADIAL_PROFILE_CFG  = f"{_RADIAL_PROFILE_NAME}_Config"


@CFGS.Register_module(_RADIAL_PROFILE_CFG)
@dataclass
class RadialProfile_Config(Composable_Config):
    """theta별 반경 프로파일 설정.

    Attributes:
        size: 캔버스 ``(H, W)``. ``r_max`` 가 None 일 때 반대각을 구한다.
        num_radial: r bin 수. ``dr`` 도출에 쓴다.
        r_max: 최대 반경(px). None 이면 캔버스 반대각.
        threshold: 셀을 "재료 있음"으로 볼 occupancy 분수 하한.
    """
    config_type: str = _RADIAL_PROFILE_CFG
    object_type: str = _RADIAL_PROFILE_NAME
    trainable: bool = False
    size: tuple[int, int] = (224, 224)
    num_radial: int = 224
    r_max: float | None = None
    threshold: float = 0.5


@MODELS.Register_module(_RADIAL_PROFILE_NAME)
class Radial_Profile(Trainable_Model):
    """극좌표 occupancy -> theta별 ``(r_outer, r_inner, coverage)``.

    기존 ``_mask_to_polar`` 는 theta마다 min/max 로 접어버려 그 사이 정보가 사라졌고,
    빈 bin 은 ``_fill_circular_nan`` 이 양끝 보간으로 **없는 재료를 지어냈다**. 여기서는
    빈 bin 을 0 으로 둔다 — "이 방향엔 재료가 없다"는 형상에 대한 사실이다.

    ``r_inner = 0`` 이 "중심에 재료 있음"과 "재료 없음" 양쪽에 쓰이므로, 구분이 필요하면
    ``coverage`` 를 함께 본다 (재료가 없으면 ``coverage == 0``).

    Args:
        size / num_radial / r_max: ``dr`` 도출. ``Polar_Raster`` 와 같은 값이어야 한다.
        threshold: 셀을 "재료 있음"으로 볼 occupancy 분수 하한.
    """

    def Out_channels(self) -> list[int]:
        """r_outer / r_inner / coverage 세 텐서."""
        return [1, 1, 1]

    def Build(
        self, size: tuple[int, int] = (224, 224), num_radial: int = 224,
        r_max: float | None = None, threshold: float = 0.5, **kwargs: Any,
    ) -> None:
        _rmax = (math.hypot((size[0] - 1) / 2.0, (size[1] - 1) / 2.0)
                 if r_max is None else float(r_max))
        self.dr = _rmax / int(num_radial)
        self.threshold = float(threshold)

    def forward(self, polar: Tensor) -> Region_Profile:
        """
        Args:
            polar: (B, NR, NT) float — occupancy 분수.

        Returns:
            :class:`Region_Profile`.
        """
        _nr = polar.shape[1]
        _idx = torch.arange(_nr, device=polar.device, dtype=polar.dtype).view(1, -1, 1)
        _hit = polar >= self.threshold                                   # (B, NR, NT)

        # 재료가 없는 셀은 max 에서 -1, min 에서 NR 로 밀어 극단값이 잡히지 않게 한다.
        _outer = torch.where(_hit, _idx, torch.full_like(_idx, -1.0)).amax(dim=1)
        _inner = torch.where(_hit, _idx, torch.full_like(_idx, float(_nr))).amin(dim=1)
        _has = _outer >= 0.0                                             # (B, NT)

        _zero = torch.zeros_like(_outer)
        return Region_Profile(
            r_outer=torch.where(_has, (_outer + 0.5) * self.dr, _zero),
            r_inner=torch.where(_has, (_inner + 0.5) * self.dr, _zero),
            coverage=polar.mean(dim=1),
        )


_RADIAL_RLE_NAME = "radial_rle"
_RADIAL_RLE_CFG  = f"{_RADIAL_RLE_NAME}_Config"


@CFGS.Register_module(_RADIAL_RLE_CFG)
@dataclass
class RadialRLE_Config(Composable_Config):
    """theta별 재료 RLE(전이점) 설정.

    Attributes:
        size: 캔버스 ``(H, W)``. ``r_max`` None 일 때 반대각.
        num_radial: r bin 수. ``dr`` 도출.
        r_max: 최대 반경(px). None 이면 캔버스 반대각.
        threshold: 셀을 "재료 있음"으로 볼 occupancy 분수 하한.
        max_transitions: theta 당 담을 최대 **전이점 수** ``K``. 넘으면 자른다. 출력 차원 = K.
    """
    config_type: str = _RADIAL_RLE_CFG
    object_type: str = _RADIAL_RLE_NAME
    trainable: bool = False
    size: tuple[int, int] = (224, 224)
    num_radial: int = 224
    r_max: float | None = None
    threshold: float = 0.5
    max_transitions: int = 8


@MODELS.Register_module(_RADIAL_RLE_NAME)
class Radial_RLE(Trainable_Model):
    """극좌표 occupancy -> theta별 **재료 RLE 간격** ``(B, NT, K)``.

    각 theta 방사선을 r 축으로 훑어 재료 있음/없음의 구간을 **간격**으로 담는다:
    ``[시작 반경, 살1 두께, 구멍1 두께, 살2 두께, 구멍2 두께, …]`` (안쪽부터). 통짜면
    ``[r0, 두께, 0, …]``, 살-구멍-살이면 ``[r0, 살1, 구멍, 살2, 0, …]``.

    전이점(절대 반경)이 아니라 **간격**인 이유: 각 슬롯 의미가 고정(시작·살1·구멍1·…)이라 stem
    마다 밴드 개수가 달라도 **위치별 평균이 성립**한다(전이점은 밴드 없는 stem 에서 슬롯이 밀려
    class 평균이 무너진다). centroid 위치와 무관하게 구멍이 표현되는 건 그대로.

    빈 자리는 0. ``Radial_Profile`` 이 theta 당 최외곽/최내곽 하나씩만 접던 한계를 여기서 편다.

    Args:
        size / num_radial / r_max: ``dr`` 도출. ``Polar_Raster`` 와 같은 값이어야 한다.
        threshold: 재료 판정 하한.
        max_transitions: theta 당 슬롯 수 K (시작 + 살/구멍 간격들).
    """

    def Out_channels(self) -> list[int]:
        """간격 한 텐서 — ``K`` 채널 ``[시작r, 살1, 구멍1, 살2, …]`` (안쪽부터)."""
        return [self.max_transitions]

    def Build(
        self, size: tuple[int, int] = (224, 224), num_radial: int = 224,
        r_max: float | None = None, threshold: float = 0.5, max_transitions: int = 8,
        **kwargs: Any,
    ) -> None:
        _rmax = (math.hypot((size[0] - 1) / 2.0, (size[1] - 1) / 2.0)
                 if r_max is None else float(r_max))
        self.dr = _rmax / int(num_radial)
        self.threshold = float(threshold)
        self.max_transitions = int(max_transitions)

    def forward(self, polar: Tensor) -> Tensor:
        """
        Args:
            polar: (B, NR, NT) float — occupancy 분수.

        Returns:
            (B, NT, K) float — theta 당 간격 ``[시작r, 살1, 구멍1, 살2, …]`` (px). 빈 자리 0.
        """
        _b, _nr, _nt = polar.shape
        _hit = (polar >= self.threshold).to(polar.dtype)                 # (B, NR, NT)

        # r 축 인접 차분의 절댓값으로 **모든 전이**(0->1, 1->0) 검출. r=0 앞·r=NR 뒤 0 패딩.
        _pad = torch.zeros(_b, 1, _nt, dtype=polar.dtype, device=polar.device)
        _h = torch.cat([_pad, _hit, _pad], dim=1)                        # (B, NR+2, NT)
        _trans = (_h[:, 1:] - _h[:, :-1]).abs() > 0.5                    # (B, NR+1, NT) 전이 위치
        _ridx = torch.arange(_nr + 1, device=polar.device, dtype=polar.dtype).view(1, -1, 1)

        _K = self.max_transitions
        _rank = torch.cumsum(_trans.to(torch.int64), dim=1)             # 전이 누적 순번
        # 먼저 전이 반경 K+1 개(시작 슬롯 + 간격 K-1 개를 만들려면 전이 K 개 필요)를 뽑는다.
        _tr = torch.zeros(_b, _nt, _K, dtype=polar.dtype, device=polar.device)
        _cnt = _rank[:, -1, :]                                           # (B, NT) theta 당 전이 수
        for _k in range(_K):
            _sel = _trans & (_rank == (_k + 1))
            _tr[:, :, _k] = (_ridx * _sel.to(_ridx.dtype)).sum(dim=1) * self.dr

        # 간격으로 변환: [t0, t1-t0, t2-t1, …]. 단 존재하는 전이까지만(그 뒤 슬롯은 0).
        _out = torch.zeros_like(_tr)
        _out[:, :, 0] = _tr[:, :, 0]                                     # 시작 반경
        for _k in range(1, _K):
            _gap = _tr[:, :, _k] - _tr[:, :, _k - 1]
            _valid = (_cnt >= (_k + 1))                                  # k+1 번째 전이가 존재
            _out[:, :, _k] = torch.where(_valid, _gap, torch.zeros_like(_gap))
        return _out
