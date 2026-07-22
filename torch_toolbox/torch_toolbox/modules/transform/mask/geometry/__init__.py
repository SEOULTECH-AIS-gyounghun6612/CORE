from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from ..... import CFGS
from .... import MODELS
from ....definition import Composable_Config
from ....model.definition import Trainable_Model

from ..canonical import Centroid_Frame, Frame_Coords
from .fourier import Fourier_Descriptor
from .moment import Chirality_Moments
from .profile import Profile_Stats
from .region import Region_Scalars
from .spec import (
    Apply_transform, Feature_Spec, Normalizer,
)
from ..occupancy import Occupancy, Radial_Profile
from ..polar import Polar_Raster

"""실루엣 마스크 -> geometry embedding. **torch 단일 소스.**

기존 ``학습 프로젝트의 옛 sample_extractor.py`` 를 대체한다. numpy/scipy/skimage/cv2 의존이
사라지므로 "학습·추론·배포가 같은 연산을 쓴다"가 라이브러리 버전 고정이 아니라
**같은 그래프**로 보장된다.

파이프라인::

    mask -> Centroid_Frame          (centroid, 주축각 — 파라미터만, raster 회전 없음)
         -> Frame_Coords            (정렬 좌표 u, v — 무차원 [-1, 1])
         -> Polar_Raster            (backward gather, occupancy 분수)
         -> Occupancy / Radial_Profile / Region_Scalars / Chirality_Moments
         -> Profile_Stats / Fourier_Descriptor
         -> transform (log1p / slog)  ->  Normalizer  ->  (B, FEAT_DIM)

``FEAT_DIM`` 은 하드코딩하지 않는다. 각 서브모듈이 :class:`Feature_Spec` 으로 자기 차원과
범위를 선언하고 여기서 합산한다 — 설정을 바꿔도 슬라이스가 조용히 어긋나지 않는다.
"""


_NAME = "geometry_embedding"
_CFG  = f"{_NAME}_Config"


@CFGS.Register_module(_CFG)
@dataclass
class Geometry_Embedding_Config(Composable_Config):
    """극좌표 기반 형상 embedding 설정.

    Attributes:
        size: 캔버스 ``(H, W)``. 데이터셋 target_size 와 같아야 한다.
        num_radial: 극좌표 r bin 수 ``NR``.
        num_angular: 극좌표 theta bin 수 ``NT``. radial_outer 차원이자 Fourier 길이.
        sub: 극좌표 셀당 축별 sub-sample 수. 늘리면 바깥쪽 얇은 구멍 민감도가 오른다.
        num_harmonics: Fourier 유지 harmonic 수 ``K``. 그룹당 mag K + phase 2K.
        r_max: 최대 반경(px). None 이면 캔버스 반대각.
        occupancy_threshold: 셀을 "재료 있음"으로 볼 occupancy 분수 하한.
        quantiles: 프로파일 통계 백분위.
    """
    config_type: str = _CFG
    object_type: str = _NAME
    trainable: bool = False
    size: tuple[int, int] = (224, 224)
    num_radial: int = 224
    num_angular: int = 512
    sub: int = 1
    num_harmonics: int = 20
    r_max: float | None = None
    occupancy_threshold: float = 0.5
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)


@MODELS.Register_module(_NAME)
class Geometry_Embedding(Trainable_Model):
    """``(B, 1, H, W)`` 이진 마스크 -> ``(B, FEAT_DIM)`` geometry feature.

    Args:
        size: 캔버스 ``(H, W)``.
        num_radial: 극좌표 r bin 수.
        num_angular: 극좌표 theta bin 수.
        sub: 극좌표 셀당 축별 sub-sample 수 (표본 ``sub**2``).
        num_harmonics: Fourier 유지 harmonic 수.

    Note:
        입력 마스크는 fill/최대연결성분 처리를 **하지 않은** 것이어야 한다. 관통 구멍은
        형상 정보이고, 실측상 GT 표본의 56%가 구멍을 갖는다.
    """

    def Build(
        self,
        size: tuple[int, int] = (224, 224),
        num_radial: int = 224,
        num_angular: int = 512,
        sub: int = 1,
        num_harmonics: int = 20,
        r_max: float | None = None,
        occupancy_threshold: float = 0.5,
        quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
        **kwargs: Any,
    ) -> None:
        _K = dict(trainable=False, size=size)
        self.frame  = Centroid_Frame(name="frame", **_K)
        self.coords = Frame_Coords(name="coords", **_K)
        self.polar  = Polar_Raster(
            name="polar", num_radial=num_radial, num_angular=num_angular,
            sub=sub, r_max=r_max, **_K)
        self.occ    = Occupancy(
            name="occ", num_radial=num_radial, num_angular=num_angular, **_K)
        self.radial = Radial_Profile(
            name="radial", num_radial=num_radial, r_max=r_max,
            threshold=occupancy_threshold, **_K)
        self.region = Region_Scalars(name="region", num_angular=num_angular, **_K)
        self.moment = Chirality_Moments(name="moment", trainable=False)
        self.stats  = Profile_Stats(
            name="stats", trainable=False, length=num_angular, quantiles=quantiles)
        self.four   = Fourier_Descriptor(
            name="fourier", trainable=False, length=num_angular,
            num_harmonics=num_harmonics)

        _rmax = self.polar.r_max
        _specs: list[Feature_Spec] = [
            *self.region.Spec(),
            *self.moment.Spec(),
            *self.occ.Spec(),
            Feature_Spec("radial_outer", num_angular, "log1p", (0.0, math.log1p(_rmax))),
            *self.stats.Spec("outer_stats",     _rmax),
            *self.stats.Spec("inner_stats",     _rmax),
            *self.stats.Spec("thickness_stats", _rmax),
            *self.stats.Spec("coverage_stats",  1.0),
            *self.four.Spec("outer", _rmax),
            *self.four.Spec("inner", _rmax),
        ]
        self.specs = tuple(_specs)
        self.norm = Normalizer(self.specs)

    @property
    def feat_dim(self) -> int:
        """총 출력 차원."""
        return sum(_s.dim for _s in self.specs)

    def Out_channels(self) -> list[int]:
        """embedding 차원 하나. 헤더 입력 차원이 이 값을 참조한다."""
        return [self.feat_dim]

    @property
    def groups(self) -> dict[str, tuple[int, int]]:
        """그룹 이름 -> ``(start, end)`` 슬라이스. spec 에서 도출하므로 어긋날 수 없다."""
        _out, _at = {}, 0
        for _s in self.specs:
            _out[_s.name] = (_at, _at + _s.dim)
            _at += _s.dim
        return _out

    def _Compute(self, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """본체. ``(transform 적용된 raw, r_outer, frame_angle)`` 을 함께 낸다.

        ``r_outer`` 와 ``frame_angle`` 은 embedding 을 만드는 과정에서 이미 계산되므로
        내보내는 데 **추가 연산이 없다**. 견본 대비 theta shift 계산에 쓰려고 남긴다 —
        정렬이 이미 회전을 대부분 먹었으므로 견본 비교가 잡아주는 것은 주로 PCA 의
        180° 모호성과 근대칭 형상에서의 주축 불안정이다.
        """
        _frame = self.frame(mask)
        _u, _v = self.coords(_frame)
        _polar = self.polar(mask, _frame)
        _prof  = self.radial(_polar)
        _thick = _prof.r_outer - _prof.r_inner

        _o_mag, _o_ph = self.four(_prof.r_outer)
        _i_mag, _i_ph = self.four(_prof.r_inner)

        _raw = torch.cat(
            [
                self.region(mask, _u, _v, _prof),      # size 7 + ratio 5 + position 2
                self.moment(mask, _u, _v),             # chirality 2 + moments 4
                self.occ(mask, _polar),                # area_cartesian / polar / 비
                _prof.r_outer,                         # radial_outer (NT)
                self.stats(_prof.r_outer),
                self.stats(_prof.r_inner),
                self.stats(_thick),
                self.stats(_prof.coverage),
                _o_mag, _o_ph, _i_mag, _i_ph,
            ],
            dim=1,
        )

        _out, _at = [], 0
        for _s in self.specs:
            _out.append(Apply_transform(_raw[:, _at: _at + _s.dim], _s.transform))
            _at += _s.dim
        return torch.cat(_out, dim=1), _prof.r_outer, _frame.angle

    def Raw(self, mask: Tensor) -> Tensor:
        """transform 까지만 적용하고 정규화 **전** 값을 낸다.

        측정 통계로 :meth:`Normalizer.Set_statistics` 를 채울 때 이 출력을 모은다.
        """
        return self._Compute(mask)[0]

    def Forward_with_aux(self, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``(feature, r_outer, frame_angle)``.

        배포 그래프에서 외곽 프로파일과 주축각을 함께 빼낼 때 쓴다. ``r_outer`` 는
        견본과의 순환 상관으로 theta shift 를 구하는 입력이고, ``frame_angle`` 은
        PCA 가 잡은 절대 회전각이다.

        Returns:
            feature (B, FEAT_DIM), r_outer (B, NT), frame_angle (B,).
        """
        _raw, _r_outer, _angle = self._Compute(mask)
        return self.norm(_raw), _r_outer, _angle

    def forward(self, mask: Tensor) -> Tensor:
        """
        Args:
            mask: (B, 1, H, W) float — 전경 1 / 배경 0.

        Returns:
            (B, FEAT_DIM) float — transform + 정규화 완료.
        """
        return self.norm(self.Raw(mask))
