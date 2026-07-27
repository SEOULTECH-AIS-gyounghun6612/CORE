from __future__ import annotations
from dataclasses import dataclass
import math
import warnings
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ..... import CFGS
from .... import MODELS
from ....definition import Composable_Config
from ....model.definition import Trainable_Model

from ..canonical import Centroid_Frame, Frame_Coords
from .moment import Chirality_Moments
from .region import Region_Scalars
from ..occupancy import Occupancy, Radial_Profile, Radial_RLE
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

#: 생성 그룹 -> 데이터 그룹(물리량 종류). 같은 key 끼리 pool/표시 단위로 묶는다. 매핑에 없으면
#: 자기 이름이 곧 key(고유 그룹). radial 계열은 모두 px 반경이라 한 pool 이어야 outer-inner 간격
#: (살 두께·구멍 깊이)이 표준화로 안 사라진다. size 는 면적(px²)·길이(px) 혼합이라 자기 그룹으로 둔다.
DATA_GROUP: dict[str, str] = {
    "radial_outer": "radius", "radial_inner": "radius",
    "outer_stats": "radius", "inner_stats": "radius", "thickness_stats": "radius",
    "coverage_stats": "coverage",
    "area_cartesian": "area", "area_polar": "area",
    "area_ratio": "ratio", "ratio": "ratio",
    "chirality": "moment", "moments": "moment",
    "position": "position", "size": "size",
    "outer_fft_mag": "spectral_mag", "inner_fft_mag": "spectral_mag",
    "outer_fft_phase": "spectral_phase", "inner_fft_phase": "spectral_phase",
}


@CFGS.Register_module(_CFG)
@dataclass
class Geometry_Embedding_Config(Composable_Config):
    """극좌표 기반 형상 **토큰** embedding 설정.

    Attributes:
        size: 캔버스 ``(H, W)``. 데이터셋 target_size 와 같아야 한다.
        num_radial: 극좌표 r bin 수 ``NR``.
        num_angular: 극좌표 theta bin 수 ``NT``. 각도 토큰 수(radial_rle).
        sub: 극좌표 셀당 축별 sub-sample 수. 늘리면 바깥쪽 얇은 구멍 민감도가 오른다.
        r_max: 최대 반경(px). None 이면 캔버스 반대각.
        occupancy_threshold: 셀을 "재료 있음"으로 볼 occupancy 분수 하한.
        max_transitions: radial_rle theta 당 최대 전이점 수 ``K``. 토큰 차원 상한에도 기여한다
            (token_dim = max(전역 그룹 차원, K)).
        variance_warn: 토큰 그룹 차원 편차 경고 배율 — max/min 이 이 값을 넘으면 경고(결합 실수 힌트).
    """
    config_type: str = _CFG
    object_type: str = _NAME
    trainable: bool = False
    size: tuple[int, int] = (224, 224)
    num_radial: int = 224
    num_angular: int = 512
    sub: int = 1
    r_max: float | None = None
    occupancy_threshold: float = 0.5
    max_transitions: int = 8
    variance_warn: float = 3.0


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
        r_max: float | None = None,
        occupancy_threshold: float = 0.5,
        max_transitions: int = 8,
        variance_warn: float = 3.0,
        **kwargs: Any,
    ) -> None:
        _K = dict(trainable=False, size=size)
        # 파이프라인(공유 인프라 — 항상). 중간값 frame/u,v/polar/prof 를 만든다.
        self.frame  = Centroid_Frame(name="frame", **_K)
        self.coords = Frame_Coords(name="coords", **_K)
        self.polar  = Polar_Raster(
            name="polar", num_radial=num_radial, num_angular=num_angular,
            sub=sub, r_max=r_max, **_K)
        self.radial = Radial_Profile(
            name="radial", num_radial=num_radial, r_max=r_max,
            threshold=occupancy_threshold, **_K)

        # descriptor(선택 대상 — "여러 출력"). 지금 토큰 스키마: region·moment·occ(전역) + radial_rle(각도).
        self.region = Region_Scalars(name="region", num_angular=num_angular, **_K)
        self.moment = Chirality_Moments(name="moment", trainable=False)
        self.occ    = Occupancy(name="occ", num_radial=num_radial, num_angular=num_angular, **_K)
        self.rle    = Radial_RLE(
            name="rle", num_radial=num_radial, num_angular=num_angular, r_max=r_max,
            threshold=occupancy_threshold, max_transitions=max_transitions, **_K)

        self._variance_warn = float(variance_warn)
        # 전역 descriptor 의 (출력, spec) 순서 — data_group 재묶기·토큰 조립에 쓴다.
        self._global = ["region", "moment", "occ"]
        # 토큰 차원 = max(전역 data_group 차원, rle 전이점 K). 조립 시 결정.
        _dims = self._global_group_dims()
        _dims["radial_rle"] = int(max_transitions)
        self._token_dim = max(_dims.values())
        self._warn_variance(_dims)

    # ── 토큰 메타 ──────────────────────────────────────────────────────────
    def _global_group_dims(self) -> dict[str, int]:
        """전역 descriptor 를 data_group 으로 재묶었을 때의 그룹별 차원."""
        _dims: dict[str, int] = {}
        for _name in self._global:
            for _s in getattr(self, _name).Spec():
                _dg = DATA_GROUP.get(_s.name, _s.name)
                _dims[_dg] = _dims.get(_dg, 0) + _s.dim
        return _dims

    def _warn_variance(self, dims: dict[str, int]) -> None:
        """토큰 조립 시 그룹 간 차원 편차가 크면 경고 — 결합 실수(엉뚱한 묶음)를 드러낸다."""
        if not dims:
            return
        _mx, _mn = max(dims.values()), min(dims.values())
        if _mx >= self._variance_warn * max(_mn, 1):
            warnings.warn(
                f"토큰 그룹 차원 편차 큼 (max {_mx} / min {_mn}) — 패딩 낭비/결합 실수 가능. "
                f"그룹별: {dims}", stacklevel=2)

    @property
    def token_dim(self) -> int:
        """토큰 하나의 차원 = max(전역 그룹, rle 전이점). 조립이 자동 결정."""
        return self._token_dim

    @property
    def token_groups(self) -> list[tuple[str, str]]:
        """토큰 순서 -> ``(data_group, axis)``, 길이 = seq. 전역(scalar) 먼저, 각도(radial_rle) NT 개.

        전역은 그룹당 토큰 1개, radial_rle 은 theta 당 토큰 1개(NT 개 모두 ``("radial_rle","angular")``).
        소비처가 어느 토큰이 무엇인지·극좌표로 그릴지(angular) 안다. **길이가 실제 seq 와 같아야 한다.**
        """
        _g = [(_dg, "scalar") for _dg in self._global_group_dims()]
        _nt = self.polar.num_angular
        return _g + [("radial_rle", "angular")] * _nt

    def Out_channels(self) -> list[int]:
        """토큰 차원. 헤더(transformer)가 이 값을 d_model 로 받는다."""
        return [self._token_dim]

    def Occupancy_grid(self, mask: Tensor) -> Tensor:
        """이진 마스크 -> 극좌표 occupancy 격자 ``(B, NR, NT)``. 시각화·밴드 RLE 용."""
        return self.polar(mask, self.frame(mask))

    # ── 토큰 조립 ──────────────────────────────────────────────────────────
    def _pad(self, x: Tensor) -> Tensor:
        """``(B, d)`` -> ``(B, 1, token_dim)`` 뒤 패딩 (전역 토큰 하나)."""
        _d = x.shape[1]
        if _d < self._token_dim:
            x = F.pad(x, (0, self._token_dim - _d))
        return x.unsqueeze(1)

    def Features(self, mask: Tensor) -> dict[str, Tensor]:
        """``(B, 1, H, W)`` 마스크 -> **도메인별 native feature** dict (병합·패딩 없음).

        각 도메인이 자기 자연 형태로 온다: 순서 없는 scalar 그룹은 ``(B, dim)``, 순서 있는
        radial_rle 은 ``(B, NT, K)``. 조립(병합)은 **소비처 책임**이다 — 검증은 도메인별로 따로 보고
        (성질별 거리·클러스터), 학습 헤더는 :meth:`Tokens` 로 병합한다. 같은 입력에서 같은 값을
        내는 것이 공유의 핵심이고, 조립까지 같을 필요는 없다.

        도메인 성질은 :attr:`domain_kinds` 가 밝힌다 (FEATURE=유클리드 / TOKEN=회전정합).
        정규화 전 raw(원본 px·값) — 정규화는 소비처가 선형 scale 로 따로 한다.
        """
        _frame = self.frame(mask)
        _u, _v = self.coords(_frame)
        _polar = self.polar(mask, _frame)
        _prof  = self.radial(_polar)

        # 전역(scalar) descriptor flat 출력을 data_group 으로 재묶기 — 각 그룹이 한 도메인.
        _flat = {
            "region": (self.region(mask, _u, _v, _prof), self.region.Spec()),
            "moment": (self.moment(mask, _u, _v),        self.moment.Spec()),
            "occ":    (self.occ(mask, _polar),           self.occ.Spec()),
        }
        _by_dg: dict[str, list[Tensor]] = {}
        for _name in self._global:
            _out, _specs = _flat[_name]
            _at = 0
            for _s in _specs:
                _dg = DATA_GROUP.get(_s.name, _s.name)
                _by_dg.setdefault(_dg, []).append(_out[:, _at: _at + _s.dim])
                _at += _s.dim

        _feats: dict[str, Tensor] = {_dg: torch.cat(_by_dg[_dg], dim=1)     # (B, dim) scalar 도메인
                                     for _dg in self._global_group_dims()}
        _feats["radial_rle"] = self.rle(_polar)                            # (B, NT, K) sequence 도메인
        return _feats

    @property
    def domain_kinds(self) -> dict[str, str]:
        """도메인 -> 성질 (``FEATURE`` 순서 없음 / ``TOKEN`` 순서 있음). 소비처가 거리 방식을 고른다."""
        from ....feature._base import FEATURE, TOKEN
        _kinds = {_dg: FEATURE for _dg in self._global_group_dims()}
        _kinds["radial_rle"] = TOKEN
        return _kinds

    def Tokens(self, mask: Tensor) -> Tensor:
        """``(B, 1, H, W)`` 마스크 -> ``(B, seq, token_dim)`` **병합 토큰** (학습 헤더 호환용).

        :meth:`Features` 의 도메인별 출력을 max 차원으로 패딩해 한 격자로 합친 것 — transformer 헤더가
        고정 ``token_dim`` 시퀀스를 먹기 때문이다. **검증은 이 병합을 쓰지 않는다**(도메인별 스케일이
        패딩으로 뭉개진다) — :meth:`Features` 를 도메인별로 본다. 학습이 조립을 가져가면 이 메서드는 걷는다.
        """
        _feats = self.Features(mask)
        _tokens = [self._pad(_feats[_dg]) for _dg in self._global_group_dims()]   # 전역 토큰 (순서 고정)
        _ang = _feats["radial_rle"]
        if _ang.shape[-1] < self._token_dim:
            _ang = F.pad(_ang, (0, self._token_dim - _ang.shape[-1]))
        return torch.cat([*_tokens, _ang], dim=1)                 # (B, α+NT, token_dim)

    def forward(self, mask: Tensor) -> Tensor:
        """
        Args:
            mask: (B, 1, H, W) float — 전경 1 / 배경 0.

        Returns:
            (B, seq, token_dim) float — **raw** 토큰(정규화 전, 절대 크기 보존).
        """
        return self.Tokens(mask)
