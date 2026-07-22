from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ..... import CFGS
from .... import MODELS
from ....definition import Composable_Config
from ....model.definition import Trainable_Model

from .spec import Feature_Spec
from ..occupancy import Region_Profile

"""크기·비율 스칼라 — skimage ``regionprops`` 없이.

기존 ``_region_scalars`` 는 ``skimage.measure.regionprops`` 에 의존했고, 그 중 두 항목이
ONNX 로 넘어가지 않았다.

``area_convex`` (-> ``solidity``) — **convex hull. 표현 불가.**

    대체: ``area_swept = 반음적분(0.5 * r_outer^2 dtheta)`` 로 외곽이 감싸는 면적을 구하고
    ``fill_ratio = area / area_swept`` 를 쓴다. "자기 외곽을 얼마나 채우는가"라는 의미역이
    같으면서 이미 계산해 둔 radial profile 만으로 나오고, **구멍에 민감**하다
    (convex hull 은 구멍을 무시했다).

``perimeter`` — skimage 는 3x3 이웃 코드별 가중 추정을 쓴다. 재현은 가능하나 버전에 따라
    구현이 바뀌어 온 함수다.

    대체: morphological gradient(dilate - erode)의 전경 화소 수. skimage 와 값은 다르지만
    자기일관적이고, 구멍을 보존하므로 **구멍 둘레도 올바르게 센다**.

나머지(area, bbox, major/minor)는 전부 리덕션과 2x2 닫힌 해라 그대로 옮겨진다.
측정은 :class:`~torch_toolbox.modules.transform.mask.canonical.Frame_Coords` 가 주는 정렬 좌표에서 한다 —
raster 를 돌리지 않으므로 회전 리샘플 손실이 없다.
"""

SIZE_NAMES  = ("area", "perimeter", "major", "minor", "bbox_u", "bbox_v", "area_swept")
RATIO_NAMES = ("bbox_aspect", "axis_ratio", "extent", "fill_ratio", "circularity")
POS_NAMES   = ("centroid_du", "centroid_dv")


def _safe_div(a: Tensor, b: Tensor) -> Tensor:
    """0 나눗셈을 0 으로 둔다 (원본 ``_d`` 와 같은 규약)."""
    return torch.where(b.abs() > 1e-12, a / b.clamp_min(1e-12), torch.zeros_like(a))


_REGION_SCALARS_NAME = "region_scalars"
_REGION_SCALARS_CFG  = f"{_REGION_SCALARS_NAME}_Config"


@CFGS.Register_module(_REGION_SCALARS_CFG)
@dataclass
class Region_Scalars_Config(Composable_Config):
    """크기·비율 스칼라 설정.

    Attributes:
        size: 캔버스 ``(H, W)``. 면적 상한과 좌표 정규화값 도출에 쓴다.
        num_angular: theta bin 수. ``area_swept`` 적분의 dtheta 도출.
    """
    config_type: str = _REGION_SCALARS_CFG
    object_type: str = _REGION_SCALARS_NAME
    trainable: bool = False
    size: tuple[int, int] = (224, 224)
    num_angular: int = 512


@MODELS.Register_module(_REGION_SCALARS_NAME)
class Region_Scalars(Trainable_Model):
    """크기 7 + 비율 5 + 위치 2 = 14차원 스칼라.

    Args:
        size: 캔버스 ``(H, W)``. 면적 상한과 좌표 정규화값을 도출한다.
        num_angular: theta bin 수. ``area_swept`` 적분의 dtheta 를 도출한다.
    """

    dim: int = len(SIZE_NAMES) + len(RATIO_NAMES) + len(POS_NAMES)

    def Out_channels(self) -> list[int]:
        return [self.dim]

    def Build(
        self, size: tuple[int, int] = (224, 224), num_angular: int = 512, **kwargs: Any
    ) -> None:
        _h, _w = int(size[0]), int(size[1])
        self.size = (_h, _w)
        # Frame_Coords 와 같은 정규화값 — 좌표가 [-1, 1] 이 되도록 나눈 값.
        self.norm = math.hypot((_h - 1) / 2.0, (_w - 1) / 2.0)
        self.dtheta = 2.0 * math.pi / int(num_angular)

    def Spec(self) -> tuple[Feature_Spec, ...]:
        """이론 범위 선언.

        면적·둘레는 픽셀 수에 유계이고, 길이는 캔버스 대각에 유계다. 비율은 원리적으로
        상한이 없어 실용 상한 10 으로 선언한다 — 정규화용이라 벗어나도 값이 잘리지 않고
        구간 밖으로 나갈 뿐이다.
        """
        _area_max = math.log1p(self.size[0] * self.size[1])
        _len_max = math.log1p(4.0 * self.norm)
        return (
            Feature_Spec("size",     len(SIZE_NAMES),  "log1p",    (0.0, max(_area_max, _len_max))),
            Feature_Spec("ratio",    len(RATIO_NAMES), "identity", (0.0, 10.0)),
            Feature_Spec("position", len(POS_NAMES),   "slog",     (-math.log1p(self.norm),
                                                                     math.log1p(self.norm))),
        )

    def forward(
        self, mask: Tensor, u: Tensor, v: Tensor, profile: Region_Profile
    ) -> Tensor:
        """
        Args:
            mask: (B, 1, H, W) float.
            u, v: (B, H, W) float — 정렬 좌표(무차원).
            profile: :class:`Region_Profile` — ``r_outer`` 를 쓴다.

        Returns:
            (B, 14) float — :data:`SIZE_NAMES` + :data:`RATIO_NAMES` + :data:`POS_NAMES` 순서.
        """
        _m = mask[:, 0]
        _n = _m.sum(dim=(1, 2)).clamp_min(1.0)
        _area = _m.sum(dim=(1, 2))

        # 둘레: morphological gradient 의 전경 화소 수 (구멍 둘레 포함)
        _dil = F.max_pool2d(mask, 3, stride=1, padding=1)
        _ero = -F.max_pool2d(-mask, 3, stride=1, padding=1)
        _perimeter = (_dil - _ero).sum(dim=(1, 2, 3))

        # 정렬 프레임 bbox — u, v 는 이미 주축 정렬이라 이것이 회전 bbox 다.
        _big = torch.full_like(u, 1e9)
        _hit = _m > 0
        _umin = torch.where(_hit, u, _big).amin(dim=(1, 2))
        _umax = torch.where(_hit, u, -_big).amax(dim=(1, 2))
        _vmin = torch.where(_hit, v, _big).amin(dim=(1, 2))
        _vmax = torch.where(_hit, v, -_big).amax(dim=(1, 2))
        _bu = (_umax - _umin) * self.norm
        _bv = (_vmax - _vmin) * self.norm

        # major/minor — u, v 가 주축이므로 공분산이 대각. skimage 규약(4*sqrt(lambda)).
        _lu = (_m * u * u).sum(dim=(1, 2)) / _n
        _lv = (_m * v * v).sum(dim=(1, 2)) / _n
        _major = 4.0 * _lu.clamp_min(0).sqrt() * self.norm
        _minor = 4.0 * _lv.clamp_min(0).sqrt() * self.norm

        # 외곽이 감싸는 면적 — convex hull 대신 radial profile 의 반음적분.
        _swept = 0.5 * (profile.r_outer * profile.r_outer).sum(dim=1) * self.dtheta

        _size = torch.stack([_area, _perimeter, _major, _minor, _bu, _bv, _swept], dim=1)
        _ratio = torch.stack(
            [
                _safe_div(_bu, _bv),
                _safe_div(_major, _minor),
                _safe_div(_area, _bu * _bv),
                _safe_div(_area, _swept),                       # fill_ratio (구 solidity)
                _safe_div(4.0 * math.pi * _area, _perimeter * _perimeter),
            ],
            dim=1,
        )
        # centroid 는 원점이므로, bbox 중심과의 오프셋이 곧 비대칭 신호다.
        _pos = torch.stack(
            [-(_umax + _umin) / 2.0 * self.norm, -(_vmax + _vmin) / 2.0 * self.norm], dim=1
        )
        return torch.cat([_size, _ratio, _pos], dim=1)
