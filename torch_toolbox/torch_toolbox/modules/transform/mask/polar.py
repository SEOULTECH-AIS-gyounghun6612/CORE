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

from .canonical import Frame

"""직교 -> 극좌표 리샘플 (**backward gather**).

방향이 중요하다. 픽셀마다 ``(r, theta)`` 셀을 찾아 쏘는 **forward scatter** 는 두 방향으로
깨진다 (224 캔버스, 256 theta bin 실측):

    안쪽 (r < 41px, 반경의 26%)  : 빈 셀 51% 가 **가짜 구멍** — 아무도 안 온 셀과
                                   재료가 없는 셀이 구분되지 않는다
    바깥쪽 (셀 폭 3.85px)         : 여러 픽셀이 한 셀에 뭉쳐 **진짜 구멍이 흡수**된다

셀마다 자기 좌표에서 마스크를 읽는 **backward gather** 는 모든 셀이 값을 받으므로
가짜 구멍이 원리적으로 없다. 바깥쪽 흡수는 occupancy 를 boolean 이 아니라 **면적 분수**로
두어 완화한다 (얇은 구멍이 사라지는 대신 분수를 떨어뜨려 흔적을 남긴다). ``sub`` 가 그
민감도의 손잡이이며, 늘려도 **연산 종류는 그대로**고 상수 테이블만 커진다.

원점이 정수(:class:`~torch_toolbox.modules.transform.mask.canonical.Centroid_Frame` 가 양자화)이므로
샘플 위치의 소수부가 **샘플과 무관하게 고정**된다. 따라서 bilinear 가중치가 상수이고,
정수 인덱스만 원점만큼 이동한다 -> ``GridSample`` 없이 ``Gather``/``Mul``/``Add`` 로 끝난다.
회전도 raster 를 돌리지 않고 theta축 circular roll 로 소비한다.
"""


_NAME = "polar_raster"
_CFG  = f"{_NAME}_Config"


@CFGS.Register_module(_CFG)
@dataclass
class Polar_Raster_Config(Composable_Config):
    """극좌표 리샘플 설정.

    Attributes:
        sampling_size: 샘플 반경 기본값을 뽑는 기준 캔버스 ``(H, W)``. 입력 캔버스가 아니다.
        num_radial: r 방향 bin 수 ``NR``.
        num_angular: theta 방향 bin 수 ``NT``.
        sub: 셀당 축별 sub-sample 수. 셀당 표본은 ``sub**2``.
        r_max: 최대 반경(px). None 이면 ``sampling_size`` 반대각.
    """
    config_type: str = _CFG
    object_type: str = _NAME
    trainable: bool = False
    sampling_size: tuple[int, int] = (224, 224)
    num_radial: int = 224
    num_angular: int = 512
    sub: int = 1
    r_max: float | None = None


@MODELS.Register_module(_NAME)
class Polar_Raster(Trainable_Model):
    """이진 마스크 + :class:`Frame` -> ``(B, NR, NT)`` occupancy 분수.

    Args:
        sampling_size: 샘플 반경 기본값을 뽑는 기준 캔버스 ``(H, W)``. 입력 캔버스가 아니다.
        num_radial: r 방향 bin 수 ``NR``.
        num_angular: theta 방향 bin 수 ``NT``.
        sub: 셀당 축별 sub-sample 수. 셀당 표본은 ``sub**2`` 개.
        r_max: 최대 반경(px). None 이면 ``sampling_size`` 반대각.

    Note:
        상수 버퍼는 ``persistent=False`` 라 state_dict 에는 안 들어가지만,
        **ONNX 에서는 initializer 로 구워진다.** 즉 그래프가 이 설정에 고정되며
        ``NR``/``NT``/``sub``/``r_max`` 를 바꾸려면 재export 다.

        런타임 메모리는 ``B * NR * NT * sub**2`` 에 비례한다. 학습은 DataLoader worker 에서
        샘플 단위(B=1)로 돌므로 문제가 없으나, 큰 배치로 GPU 에서 한번에 돌릴 때는
        중간 텐서 크기를 확인해야 한다.
    """

    _base_col: Tensor
    _base_row: Tensor
    _frac_x: Tensor
    _frac_y: Tensor
    _theta_idx: Tensor

    def Out_channels(self) -> list[int]:
        """occupancy 격자 하나. 채널 축이 없으므로 r bin 수를 낸다."""
        return [self.num_radial]

    def Build(
        self,
        sampling_size: tuple[int, int] = (224, 224),
        num_radial: int = 224,
        num_angular: int = 512,
        sub: int = 1,
        r_max: float | None = None,
        **kwargs: Any,
    ) -> None:
        _h, _w = int(sampling_size[0]), int(sampling_size[1])
        self.sampling_size = (_h, _w)
        self.num_radial = int(num_radial)
        self.num_angular = int(num_angular)
        self.sub = int(sub)

        _rmax = math.hypot((_h - 1) / 2.0, (_w - 1) / 2.0) if r_max is None else float(r_max)
        self.r_max = _rmax
        self.dr = _rmax / self.num_radial
        self.dtheta = 2.0 * math.pi / self.num_angular

        # 셀 중심 정렬 + 셀 내부 sub x sub 격자
        _i = torch.arange(self.num_radial, dtype=torch.float64)
        _j = torch.arange(self.num_angular, dtype=torch.float64)
        _a = (torch.arange(self.sub, dtype=torch.float64) + 0.5) / self.sub
        _r = (_i.view(-1, 1, 1, 1) + _a.view(1, 1, -1, 1)) * self.dr
        _t = (_j.view(1, -1, 1, 1) + _a.view(1, 1, 1, -1)) * self.dtheta - math.pi

        # 원점 기준 오프셋. 원점이 정수라 이 소수부가 곧 bilinear 가중치가 된다.
        _ox = _r * torch.cos(_t)                       # (NR, NT, sub, sub)
        _oy = _r * torch.sin(_t)
        _c0 = torch.floor(_ox)
        _r0 = torch.floor(_oy)

        self.register_buffer("_base_col", _c0.reshape(-1).to(torch.int32), persistent=False)
        self.register_buffer("_base_row", _r0.reshape(-1).to(torch.int32), persistent=False)
        self.register_buffer("_frac_x", (_ox - _c0).reshape(-1).to(torch.float32), persistent=False)
        self.register_buffer("_frac_y", (_oy - _r0).reshape(-1).to(torch.float32), persistent=False)
        self.register_buffer("_theta_idx", torch.arange(self.num_angular), persistent=False)

    def _sample(self, flat: Tensor, row: Tensor, col: Tensor, h: int, w: int) -> Tensor:
        """``(B, H*W)`` 마스크에서 정수 격자점을 읽는다. 범위 밖은 0.

        캔버스 크기는 build 상수가 아니라 인자로 받는다 — 샘플 반경은 ``r_max`` 가 정해
        입력 크기와 무관하므로, 크롭 없이 원본 프레임을 그대로 넣을 수 있다
        (``Center_Crop`` 과 같은 규약).
        """
        # stride 가 틀어지면 유효 인덱스 범위 안에서 엉뚱한 화소를 읽어 조용히 틀린다.
        # 채널이 1이 아닌 입력도 여기서 걸린다.
        if flat.shape[-1] != h * w:
            raise ValueError(
                f"flat 길이 {flat.shape[-1]} != h*w ({h}*{w}={h * w}). "
                f"입력이 (B, 1, H, W) 인지 확인할 것."
            )
        _valid = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        _idx = row.clamp(0, h - 1) * w + col.clamp(0, w - 1)
        return flat.gather(1, _idx) * _valid.to(flat.dtype)

    def forward(self, mask: Tensor, frame: Frame) -> Tensor:
        """
        Args:
            mask: (B, 1, H, W) float. 전경 1, 배경 0.
            frame: :class:`Frame`. ``origin`` 과 ``angle`` 을 쓴다.

        Returns:
            (B, NR, NT) float — 셀별 occupancy 분수 [0, 1]. theta 는 주축 정렬 완료.
        """
        _b = mask.shape[0]
        _h, _w = mask.shape[-2], mask.shape[-1]
        _flat = mask.reshape(_b, -1)

        _oc = frame.origin[:, 0].view(-1, 1)                   # (B, 1) col
        _or = frame.origin[:, 1].view(-1, 1)                   # (B, 1) row
        _c0 = self._base_col.view(1, -1).long() + _oc          # (B, P)
        _r0 = self._base_row.view(1, -1).long() + _or

        _fx = self._frac_x.view(1, -1)
        _fy = self._frac_y.view(1, -1)

        _v00 = self._sample(_flat, _r0,     _c0,     _h, _w)
        _v01 = self._sample(_flat, _r0,     _c0 + 1, _h, _w)
        _v10 = self._sample(_flat, _r0 + 1, _c0,     _h, _w)
        _v11 = self._sample(_flat, _r0 + 1, _c0 + 1, _h, _w)

        _val = (
            _v00 * ((1.0 - _fy) * (1.0 - _fx))
            + _v01 * ((1.0 - _fy) * _fx)
            + _v10 * (_fy * (1.0 - _fx))
            + _v11 * (_fy * _fx)
        )

        _s = self.sub * self.sub
        _val = _val.view(_b, self.num_radial, self.num_angular, _s).mean(dim=3)

        # 회전은 raster 를 돌리지 않고 theta축 circular roll 로 소비한다.
        _roll = torch.round(frame.angle / self.dtheta).long().view(-1, 1)   # (B, 1)
        _jdx = (self._theta_idx.view(1, -1) + _roll) % self.num_angular     # (B, NT)
        _jdx = _jdx.unsqueeze(1).expand(-1, self.num_radial, -1)
        return _val.gather(2, _jdx)
