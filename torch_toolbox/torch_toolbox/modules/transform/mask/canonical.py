from __future__ import annotations
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from torch import Tensor

from .... import CFGS
from ... import MODELS
from ...definition import Composable_Config
from ...model.definition import Trainable_Model

"""정준 좌표계(frame) 산출 — **파라미터만 낸다. raster를 돌리지 않는다.**

기존 :func:`dataloader.classification.mask_sdf_edge.align_pca` 는 PCA 주축각을 구한 뒤
``cv2.warpAffine`` 으로 마스크를 실제로 회전시켰다. 이 모듈은 각도를 **값으로만** 내고,
회전은 하류(:class:`~torch_toolbox.modules.transform.mask.polar.Polar_Raster`)에서 theta축 roll 로 소비한다.

그 결과:
    - ``warpAffine`` 이 기하 분기에서 사라진다 (NEAREST 리샘플 손실·cv2/torch 불일치 제거)
    - ``_recenter_to_centroid`` 의 정수 시프트 + **클리핑**(밀려난 픽셀을 테두리에 쌓아
      없는 경계를 만들던 결함)이 사라진다
    - 극좌표와 해석적 측정이 **같은 원점**을 쓴다 (기존엔 극좌표=정수 반올림 centroid,
      chirality/moments=float centroid 로 원점이 둘이었다)

원점은 **정수로 양자화**한다. 이 결정이 하류의 bilinear 가중치를 상수로 만들어
``GridSample`` 없이 ``Gather``/``Mul``/``Add`` 만으로 극좌표 리샘플이 되게 한다.
sub-pixel 원점으로 바꾸면 그 상수성이 깨진다.
"""


class Frame(NamedTuple):
    """정준 좌표계 파라미터.

    Attributes:
        center: (B, 2) float — centroid ``(cx, cy)``. 원본 픽셀 좌표, sub-pixel.
        origin: (B, 2) long  — 정수 양자화한 극좌표 원점 ``(col, row)``.
        angle:  (B,)  float  — 주축각(rad). 180° 모호성 확정 완료.
    """

    center: Tensor
    origin: Tensor
    angle:  Tensor


_FRAME_NAME = "centroid_frame"
_FRAME_CFG  = f"{_FRAME_NAME}_Config"
_COORD_NAME = "frame_coords"
_COORD_CFG  = f"{_COORD_NAME}_Config"


@CFGS.Register_module(_FRAME_CFG)
@dataclass
class Centroid_Frame_Config(Composable_Config):
    """정준 좌표계 산출 설정.

    Attributes:
        size: 캔버스 ``(H, W)``.
    """
    config_type: str = _FRAME_CFG
    object_type: str = _FRAME_NAME
    trainable: bool = False
    size: tuple[int, int] = (224, 224)


@MODELS.Register_module(_FRAME_NAME)
class Centroid_Frame(Trainable_Model):
    """이진 마스크 → centroid + PCA 주축각.

    주축각은 2x2 공분산의 고유분해 닫힌 해로 구한다 (``eigh`` 불필요)::

        angle = 0.5 * atan2(2 * Sxy, Sxx - Syy)

    주축은 선(90° 주기)이라 방위가 4겹 모호하다. {θ, θ+90, θ+180, θ+270} 중 **BR 사분면
    (+u=우측, +v=하단)의 질량이 최대**가 되는 회전을 고른다 — 네 후보가 물체의 네 사분면을
    차례로 BR 에 놓으므로, 이는 곧 **최대 질량 사분면을 BR 로 보내는** 회전이다. 반사가 아니라
    회전이라 좌우/카이랄 정보가 보존된다(거울상 구분 신호는 상하 비대칭으로 남는다).

    좌우 질량 부호로 180° 만 확정하던 방식보다 근대칭 형상에서 안정적이다 — 주축이 거의
    불안정한(``Sxx≈Syy``) 경우에도 4겹 방위가 하나로 굳는다. 대신 u 가 major 축이라는 보장이
    사라지므로(θ±90 후보), major/minor 크기를 쓰는 :class:`Region_Scalars` 는 축 이름이 아니라
    **고유값 크기순**으로 뽑아야 한다(둘은 한 벌로 바뀐다).

    Note:
        입력 마스크는 fill/최대연결성분 처리를 **하지 않은** 원본이어야 한다.
        관통 구멍은 형상 정보이므로 메우지 않는다.
    """

    _dx: Tensor
    _dy: Tensor

    def Out_channels(self) -> list[int]:
        """center (2) / origin (2) / angle (1) — 출력 텐서별 성분 수."""
        return [2, 2, 1]

    def Build(self, size: tuple[int, int] = (224, 224), **kwargs: Any) -> None:
        _h, _w = int(size[0]), int(size[1])
        self.size = (_h, _w)

        # 픽셀 중심 좌표. 캔버스 기하중심 기준 오프셋으로 둔다.
        _ys = torch.arange(_h, dtype=torch.float32) - (_h - 1) / 2.0
        _xs = torch.arange(_w, dtype=torch.float32) - (_w - 1) / 2.0
        self.register_buffer("_dy", _ys.view(1, _h, 1), persistent=False)
        self.register_buffer("_dx", _xs.view(1, 1, _w), persistent=False)

    def forward(self, mask: Tensor) -> Frame:
        """
        Args:
            mask: (B, 1, H, W) float. 전경 1, 배경 0.

        Returns:
            :class:`Frame`.
        """
        _m = mask[:, 0]                                        # (B, H, W)
        _n = _m.sum(dim=(1, 2)).clamp_min(1.0)                 # (B,) 전경 픽셀 수

        # centroid — 캔버스 기하중심 기준 오프셋
        _cx = (_m * self._dx).sum(dim=(1, 2)) / _n             # (B,)
        _cy = (_m * self._dy).sum(dim=(1, 2)) / _n

        # centroid 기준 중심 모멘트 (2차)
        _ddx = self._dx - _cx.view(-1, 1, 1)                   # (B, H, W)
        _ddy = self._dy - _cy.view(-1, 1, 1)
        _sxx = (_m * _ddx * _ddx).sum(dim=(1, 2)) / _n
        _syy = (_m * _ddy * _ddy).sum(dim=(1, 2)) / _n
        _sxy = (_m * _ddx * _ddy).sum(dim=(1, 2)) / _n

        _angle = 0.5 * torch.atan2(2.0 * _sxy, _sxx - _syy)    # (B,) **주축(major) 각** (선, 180° 주기)

        # 2-fold 확정: {θ, θ+180} 중 BR 사분면(+u 우측·+v 하단) 질량이 큰 쪽. u축은 **항상 major
        # 축**이고(θ±90 = 부축 후보를 넣지 않는다 — 그게 90° 회전의 원인이었다), major 선의 180°
        # 방향 모호성만 BR 질량으로 가른다. 비대칭 부품은 여기서 방향이 유일해지고(작은 비대칭이
        # 노이즈만 이기면 됨 — 인접 사분면과 경쟁 안 함), 대칭 부품은 BR 질량이 같아 180° 가 남는다
        # (원리적 — 대칭 객체는 실루엣이 180° 돌아도 같으므로 문제없다).
        _br = []
        for _off in (0.0, torch.pi):
            _cos = torch.cos(_angle + _off).view(-1, 1, 1)
            _sin = torch.sin(_angle + _off).view(-1, 1, 1)
            _u =  _ddx * _cos + _ddy * _sin                    # +u 우측
            _v = -_ddx * _sin + _ddy * _cos                    # +v 하단 (row 증가)
            _br.append((_m * ((_u > 0) & (_v > 0)).to(_m.dtype)).sum(dim=(1, 2)))
        _k = torch.stack(_br, dim=1).argmax(dim=1)             # (B,) 0/1 — BR 질량 큰 방향
        _angle = _angle + _k.to(_angle.dtype) * torch.pi

        _h, _w = self.size
        _center = torch.stack([_cx + (_w - 1) / 2.0, _cy + (_h - 1) / 2.0], dim=1)
        _origin = _center.round().long()                       # 정수 양자화 (col, row)
        return Frame(center=_center, origin=_origin, angle=_angle)


@CFGS.Register_module(_COORD_CFG)
@dataclass
class Frame_Coords_Config(Composable_Config):
    """정렬 좌표 ``(u, v)`` 산출 설정.

    Attributes:
        size: 캔버스 ``(H, W)``.
    """
    config_type: str = _COORD_CFG
    object_type: str = _COORD_NAME
    trainable: bool = False
    size: tuple[int, int] = (224, 224)


@MODELS.Register_module(_COORD_NAME)
class Frame_Coords(Trainable_Model):
    """centroid 원점·주축 정렬 좌표 ``(u, v)`` 를 낸다 — raster 회전 없이 해석적으로.

    ``u`` 는 주축(장축) 방향, ``v`` 는 그 수직 방향. 기존 구현은 마스크를 실제로 회전시킨 뒤
    image frame 에서 측정했지만, 여기서는 좌표만 돌린다. NEAREST 리샘플 손실이 없고
    극좌표 분기와 **같은 원점(centroid)** 을 쓴다.

    좌표는 ``norm`` 으로 나눠 [-1, 1] 근방에 두고 낸다. 이것이 FP16 안전성의 핵심이다 —
    정규화 없이 3차 모멘트를 만들면 ``dx^3 = 112^3 ~= 1.4e6`` 으로 FP16 최대값(65504)을
    단일 항에서 넘긴다.
    """

    _dx: Tensor
    _dy: Tensor

    def Out_channels(self) -> list[int]:
        """u, v 두 텐서."""
        return [1, 1]

    def Build(self, size: tuple[int, int] = (224, 224), **kwargs: Any) -> None:
        _h, _w = int(size[0]), int(size[1])
        self.size = (_h, _w)
        self.norm = float(torch.tensor([(_h - 1) / 2.0, (_w - 1) / 2.0]).square().sum().sqrt())

        _ys = torch.arange(_h, dtype=torch.float32) - (_h - 1) / 2.0
        _xs = torch.arange(_w, dtype=torch.float32) - (_w - 1) / 2.0
        self.register_buffer("_dy", _ys.view(1, _h, 1), persistent=False)
        self.register_buffer("_dx", _xs.view(1, 1, _w), persistent=False)

    def forward(self, frame: Frame) -> tuple[Tensor, Tensor]:
        """
        Args:
            frame: :class:`Frame`.

        Returns:
            ``(u, v)`` — 각각 (B, H, W) float. ``norm`` 으로 나눈 무차원 좌표.
        """
        _h, _w = self.size
        _cx = frame.center[:, 0].view(-1, 1, 1) - (_w - 1) / 2.0
        _cy = frame.center[:, 1].view(-1, 1, 1) - (_h - 1) / 2.0
        _ddx = (self._dx - _cx) / self.norm
        _ddy = (self._dy - _cy) / self.norm

        _cos = torch.cos(frame.angle).view(-1, 1, 1)
        _sin = torch.sin(frame.angle).view(-1, 1, 1)
        return _ddx * _cos + _ddy * _sin, -_ddx * _sin + _ddy * _cos
