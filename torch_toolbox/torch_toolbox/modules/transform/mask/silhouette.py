from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .... import CFGS
from ... import MODELS
from ...definition import Composable_Config
from ...model.definition import Trainable_Model

"""실루엣 전단 — 원본 마스크 -> 정준 캔버스 이진 실루엣. **geometry/image 분기의 입력.**

기존 numpy/cv2 사슬(`module.utils.crop.Crop_centered` + `dataloader.utils.silhouette.
resize_and_binarize`)의 torch 등가다. 학습 dataset(`_prepare_binary`)·추론·평가가 이 사슬을
따로 재구현하지 않고 **같은 그래프**를 조립해 쓰도록 여기로 올린다 — 그러면 "같은 연산"이
라이브러리 버전 고정이 아니라 그래프로 보장된다(전처리 non-split 원칙).

사슬(annToMask 등 소스별 이진화는 제외 — 데이터측 I/O)::

    (원본 이진 마스크, 중심)
      -> Center_Crop      (bbox 중심 고정 창, 경계 밖 0)
      -> Resize_Binarize  (target 캔버스로 bilinear 리샘플 후 재이진화)
      -> (정렬 전 {0,1} 실루엣) -> geometry / image 분기

정렬은 하지 않는다 — geometry 는 좌표계로(Centroid_Frame), image 는 Align_Raster 로 각자
정준화한다. 여기서는 "어느 창을, 어느 해상도로"만 정한다.
"""


_CROP_NAME = "center_crop"
_CROP_CFG  = f"{_CROP_NAME}_Config"


@CFGS.Register_module(_CROP_CFG)
@dataclass
class Center_Crop_Config(Composable_Config):
    """중심점 기준 정사각 크롭 설정.

    Attributes:
        crop_size: 크롭 창 한 변(px, 원본 스케일). **추론 CropSilhouette 와 같아야 한다.**
    """
    config_type: str = _CROP_CFG
    object_type: str = _CROP_NAME
    trainable: bool = False
    crop_size: int = 280


@MODELS.Register_module(_CROP_NAME)
class Center_Crop(Trainable_Model):
    """이진 마스크 + 중심점 ``(cx, cy)`` -> ``(B, 1, S, S)`` 정사각 창. 경계 밖 0.

    numpy ``Crop_centered`` 의 torch 등가다. **정수 창**이라 보간 없이 gather 로 읽고 범위
    밖은 0 으로 둔다(``Polar_Raster._sample`` 과 같은 패턴). 입력 캔버스 크기는 build 상수가
    아니라 ``forward`` 에서 읽으므로 프레임 해상도가 달라도 된다(원본 전 프레임에서 크롭).

    bbox 중심 기준 **고정 창**이라 여백이 있어, 뒤이은 정렬/리샘플에도 형태가 잘려 나가지
    않는다. 중심은 원본 좌표 ``(col, row)`` 로, ``Centroid_Frame.origin`` 과 같은 규약이다.
    """

    def Build(self, crop_size: int = 280, **kwargs: Any) -> None:
        self.crop_size = int(crop_size)

    def Out_channels(self) -> list[int]:
        """크롭만 하므로 채널 수는 그대로 1."""
        return [1]

    def forward(self, mask: Tensor, center: Tensor) -> Tensor:
        """
        Args:
            mask: (B, 1, H, W) float — 전경 1 / 배경 0. 원본 스케일.
            center: (B, 2) long — 창 중심 ``(cx, cy)`` = (col, row), 원본 좌표.

        Returns:
            (B, 1, S, S) float — ``S = crop_size``. mask 값 스케일 유지({0,1}이면 {0,1}).
        """
        _b = mask.shape[0]
        _h, _w = int(mask.shape[-2]), int(mask.shape[-1])
        _s = self.crop_size
        _flat = mask.reshape(_b, -1)

        # 창 좌상단 기준 정수 오프셋 (음수 가능 — 경계 밖은 valid 로 걸러 0).
        _off = torch.arange(_s, device=mask.device) - _s // 2          # (S,)
        _cols = center[:, 0].view(_b, 1, 1) + _off.view(1, 1, _s)       # (B, 1, S)
        _rows = center[:, 1].view(_b, 1, 1) + _off.view(1, _s, 1)       # (B, S, 1)
        _cols = _cols.expand(_b, _s, _s)
        _rows = _rows.expand(_b, _s, _s)

        _valid = (_rows >= 0) & (_rows < _h) & (_cols >= 0) & (_cols < _w)
        _idx = _rows.clamp(0, _h - 1) * _w + _cols.clamp(0, _w - 1)     # (B, S, S)
        _out = _flat.gather(1, _idx.reshape(_b, -1)) * _valid.reshape(_b, -1).to(mask.dtype)
        return _out.reshape(_b, 1, _s, _s)


_RB_NAME = "resize_binarize"
_RB_CFG  = f"{_RB_NAME}_Config"


@CFGS.Register_module(_RB_CFG)
@dataclass
class Resize_Binarize_Config(Composable_Config):
    """리샘플 후 재이진화 설정.

    Attributes:
        target_size: 출력 캔버스 ``(H, W)``. geometry ``size`` 와 같아야 한다.
        thresh: 재이진화 임계값. 입력이 {0,1} 스케일이면 0.5.
    """
    config_type: str = _RB_CFG
    object_type: str = _RB_NAME
    trainable: bool = False
    target_size: tuple[int, int] = (224, 224)
    thresh: float = 0.5


@MODELS.Register_module(_RB_NAME)
class Resize_Binarize(Trainable_Model):
    """``(B, 1, H, W)`` {0,1} 마스크 -> target 캔버스 bilinear 리샘플 -> ``> thresh`` 재이진화.

    numpy ``resize_and_binarize`` 의 torch 등가다. 보간(bilinear, ``align_corners=False`` =
    cv2 half-pixel 규약에 대응)과 임계값의 **단일 출처**. 보간이 만드는 경계 중간값을 임계로
    다시 이진화하므로 입력·임계 스케일이 같아야 한다({0,1} -> ``thresh`` 0.5).
    """

    def Build(
        self,
        target_size: tuple[int, int] = (224, 224),
        thresh: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.thresh = float(thresh)

    def Out_channels(self) -> list[int]:
        """리샘플만 하므로 채널 수는 그대로 1."""
        return [1]

    def forward(self, mask: Tensor) -> Tensor:
        """
        Args:
            mask: (B, 1, H, W) float — {0,1} 스케일.

        Returns:
            (B, 1, target_H, target_W) float — 재이진화 {0,1}.
        """
        _r = F.interpolate(mask, size=self.target_size, mode="bilinear", align_corners=False)
        return (_r > self.thresh).to(mask.dtype)
