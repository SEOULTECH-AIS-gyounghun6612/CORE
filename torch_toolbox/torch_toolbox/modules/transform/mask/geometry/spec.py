from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

"""feature 선언과 정규화.

파이프라인 순서를 고정한다::

    descriptor (raw)  ->  transform (log1p / slog, 그룹별 고정)  ->  normalize (offset, scale)

- descriptor 모듈은 **raw 값만** 낸다. 정규화를 계산식 안에 섞지 않는다.
- ``transform`` 은 자릿수 압축이라 **의미의 일부**다. ``area`` 처럼 몇 자릿수를 오가는 양은
  선형 정규화만 하면 좁은 구간에 뭉친다. 교체 대상이 아니다.
- ``normalize`` 만 교체 대상. ``(x - offset) / scale`` 을 한 번 적용하고, 그 상수를
  **buffer 로 들고 있어 export 직전에 갈아끼울 수 있다.**

운용: 이론값으로 먼저 뽑아 실제 분포를 보고, 필요하면 그 통계로 buffer 를 채워 다시
export 한다. 어느 쪽이든 **그래프 구조는 동일**하다.

``FEAT_DIM`` 과 그룹 슬라이스를 하드코딩하지 않는 이유: 기존 ``sample_extractor`` 는
``FEAT_DIM = 426`` 과 ``FEAT_GROUPS`` 슬라이스가 리터럴이라 ``num_angles`` 를 바꾸면
조용히 어긋났다. 여기서는 모듈이 자기 spec 을 선언하고 조립기가 합산한다.
"""


@dataclass(frozen=True)
class Feature_Spec:
    """feature 그룹 하나의 선언.

    Attributes:
        name: 그룹 이름.
        dim: 차원 수.
        transform: ``identity`` | ``log1p`` | ``slog``.
        value_range: **transform 적용 후** 범위 ``(lo, hi)``.
    """

    name: str
    dim: int
    transform: str
    value_range: tuple[float, float]


def Apply_transform(x: Tensor, kind: str) -> Tensor:
    """그룹별 자릿수 압축. ``slog`` 는 부호를 보존한다."""
    if kind == "identity":
        return x
    if kind == "log1p":
        return torch.log1p(x.clamp_min(0.0))
    if kind == "slog":
        return torch.sign(x) * torch.log1p(x.abs())
    raise ValueError(f"알 수 없는 transform: {kind}")


class Normalizer(nn.Module):
    """선언된 spec 에서 ``(offset, scale)`` 을 만들어 마지막에 한 번 적용한다.

    기본값은 각 spec 의 **이론 범위**에서 나온다. ``Set_statistics`` 로 데이터 측정
    통계(평균·표준편차)로 교체할 수 있으며, 그래프 구조는 바뀌지 않는다.

    Args:
        specs: 이어붙일 순서대로의 :class:`Feature_Spec`.
    """

    def __init__(self, specs: tuple[Feature_Spec, ...]) -> None:
        super().__init__()
        self.specs = specs

        _off, _scl = [], []
        for _s in specs:
            _lo, _hi = _s.value_range
            _mid = (_lo + _hi) / 2.0
            _half = max((_hi - _lo) / 2.0, 1e-6)
            _off += [_mid] * _s.dim
            _scl += [_half] * _s.dim
        self.register_buffer("offset", torch.tensor(_off, dtype=torch.float32), persistent=False)
        self.register_buffer("scale",  torch.tensor(_scl, dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def Set_statistics(self, mean: Tensor, std: Tensor) -> None:
        """정규화 상수를 데이터 측정 통계로 교체한다. export 직전에 호출한다.

        Args:
            mean: (FEAT_DIM,) — transform 적용 **후** 값의 평균.
            std:  (FEAT_DIM,) — 같은 값의 표준편차.
        """
        if mean.shape != self.offset.shape or std.shape != self.scale.shape:
            raise ValueError(
                f"차원 불일치: spec {tuple(self.offset.shape)} vs "
                f"mean {tuple(mean.shape)} / std {tuple(std.shape)}"
            )
        self.offset.copy_(mean.to(self.offset))
        self.scale.copy_(std.to(self.scale).clamp_min(1e-6))

    def forward(self, x: Tensor) -> Tensor:
        """(B, FEAT_DIM) raw(=transform 적용 후) -> 정규화."""
        return (x - self.offset) / self.scale
