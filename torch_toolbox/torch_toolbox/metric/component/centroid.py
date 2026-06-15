from __future__ import annotations
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..definition import Accumulator
from .. import ACCUMULATORS


@ACCUMULATORS.Register_module("centroid_accumulator")
class Centroid_Accumulator(Accumulator):
    """키 기준 그룹 평균 벡터를 스트리밍 누적함. 전체 벡터 미적재(O(K) 메모리).

    batch마다 Update로 키별 벡터 합·표본수를 적층하고, Finalize에서 평균을 확정함.
    합은 교환·결합 법칙을 만족하므로 batch 도착 순서와 무관함.

    normalize:
        True  : 합산 전 각 벡터를 단위구에 투영(1표 동등, 구면 평균 방향).
        False : raw 벡터 합산(magnitude 가중 centroid).
    """

    def __init__(self, normalize: bool = True) -> None:
        self._normalize = normalize
        self._sums: dict[int, Tensor] = {}
        self._counts: dict[int, int] = {}

    def Update(self, *, embeddings: Tensor, gt_class_id: Tensor, **kwargs: Any) -> None:
        """(N, D) 임베딩과 (N,) 클래스 id를 키별로 합산 누적함."""
        _vec = F.normalize(embeddings, dim=-1) if self._normalize else embeddings
        for _k in gt_class_id.unique():
            _kid = int(_k)
            _mask = gt_class_id == _k
            _v = _vec[_mask].sum(dim=0)
            if _kid in self._sums:
                self._sums[_kid] += _v
                self._counts[_kid] += int(_mask.sum())
            else:
                self._sums[_kid] = _v
                self._counts[_kid] = int(_mask.sum())

    def Finalize(self) -> tuple[Tensor, Tensor, Tensor]:
        """누적 확정 → (means (K, D), class_ids (K,), counts (K,)). ids 오름차순."""
        _ids = sorted(self._sums)
        return (
            torch.stack([self._sums[_k] / self._counts[_k] for _k in _ids], dim=0),
            torch.tensor(_ids, dtype=torch.long),
            torch.tensor([self._counts[_k] for _k in _ids], dtype=torch.long),
        )

    def Reset(self) -> None:
        self._sums.clear()
        self._counts.clear()
