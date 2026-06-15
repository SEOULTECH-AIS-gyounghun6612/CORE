from __future__ import annotations
from typing import Any

import torch

from ..definition import Accumulator
from .. import ACCUMULATORS


@ACCUMULATORS.Register_module("scalar_accumulator")
class Scalar_Accumulator(Accumulator):
    """(value, count) 스칼라 쌍을 가중평균으로 누적하는 범용 누적기.

    _Forward 출력에서 (int|float, int|float) 2-튜플인 키만 자동으로 선택함.
    나머지 키(Tensor 등)는 무시됨.
    """

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, float] = {}

    def Update(self, **output: Any) -> None:
        """(value, count) 2-튜플인 키만 선택해 가중합으로 누적한다.

        Args:
            **output: _Forward 전체 출력. (int|float, int|float) 튜플 외의 값은 무시.
        """
        for _k, _v in output.items():
            # (value, count) 2-튜플만 처리; Tensor 등 다른 타입은 skip
            if not (
                isinstance(_v, tuple) and len(_v) == 2
                and all(isinstance(_x, (int, float)) for _x in _v)
            ):
                continue
            _val, _cnt = float(_v[0]), float(_v[1])
            self._sums[_k] = self._sums.get(_k, 0.0) + _val * _cnt
            self._counts[_k] = self._counts.get(_k, 0.0) + _cnt

    def Finalize(self) -> dict[str, float]:
        """누적된 (weighted_sum / total_count) 가중평균을 반환한다.

        count가 0인 키는 제외한다.

        Returns:
            {metric_name: weighted_average} 형태의 dict.
        """
        return {
            _k: self._sums[_k] / self._counts[_k]
            for _k in self._sums
            if self._counts[_k] > 0
        }

    def Is_diverged(self) -> bool:
        """누적된 스칼라 중 NaN/Inf가 있으면 True."""
        return any(
            not torch.isfinite(torch.tensor(_v))
            for _v in self.Finalize().values()
        )

    def Reset(self) -> None:
        self._sums.clear()
        self._counts.clear()
