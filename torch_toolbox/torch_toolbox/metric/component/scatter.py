from __future__ import annotations
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..definition import Accumulator
from .. import ACCUMULATORS


@ACCUMULATORS.Register_module("within_class_scatter_accumulator")
class Within_Class_Scatter_Accumulator(Accumulator):
    """클래스별 평균과 **클래스 내 공분산의 풀링**을 한 패스로 누적한다.

    클래스 간 산포를 빼고 클래스 **안**의 산포만 모은다. 전체 공분산을 모으면 클래스가
    떨어져 있다는 사실이 산포로 잡혀, 이것으로 화이트닝한 거리는 "클래스 안에서 얼마나
    전형적인가"를 재지 못한다.

    메모리는 공유 스캐터 `(D, D)` 하나와 클래스별 평균 `(K, D)` 다. 클래스마다 공분산을
    따로 들면 `K·D²` 이 되어 못 쓴다 — 풀링이 목적이므로 하나로 합치며 누적한다.

    **왜 단순 2차 모멘트 차가 아닌가.** `Σxxᵀ − Σ n_k μ_k μ_kᵀ` 로도 같은 값이 나오지만,
    클래스가 뭉쳐 있을수록 두 큰 값의 차가 되어 자리수가 상쇄된다(정규화 임베딩이면
    `Σxxᵀ` 의 trace 가 `N`, 결과의 trace 는 그보다 훨씬 작다). 여기서는 배치마다 배치
    평균 기준으로 중심화한 뒤 평균 차이를 보정해 합치므로(Chan 병합) 큰 값의 차가 생기지
    않는다. 누적은 `float64` 로 한다.

    합침은 교환·결합 법칙을 만족하므로 batch 도착 순서와 무관하다.

    normalize:
        True  : 합산 전 각 벡터를 단위구에 투영. 코사인 헤드의 참조 공간이 구면이므로
                그쪽에 쓸 때는 이 값이어야 한다.
        False : raw 벡터. 로짓이 벡터에서 직접 나오는 헤드(FC 등)용.
    """

    def __init__(self, normalize: bool = True) -> None:
        self._normalize = normalize
        self._means: dict[int, Tensor] = {}
        self._counts: dict[int, int] = {}
        self._scatter: Tensor | None = None

    def Update(self, *, embeddings: Tensor, gt_class_id: Tensor, **kwargs: Any) -> None:
        """(N, D) 임베딩과 (N,) 클래스 id 를 클래스별로 합쳐 넣는다."""
        _vec = F.normalize(embeddings, dim=-1) if self._normalize else embeddings
        _vec = _vec.detach().double()
        if self._scatter is None:
            _d = _vec.shape[1]
            self._scatter = torch.zeros(_d, _d, dtype=torch.float64, device=_vec.device)

        for _k in gt_class_id.unique():
            _kid = int(_k)
            _batch = _vec[gt_class_id == _k]
            _nb = _batch.shape[0]
            _mb = _batch.mean(dim=0)

            # 배치 안의 중심화 스캐터 — 큰 값의 차가 아니라 잔차의 합이다.
            _resid = _batch - _mb
            self._scatter += _resid.T @ _resid

            _na = self._counts.get(_kid, 0)
            if _na == 0:
                self._means[_kid] = _mb
                self._counts[_kid] = _nb
                continue

            # 두 묶음을 합칠 때 생기는 평균 차이 항(Chan 병합). 이게 없으면 배치 경계에서
            # 산포가 새어 나간다.
            _delta = _mb - self._means[_kid]
            self._scatter += (_na * _nb / (_na + _nb)) * torch.outer(_delta, _delta)
            self._means[_kid] = self._means[_kid] + _delta * (_nb / (_na + _nb))
            self._counts[_kid] = _na + _nb

    def Finalize(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """누적 확정.

        Returns:
            (mean (K, D), class_ids (K,), within_cov (D, D), counts (K,)).
            ids 오름차순. `within_cov` 는 `N - K` 로 나눈 불편추정이다.

        Raises:
            RuntimeError: 누적이 없거나 표본이 클래스 수 이하라 자유도가 남지 않는 경우.
        """
        if self._scatter is None or not self._counts:
            raise RuntimeError(
                "누적된 표본이 없다 — Update 가 한 번도 불리지 않았다."
            )
        _ids = sorted(self._means)
        _n = sum(self._counts.values())
        _dof = _n - len(_ids)
        if _dof <= 0:
            raise RuntimeError(
                f"자유도가 없다 (표본 {_n}, 클래스 {len(_ids)}). 클래스당 2개 이상 필요하다."
            )
        return (
            torch.stack([self._means[_k] for _k in _ids], dim=0).float(),
            torch.tensor(_ids, dtype=torch.long),
            (self._scatter / _dof).float(),
            torch.tensor([self._counts[_k] for _k in _ids], dtype=torch.long),
        )

    def Reset(self) -> None:
        self._means.clear()
        self._counts.clear()
        self._scatter = None
