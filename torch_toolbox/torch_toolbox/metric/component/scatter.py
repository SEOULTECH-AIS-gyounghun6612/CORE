from __future__ import annotations
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..definition import Accumulator
from .. import ACCUMULATORS


def Oas_shrinkage(cov: Tensor, dof: int) -> float:
    """공분산과 자유도만으로 shrinkage 계수를 정한다 (OAS 해석해).

    ``Sigma_hat = (1-r)*S + r*(tr S/d)*I`` 의 ``r`` 을 닫힌 식으로 낸다. 잡을 값은
    "표본이 부족한 만큼 구(球) 쪽으로 얼마나 당길까" 다 — 표본이 많으면 0 으로, 차원에 비해
    적으면 1 로 간다.

    **Ledoit-Wolf 대신 OAS 를 쓰는 이유는 비용이다.** LW 원식은 표본별 4차 항
    ``sum ||r_k||^4`` 이 필요해 전수 패스를 한 번 더 돌아야 한다. OAS 는 같은 목표(스케일
    항등행렬)에 대해 ``tr(S)``·``tr(S^2)``·자유도만 쓰므로 이미 모은 것으로 끝난다.

    Args:
        cov: (d, d) 대칭 공분산 추정치.
        dof: 그 추정에 쓰인 자유도. 클래스 내 풀링이면 ``표본 수 - 클래스 수`` 다.

    Returns:
        ``[0, 1]`` 의 shrinkage 계수.

    Raises:
        ValueError: ``cov`` 가 정사각이 아니거나 ``dof`` 가 1 미만인 경우.
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov 는 정사각이어야 한다: {tuple(cov.shape)}")
    if dof < 1:
        raise ValueError(f"자유도가 1 미만이다: {dof}")

    _s = cov.double()
    _d = _s.shape[0]
    _tr = torch.diagonal(_s).sum()
    _tr_sq = (_s * _s).sum()                      # 대칭이라 tr(S^2) = sum(S*S)

    # S 가 이미 등방이면 분모가 0 이다 — 당길 방향이 없으니 목표와 같다고 본다.
    _den = (dof + 1 - 2.0 / _d) * (_tr_sq - _tr ** 2 / _d)
    if _den <= 0:
        return 1.0

    _num = (1 - 2.0 / _d) * _tr_sq + _tr ** 2
    return float(torch.clamp(_num / _den, 0.0, 1.0))


def _Batch_moments(values: Tensor) -> tuple[int, float, float, float]:
    """(n,) 스칼라 묶음의 `(n, mean, M2, M3)`. `M_j` 는 중심 j 차 적률의 **합**이다."""
    _n = int(values.numel())
    _mean = values.mean()
    _dev = values - _mean
    return (
        _n, float(_mean), float((_dev ** 2).sum()), float((_dev ** 3).sum()),
    )


def _Merge_moments(
    a: tuple[int, float, float, float] | None, b: tuple[int, float, float, float],
) -> tuple[int, float, float, float]:
    """두 묶음의 `(n, mean, M2, M3)` 를 합친다 (Chan/Pébay 병합).

    합이 아니라 병합인 이유는 §`Within_Class_Scatter_Accumulator` 와 같다 — 원적률을
    빼는 식은 표본이 평균 근처에 뭉칠수록 큰 값의 차가 되어 자리수를 잃는다.
    """
    if a is None:
        return b
    _na, _ma, _m2a, _m3a = a
    _nb, _mb, _m2b, _m3b = b
    _n = _na + _nb
    _d = _mb - _ma
    return (
        _n,
        _ma + _d * _nb / _n,
        _m2a + _m2b + _d ** 2 * _na * _nb / _n,
        (
            _m3a + _m3b
            + _d ** 3 * _na * _nb * (_na - _nb) / _n ** 2
            + 3.0 * _d * (_na * _m2b - _nb * _m2a) / _n
        ),
    )


@ACCUMULATORS.Register_module("within_class_scatter_accumulator")
class Within_Class_Scatter_Accumulator(Accumulator):
    """클래스별 평균과 **클래스 내 공분산의 풀링**을 한 패스로 누적한다.

    클래스 간 산포를 빼고 클래스 **안**의 산포만 모은다. 전체 공분산을 모으면 클래스가
    떨어져 있다는 사실이 산포로 잡혀, 이것으로 화이트닝한 거리는 "클래스 안에서 얼마나
    전형적인가"를 재지 못한다.

    메모리는 공유 스캐터 `(D, D)` 하나와 클래스별 평균 `(K, D)` 다. 클래스마다 공분산을
    따로 들면 `K·D²` 이 되어 못 쓴다 — 풀링이 목적이므로 하나로 합치며 누적한다.

    **왜 단순 2차 모멘트 차가 아닌가.** `Σxxᵀ - Σ n_k μ_k μ_kᵀ` 로도 같은 값이 나오지만,
    클래스가 뭉쳐 있을수록 두 큰 값의 차가 되어 자리수가 상쇄된다(정규화 임베딩이면
    `Σxxᵀ` 의 trace 가 `N`, 결과의 trace 는 그보다 훨씬 작다). 여기서는 배치마다 배치
    평균 기준으로 중심화한 뒤 평균 차이를 보정해 합치므로(Chan 병합) 큰 값의 차가 생기지
    않는다. 누적은 `float64` 로 한다.

    합침은 교환·결합 법칙을 만족하므로 batch 도착 순서와 무관하다.

    **클래스별 스칼라(`class_scalar`) 도 같은 패스에서 받는다.** 벡터와 별개의 통계지만
    묶는 이유는 한 패스에서 나오기 때문이다 — 소비처가 두 누적기에 같은 배치를 두 번
    흘리고 두 결과의 클래스 정렬을 다시 맞추는 일을 없앤다. 모멘트를 3차까지 드는 것은
    표준편차만으로는 정규 근사가 성립하는지 알 수 없어서다(왜도가 그것을 말한다).

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
        #: class_id -> (n, mean, M2, M3). `class_scalar` 를 안 받으면 비어 있다.
        self._scalar: dict[int, tuple[int, float, float, float]] = {}

    def Update(
        self,
        *,
        embeddings: Tensor,
        gt_class_id: Tensor,
        class_scalar: Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """(N, D) 임베딩과 (N,) 클래스 id 를 클래스별로 합쳐 넣는다.

        Args:
            embeddings: (N, D) 임베딩.
            gt_class_id: (N,) 정수 class_id.
            class_scalar: (N,) 표본별 스칼라. 주면 클래스별 1·2·3차 중심 모멘트를 함께
                누적한다(`Finalize_scalar`). 안 주면 그 통계만 비어 있다.
        """
        _vec = F.normalize(embeddings, dim=-1) if self._normalize else embeddings
        _vec = _vec.detach().double()
        if self._scatter is None:
            _d = _vec.shape[1]
            self._scatter = torch.zeros(_d, _d, dtype=torch.float64, device=_vec.device)

        _scalar = None if class_scalar is None else class_scalar.detach().double()

        for _k in gt_class_id.unique():
            _kid = int(_k)
            _mask = gt_class_id == _k
            if _scalar is not None:
                self._scalar[_kid] = _Merge_moments(
                    self._scalar.get(_kid), _Batch_moments(_scalar[_mask]))

            _batch = _vec[_mask]
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

    def Finalize_scalar(self) -> tuple[Tensor, Tensor]:
        """`class_scalar` 의 클래스별 표준편차와 왜도. `Finalize` 와 같은 id 순서다.

        분리해 둔 이유는 벡터 통계와 쓰이는 곳이 달라서다 — 표준편차는 굽는 상수의 재료고,
        왜도는 그 상수가 전제하는 정규 근사를 점검하는 진단값이다.

        Returns:
            (std (K,), skew (K,)). 표본이 모자라 정의되지 않으면 NaN 이다
            (std 는 `n < 2`, skew 는 `n < 3` 이거나 산포가 0). `class_scalar` 를 한 번도
            안 받았으면 둘 다 전부 NaN.

        Raises:
            RuntimeError: 누적이 없는 경우.
        """
        if not self._counts:
            raise RuntimeError(
                "누적된 표본이 없다 — Update 가 한 번도 불리지 않았다."
            )
        _nan = float("nan")
        _std: list[float] = []
        _skew: list[float] = []
        for _k in sorted(self._means):
            _mom = self._scalar.get(_k)
            if _mom is None or _mom[0] < 2:
                _std.append(_nan)
                _skew.append(_nan)
                continue
            _n, _, _m2, _m3 = _mom
            _std.append((_m2 / (_n - 1)) ** 0.5)
            # 표본 왜도 g1. 산포가 0 이면 정의되지 않는다 — 0 으로 두면 "대칭"이라는
            # 거짓 진단이 된다.
            _skew.append(
                _m3 / _n / (_m2 / _n) ** 1.5 if _n >= 3 and _m2 > 0 else _nan)
        return (
            torch.tensor(_std, dtype=torch.float32),
            torch.tensor(_skew, dtype=torch.float32),
        )

    def Reset(self) -> None:
        self._means.clear()
        self._counts.clear()
        self._scalar.clear()
        self._scatter = None
