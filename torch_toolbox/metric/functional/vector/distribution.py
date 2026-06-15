from __future__ import annotations

from torch import Tensor


def Vmf_concentration(
    means: Tensor, dim: int | None = None, eps: float = 1e-6
) -> tuple[Tensor, Tensor]:
    """단위벡터 그룹 평균으로부터 평균 resultant length R̄와 von Mises-Fisher 농도 κ를 산출함.

    means는 단위벡터들의 (미정규화) 평균임 (예: normalize=True로 누적한
    Centroid_Accumulator.Finalize 결과). R̄ = ‖means‖ 이며 그룹 prototype에 대한
    평균 cosine과 같음. 농도는 Banerjee 근사 κ̂ = R̄(p − R̄²)/(1 − R̄²)(p=차원)를 씀.
    R̄→1(완전 집중, 표본 1개 포함)에서 분모를 eps로 하한함.

    Args:
        means: (K, D) 단위벡터 그룹 평균.
        dim: 차원 p. None이면 means.shape[-1].
        eps: (1 − R̄²) 하한.

    Returns:
        (R_bar (K,), kappa (K,)).
    """
    _R = means.norm(dim=-1)
    _p = means.shape[-1] if dim is None else dim
    _kappa = _R * (_p - _R**2) / (1.0 - _R**2).clamp(min=eps)
    return _R, _kappa
