"""연속 오차 기반 threshold-sweep 정확도 및 AUC 산출."""
from __future__ import annotations

import torch
from torch import Tensor


def Get_accs(
    errors: Tensor,
    max_threshold: float,
    step_ct: int,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """threshold를 0부터 max_threshold까지 선형 분할하며 정확도 곡선을 산출한다.

    각 threshold t에서 errors <= t 인 샘플 비율을 정확도로 정의하며,
    mask가 주어지면 유효 샘플만 대상으로 집계한다.

    Args:
        errors: (N,) 샘플별 오차값.
        max_threshold: threshold 상한.
        step_ct: 구간 분할 수. threshold 포인트 수는 step_ct + 1.
        mask: (N,) bool/float 마스크. None이면 전체 샘플 사용.

    Returns:
        tuple:
            - acc_curve: (step_ct + 1,) threshold별 정확도.
            - thresholds: (step_ct + 1,) 선형 분할된 threshold 값.
    """
    _ths = torch.linspace(0, max_threshold, step_ct + 1, device=errors.device)
    _acc_curve = torch.stack([
        # mask 있으면 유효 샘플 평균, 없으면 전체 평균
        ((errors <= _t) if mask is None else (errors <= _t) * mask).float().mean()
        for _t in _ths
    ])
    return _acc_curve, _ths


def Compute_auc(
    errors_list: list[Tensor],
    max_thresholds: list[float],
    step_ct: int,
    mask: Tensor | None = None,
) -> tuple[Tensor, list[Tensor]]:
    """복수 오차 기준을 결합한 AUC를 산출한다.

    첫 번째 오차를 기준 threshold 축으로 삼고,
    이후 오차는 동일 축에서 정확도를 집계한 뒤 포인트별 최솟값(worst-case)으로 병합한다.
    최종 AUC는 병합된 곡선의 梯形 적분으로 계산한다.

    Args:
        errors_list: 오차 Tensor 목록. 길이는 max_thresholds와 같아야 한다.
        max_thresholds: 각 오차에 대응하는 threshold 상한 목록.
        step_ct: threshold 분할 수.
        mask: 유효 샘플 마스크. None이면 전체 샘플 사용.

    Returns:
        tuple:
            - auc: 병합 곡선의 정규화된 AUC 스칼라.
            - auc_per_error: 오차별 단독 AUC 목록.
    """
    assert len(errors_list) == len(max_thresholds), "errors_list와 max_thresholds 길이 불일치"

    _auc_per_error: list[Tensor] = []

    # 첫 번째 오차로 기준 threshold 축·정확도 곡선 확정
    _base_e, _base_th = errors_list[0], max_thresholds[0]
    _merged, _ths = Get_accs(_base_e, _base_th, step_ct, mask)
    _auc_per_error.append(torch.trapz(_merged, _ths) / _base_th)

    for _e, _th in zip(errors_list[1:], max_thresholds[1:]):
        # 추가 오차는 기준 threshold 축 위에서 정확도 집계
        _curve, _ = Get_accs(_e, _th, step_ct, mask)
        _auc_per_error.append(torch.trapz(_curve, _ths) / _base_th)
        # 포인트별 worst-case 병합: 가장 낮은 정확도를 기준으로 하한 설정
        _merged = torch.minimum(_merged, _curve)

    return torch.trapz(_merged, _ths) / _base_th, _auc_per_error
