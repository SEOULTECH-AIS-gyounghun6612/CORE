from __future__ import annotations

import torch
from torch import Tensor


def Top_k_accuracy(scores: Tensor, targets: Tensor, k: int = 1) -> float:
    """top-k 정확도. scores 상위 k개 안에 target이 있으면 정답으로 집계한다.

    Args:
        scores: (N, C) 클래스별 점수/유사도. 열 index = 클래스 id.
        targets: (N,) 정답 클래스 id ([0, C) 범위).
        k: 상위 후보 개수. k=1이면 일반 top-1 정확도.

    Returns:
        [0, 1] 범위 정확도 스칼라.
    """
    _topk = scores.topk(k, dim=-1).indices            # (N, k)
    return (_topk == targets[:, None]).any(dim=1).float().mean().item()


def Confusion_matrix(preds: Tensor, targets: Tensor, num_classes: int) -> Tensor:
    """(C, C) 혼동 행렬을 산출한다. row=target, col=pred.

    bincount에 선형 인덱스를 전달하는 방식으로 O(N) 시간에 집계한다.

    Args:
        preds: (N,) top-1 예측 클래스 id.
        targets: (N,) 정답 클래스 id.
        num_classes: 클래스 수 C.

    Returns:
        (C, C) long Tensor. [i, j] = class i를 j로 예측한 샘플 수.
    """
    # (target * C + pred) 선형 인덱스 → bincount → (C, C) reshape
    _idx = targets.long() * num_classes + preds.long()
    return torch.bincount(
        _idx, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)
