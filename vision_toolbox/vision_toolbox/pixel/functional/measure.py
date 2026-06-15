"""측정 및 기하학적 형태 탐색 유틸리티 모듈."""

import cv2
from typing import Literal
from numpy import count_nonzero

from vision_toolbox.typing.images import IMG_1C
from vision_toolbox.typing.components import BBox


def find_contours(
    mask: IMG_1C,
    mode: Literal["max", "min", "all"] = "max"
) -> list:
    """이진 마스크에서 외곽선을 탐색합니다.

    Args:
        mask (IMG_1C): 입력 이진 마스크.
        mode (Literal["max", "min", "all"], optional): 외곽선 선택 모드.
            기본값은 "max".

    Returns:
        list: 탐색된 외곽선 리스트.
    """
    # 외곽선 탐색
    _cntrs, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 외곽선 없음
    if not _cntrs:
        return []

    # 모드별 외곽선 필터링
    if mode in ["max", "min"]:
        _func = {"max": max, "min": min}[mode]
        _cntr = _func(_cntrs, key=cv2.contourArea)
        return [_cntr]

    return list(_cntrs)


def find_bbox(
    mask: IMG_1C
) -> BBox | None:
    """이진 마스크에서 Bounding Box(x, y, w, h)를 탐색합니다.

    Args:
        mask (IMG_1C): 입력 이진 마스크.

    Returns:
        BBox | None: 탐색된 Bounding Box. 없을 경우 None.
    """
    # 0이 아닌 픽셀 탐색
    _pts = cv2.findNonZero(mask)

    if _pts is None:
        return None

    # Bounding Box 반환
    return cv2.boundingRect(_pts)


def calculate_iou(mask1: IMG_1C, mask2: IMG_1C) -> float:
    """두 이진 마스크 간의 IoU 계산.

    Args:
        mask1 (IMG_1C): 첫 번째 이진 마스크.
        mask2 (IMG_1C): 두 번째 이진 마스크.

    Returns:
        float: 계산된 IoU 값.
    """
    if mask1.shape != mask2.shape:
        return 0.0

    # 1. mask1 검사: 비어있다면 mask2는 계산할 필요도 없이 즉시 0.0 반환
    _area_1 = count_nonzero(mask1)
    if _area_1 == 0:
        return 0.0

    # 2. mask2 검사: 비어있다면 즉시 0.0 반환
    _area_2 = count_nonzero(mask2)
    if _area_2 == 0:
        return 0.0

    # 3. 두 마스크 모두 유효할 때만 유일한 임시 배열 생성 및 교집합 연산 수행
    _inter = count_nonzero(mask1 & mask2)
    _union = _area_1 + _area_2 - _inter

    return _inter / _union
