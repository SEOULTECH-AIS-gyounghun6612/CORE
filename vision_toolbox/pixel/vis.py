"""이미지 시각화 및 변환 유틸리티 모듈.

이 모듈은 바운딩 박스 그리기, 컬러맵 적용 등 이미지 처리를 위한 
시각화 관련 함수들을 제공합니다.
"""

import numpy as np
import cv2

from vision_toolbox.typing.images import IMG_3C, IMG_1C, IMG_with_ALPHA
from vision_toolbox.typing.components import BBox

from vision_toolbox.pixel.state import ImageVisionState
from vision_toolbox.pixel.functional.measure import find_contours


def Make_canvas(
    img: IMG_1C | IMG_3C, bg: tuple[int | float, ...] | int = -1
) -> IMG_1C | IMG_3C | IMG_with_ALPHA:
    if bg == -1:
        if img.ndim == 2:  # 1채널 이미지인 경우
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(img.copy())

    _bg = bg if isinstance(bg, tuple) else (bg, bg, bg)
    _ch = len(_bg)

    if _ch not in (3, 4):
        raise ValueError(
            f"Length of 'bg' tuple must be 3 or 4. (Given: {_ch})"
        )
    return np.full((*img.shape[:2], _ch), bg, dtype=img.dtype)


def Draw_bbox(
    img: IMG_3C | IMG_1C,
    bbox: BBox,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2
) -> IMG_3C:
    """이미지에 바운딩 박스를 그립니다.

    Args:
        img: 입력 이미지 (1채널 또는 3채널).
        bbox: 바운딩 박스 좌표 (x, y, w, h).
        color: 박스 색상 (B, G, R). 기본값은 빨간색.
        thickness: 선 두께. 기본값은 2.

    Returns:
        바운딩 박스가 그려진 3채널 이미지.
    """
    # 1채널인 경우 3채널로 확장, 아니면 복사
    _canvas = Make_canvas(img)

    _bx, _by, _bw, _bh = bbox
    # 사각형 그리기
    cv2.rectangle(
        _canvas, (_bx, _by), (_bx + _bw, _by + _bh), color, thickness
    )
    return _canvas


def Draw_mask(
    img: IMG_3C | IMG_1C,
    mask: IMG_1C,
    offset: tuple[int, int] | None = None,
    is_contour_style: bool = True,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5,
    thickness: int = 1
) -> IMG_3C:
    """마스크를 캔버스(이미지) 위에 시각화합니다.

    Args:
        mask (IMG_1C): 1채널 이진 마스크 이미지.
        canvas (IMG_3C): 마스크를 그릴 3채널 배경 캔버스. (In-place 수정 됨)
        is_contour_style (bool, optional): True면 외곽선만, False면 내부를 칠함. 기본값 True.
        color (tuple[int, int, int], optional): 그릴 색상 (B, G, R). 기본값은 녹색.
        alpha (float, optional): is_contour_style이 False일 때 칠할 투명도. 기본값 0.5.
        thickness (int, optional): 외곽선 두께. 기본값 1.

    Returns:
        IMG_3C: 마스크가 그려진 캔버스.
    """
    _canvas = Make_canvas(img)

    if mask is None or np.count_nonzero(mask) == 0:
        return _canvas

    if is_contour_style:
        _cnts = find_contours(mask, "all")
        _offset = offset if offset is not None else (0, 0)
        cv2.drawContours(_canvas, _cnts, -1, color, thickness, offset=_offset)
    else:
        _ow, _oh = offset if offset is not None else (0, 0)
        _mh, _mw = mask.shape[:2]
        _roi = _canvas[_oh:_oh+_mh, _ow:_ow+_mw]
        _patch = np.full(_roi.shape, color, dtype=np.uint8)

        _blended = cv2.addWeighted(_patch, alpha, _roi, 1.0 - alpha, 0)

        _mask = mask > 0
        np.copyto(_roi, _blended, where=_mask[..., np.newaxis])

    return _canvas


def Apply_colormap(
    _img: IMG_1C,
    v_min: float | None = None,
    v_max: float | None = None,
    color_map=cv2.COLORMAP_JET,
    use_cv_norm: bool = False
) -> ImageVisionState:
    """1채널 데이터에 컬러맵을 적용하여 시각화 상태로 반환합니다.

    Args:
        _img: 1채널 입력 이미지.
        v_min: 최소 픽셀 값. None이면 마스크 내 최소값 사용.
        v_max: 최대 픽셀 값. None이면 마스크 내 최대값 사용.
        color_map: 적용할 OpenCV 컬러맵. 기본값 COLORMAP_JET.
        use_cv_norm: OpenCV 정규화 사용 여부. 기본값 False.

    Returns:
        이미지 및 마스크 정보가 포함된 ImageVisionState 객체.
    """
    # 2D 이미지로 차원 제한
    _img = _img[:2] if _img.ndim > 2 else _img

    # 유효 픽셀 및 값 범위 마스킹
    _mask = np.isfinite(_img) & (_img >= 0)
    if v_min is not None:
        _mask &= (_img >= v_min)
    if v_max is not None:
        _mask &= (_img <= v_max)

    _holder = ImageVisionState(_img)
    _holder.mask = _mask

    # 유효 픽셀 없으면 빈 이미지 반환
    if not _mask.any():
        _holder.image = np.zeros((*_img.shape, 3), dtype=np.uint8)
        return _holder

    if use_cv_norm:
        # bool 배열을 추가 할당 없이 uint8로 취급
        _mask_u8 = _mask.view(np.uint8)

        # OpenCV 백엔드를 활용한 정규화 (메모리 재할당 최소화)
        _img_u8 = np.zeros_like(_img, dtype=np.uint8)

        cv2.normalize(
            src=_img,
            dst=_img_u8,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
            mask=_mask_u8
        )
    else:
        # 단일 연산으로 최소/최대값 추출
        _min = np.min(_img, where=_mask, initial=np.inf)
        _max = np.max(_img, where=_mask, initial=-np.inf)

        # 픽셀값 범위 정규화 및 클리핑
        _diff = max(_max - _min, 1e-8)
        _factor = 255.0 / _diff

        # 정밀도 보존을 위해 float 연산 후 uint8 변환
        _img_u8 = np.clip(
            ((_img - _min) * _factor), 0, 255
        ).astype(np.uint8)

    # 컬러맵 적용
    _holder.image = cv2.applyColorMap(_img_u8, color_map)
    return _holder
