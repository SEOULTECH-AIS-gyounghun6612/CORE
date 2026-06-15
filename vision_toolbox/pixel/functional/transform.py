"""이미지 처리 및 공간적 변환 기능 모듈."""

from typing import Literal, Tuple, Union
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import cv2

from vision_toolbox.typing.images import IMG_3C, IMG_1C

CROP_MODE = Literal["pad", "crop_n", "crop_f", "crop_c"]

def _get_target_shape(
    size: Tuple[int, int], target_size: int, unit: int = 14,
    by_width: bool = True, use_pad: bool = True
) -> Tuple[Tuple[int, int], int]:
    """(Internal) 목표 크기 및 보정치 계산."""
    _ref, _other = size if by_width else (size[1], size[0])

    _rate = target_size / _ref
    _target_other_dim = round(_other * _rate)

    _dim_to_check = _target_other_dim if by_width else target_size
    _gap = (unit - (_dim_to_check % unit)) % unit

    if not use_pad:
        _gap = - (_dim_to_check % unit) if _dim_to_check % unit != 0 else 0

    _new_size = (
        target_size, _target_other_dim
    ) if by_width else (
        _target_other_dim, target_size
    )

    return _new_size, _gap

def _adjust_boundary(
    img: Union[IMG_3C, IMG_1C], mode: CROP_MODE, gap: int,
    is_w_dim: bool, fill: Union[int, float] = 0
) -> np.ndarray:
    """(Internal) 이미지 패딩 및 크롭 처리."""
    if gap == 0:
        return img

    _abs_gap = abs(gap)

    # Crop logic
    if gap < 0:
        if mode == "crop_n":
            return img[:, :-_abs_gap] if is_w_dim else img[:-_abs_gap, :]
        if mode == "crop_f":
            return img[:, _abs_gap:] if is_w_dim else img[_abs_gap:, :]

        _st, _ed = _abs_gap // 2, _abs_gap - (_abs_gap // 2)
        return img[:, _st:-_ed] if is_w_dim else img[_st:-_ed, :]

    # Pad logic
    _st, _ed = gap // 2, gap - (gap // 2)
    _pad_v = (
        (0, 0), (_st, _ed), (0, 0)
    ) if is_w_dim else (
        (_st, _ed), (0, 0), (0, 0)
    )
    _pad_dims = _pad_v[:img.ndim]

    return np.pad(img, _pad_dims, 'constant', constant_values=fill)

def resize_with_pad(
    img: Union[IMG_3C, IMG_1C],
    ref_size: int,
    unit_step: int = 14,
    mode: CROP_MODE = "pad",
    by_width: bool = True
) -> np.ndarray:
    """이미지 크기 조정 및 단위 정렬."""
    _h, _w = img.shape[:2]
    _sz_wh, _pad_gap = _get_target_shape(
        (_w, _h), ref_size, unit_step, by_width, mode == "pad"
    )
    
    _r_img = cv2.resize(img, _sz_wh, interpolation=cv2.INTER_AREA)
    if _r_img.ndim == 2:
        _r_img = _r_img[..., None]
    return _adjust_boundary(_r_img, mode, _pad_gap, not by_width)

def extract_sliding_windows(
    arr: np.ndarray,
    window_shape: Tuple[int, int],
    stride: Tuple[int, int]
) -> np.ndarray:
    """배열에서 슬라이딩 윈도우 패치를 추출합니다.

    Args:
        arr: 입력 배열. (H, W) 또는 (H, W, C)
        window_shape: 패치 크기 (H, W).
        stride: 추출 간격 (Y, X).

    Returns:
        추출된 패치들의 뷰 (N_y, N_x, window_H, window_W, [C]).
    """
    _stride_y, _stride_x = stride
    _view = sliding_window_view(arr, window_shape=window_shape, axis=(0, 1))
    return _view[::_stride_y, ::_stride_x]
