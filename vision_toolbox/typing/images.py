"""프로젝트 공통 타입 별칭(상수) 정의."""

from typing import Annotated
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "M_ROT", "V_TRANS", "M_RT", "V_FLAT_RT",
    "M_TF", "V_FLAT_TF",
    "M_IN",
    "V_1D", "V_2D", "V_3D", "V_4D",
    "V_SIZE", "V_CLR",
    "IMG_1C", "IMG_3C", "IMG_with_ALPHA"
]

# 행렬 및 파라미터 타입
M_ROT = Annotated[NDArray, (None, 3, 3)]    # 회전 행렬
V_TRANS = Annotated[NDArray, (None, 3, 1)]  # 이동 벡터
M_RT = Annotated[NDArray, (None, 3, 4)]    # R|T matrix
V_FLAT_RT = Annotated[NDArray, (None, 12)]  # flat R|T

M_TF = Annotated[NDArray, (None, 4, 4)]     # pose matrix (transfer)
V_FLAT_TF = Annotated[NDArray, (None, 16)]  # flat pose

M_IN = Annotated[NDArray, (None, 3, 3)]     # 카메라 내부 파라미터

# 벡터 타입
V_1D = Annotated[NDArray, (None, 1)]  # 1D 벡터
V_2D = Annotated[NDArray, (None, 2)]  # 2D 벡터
V_3D = Annotated[NDArray, (None, 3)]  # 3D 벡터
V_4D = Annotated[NDArray, (None, 4)]  # 4D 벡터 (동차 좌표)

# 이미지 및 포인트 타입
V_SIZE = Annotated[NDArray[np.int32], (None, 2)]  # 이미지 크기
V_CLR = Annotated[NDArray[np.uint8], (None, 3)]   # 포인트 색상

IMG_1C = Annotated[NDArray, (None, None)]           # 1채널 이미지
IMG_3C = Annotated[NDArray, (None, None, 3)]        # 3채널 이미지
IMG_with_ALPHA = Annotated[NDArray, (None, None, 4)]# 투명 채널 존재

SQ_IMG_1C = Annotated[NDArray, (None, None, None)]  # 1채널 연속 이미지
SQ_IMG_3C = Annotated[NDArray, (None, None, None, 3)]  # 3채널 연속 이미지
