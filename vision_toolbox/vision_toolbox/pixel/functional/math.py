import cv2
import numpy as np
from vision_toolbox.typing.images import IMG_1C

def Normalize(img: IMG_1C) -> IMG_1C:
    """
    이미지 픽셀 값을 0-255 범위(Min-Max)로 정규화합니다.
    결과는 uint8 타입으로 변환됩니다.
    """
    if img is None or img.size == 0:
        return img
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def Gradient_magnitude(img: IMG_1C) -> IMG_1C:
    """
    Sobel 연산자를 사용하여 이미지의 그래디언트 크기(Edge 강도)를 계산합니다.
    """
    if img is None or img.size == 0:
        return img
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)

def Histogram(img: IMG_1C, mask: IMG_1C | None = None) -> np.ndarray:
    """
    이미지의 밝기 히스토그램을 계산합니다.
    """
    if img is None or img.size == 0:
        return np.zeros(256, dtype=np.float32)
    return cv2.calcHist([img], [0], mask, [256], [0, 256])
