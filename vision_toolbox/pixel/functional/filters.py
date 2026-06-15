import cv2
import numpy as np
from vision_toolbox.typing.images import IMG_1C
from vision_toolbox.pixel.functional.math import Normalize

def frangi_filter(
    img: IMG_1C,
    sigmas: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 15.0,
    black_ridges: bool = False
) -> IMG_1C:
    """
    Frangi Vesselness 필터를 적용합니다.
    """
    if img is None or img.size == 0:
        return img

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = img.astype(np.float32) / 255.0
    vesselness = np.zeros_like(img)

    for sigma in sigmas:
        ksize = int(2 * round(3 * sigma) + 1)
        gaussian = cv2.GaussianBlur(img, (ksize, ksize), sigma)

        Dxx = cv2.Sobel(gaussian, cv2.CV_32F, 2, 0, ksize=3) * (sigma ** 2)
        Dyy = cv2.Sobel(gaussian, cv2.CV_32F, 0, 2, ksize=3) * (sigma ** 2)
        Dxy = cv2.Sobel(gaussian, cv2.CV_32F, 1, 1, ksize=3) * (sigma ** 2)

        trace = Dxx + Dyy
        det = Dxx * Dyy - Dxy * Dxy
        discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))

        lambda1 = (trace + discriminant) / 2
        lambda2 = (trace - discriminant) / 2

        swap_mask = np.abs(lambda1) > np.abs(lambda2)
        l1 = np.where(swap_mask, lambda2, lambda1)
        l2 = np.where(swap_mask, lambda1, lambda2)

        Rb = np.abs(l1) / (np.abs(l2) + 1e-10)
        S = np.sqrt(l1**2 + l2**2)

        V = np.exp(-(Rb**2) / (2 * beta**2)) * (1 - np.exp(-(S**2) / (2 * gamma**2)))

        if black_ridges:
            V[l2 <= 0] = 0
        else:
            V[l2 >= 0] = 0

        vesselness = np.maximum(vesselness, V)

    return Normalize(vesselness)
