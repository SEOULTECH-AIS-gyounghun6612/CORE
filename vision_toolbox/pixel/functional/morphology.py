import cv2
import numpy as np
from vision_toolbox.typing.images import IMG_1C

def fill_light_area(
    mask: IMG_1C,
    target_size: int = 64,
    kernel_size: int = 5
) -> IMG_1C:
    """
    이진화된 마스크를 축소하여 조명 영역의 밀도를 파악하고, 
    모폴로지 연산을 통해 줄무늬 사이의 간격을 메운 뒤 복원합니다.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return mask

    h, w = mask.shape
    scale = target_size / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    
    small_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, small_mask = cv2.threshold(small_mask, 127, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(small_mask, kernel, iterations=1)
    
    return cv2.resize(dilated, (w, h), interpolation=cv2.INTER_NEAREST)

def morphology_ex(
    mask: IMG_1C,
    op: int,
    kernel_size: int = 5,
    iterations: int = 1
) -> IMG_1C:
    """
    기본 모폴로지 연산을 수행합니다.
    """
    if mask is None or iterations <= 0:
        return mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, op, kernel, iterations=iterations)

def find_blob_seeds(
    mask: IMG_1C,
    top_k: int = 1,
    min_size: int = 32,
    downscale_steps: int = 5
) -> IMG_1C:
    """
    마스크를 축소하며 밀도가 높은 영역(Seed)을 찾습니다.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return np.zeros_like(mask) if mask is not None else None

    curr = mask.copy()
    for _ in range(downscale_steps):
        h, w = curr.shape
        if h <= min_size or w <= min_size:
            break

        small_w = max(1, w // 2)
        small_h = max(1, h // 2)
        curr = cv2.resize(curr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        _, curr = cv2.threshold(curr, 127, 255, cv2.THRESH_BINARY)

        if cv2.countNonZero(curr) == 0:
            break

    seed_mask_small = np.zeros_like(curr)
    density_map = curr.copy()

    found_seeds = 0
    for _ in range(top_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(density_map)
        if max_val == 0:
            break
        cv2.circle(seed_mask_small, max_loc, 1, 255, -1)
        found_seeds += 1
        cv2.circle(density_map, max_loc, 3, 0, -1)

    if found_seeds == 0:
        return np.zeros_like(mask)

    return cv2.resize(seed_mask_small, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

def region_growing(
    seed_mask: IMG_1C,
    constraint_mask: IMG_1C
) -> IMG_1C:
    """
    시드에서 시작하여 제약 마스크 내의 연결된 영역을 모두 찾습니다.
    (Morphological Reconstruction)
    """
    if seed_mask is None or constraint_mask is None:
        return seed_mask

    current_blob = cv2.bitwise_and(seed_mask, constraint_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    while True:
        dilated = cv2.dilate(current_blob, kernel)
        next_blob = cv2.bitwise_and(dilated, constraint_mask)
        if cv2.countNonZero(cv2.bitwise_xor(current_blob, next_blob)) == 0:
            break
        current_blob = next_blob
        
    return current_blob

def extract_blob_mask(
    mask: IMG_1C,
    top_k: int = 1
) -> IMG_1C:
    """
    시드 찾기와 영역 확장을 결합하여 주요 덩어리를 추출합니다.
    """
    seeds = find_blob_seeds(mask, top_k=top_k)
    return region_growing(seeds, mask)
