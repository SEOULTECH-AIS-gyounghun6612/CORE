import numpy as np
from typing import Tuple, List

from vision_toolbox.pipeline.core import Base_Step
from vision_toolbox.pixel.state import ImageVisionState
from vision_toolbox.pixel.functional.transform import resize_with_pad, extract_sliding_windows, CROP_MODE

class CropToROI(Base_Step):
    """현재 State에 설정된 ROI 영역에 맞게 이미지와 마스크를 자릅니다."""
    def __init__(self, strict: bool = True, name: str = "CropToROI"):
        super().__init__(name=name)
        self.strict = strict

    def forward(self, state: ImageVisionState, *args, **kwargs) -> ImageVisionState:
        if state.roi is not None:
            state.crop(state.roi, strict=self.strict)
        return state

class ResizeView(Base_Step):
    """현재 State의 이미지를 리사이징하고, 이를 반영한 새로운 State를 반환합니다.
    (주의: 원본 좌표계 매핑을 유지해야 하는 경우 주의해서 사용)"""
    def __init__(self, ref_size: int, unit_step: int = 14, mode: CROP_MODE = "pad", by_width: bool = True, name: str = "ResizeView"):
        super().__init__(name=name)
        self.ref_size = ref_size
        self.unit_step = unit_step
        self.mode = mode
        self.by_width = by_width

    def forward(self, state: ImageVisionState, *args, **kwargs) -> ImageVisionState:
        if state.image is None:
            return state
        
        resized_img = resize_with_pad(state.image, self.ref_size, self.unit_step, self.mode, self.by_width)
        
        new_state = ImageVisionState(image=resized_img)
        # 리사이즈 시 기존의 mask나 roi는 스케일이 달라지므로 초기화하거나 
        # 스케일 팩터에 맞춰 조절해야 합니다. 이 구현에서는 새로운 뷰로 취급합니다.
        return new_state

class ExtractPatches(Base_Step):
    """현재 State의 이미지를 슬라이딩 윈도우 패치로 분할하여 배열의 리스트를 반환합니다."""
    def __init__(self, patch_size: Tuple[int, int] = (64, 64), stride: Tuple[int, int] = (32, 32), is_positive: bool = True, name: str = "ExtractPatches"):
        super().__init__(name=name)
        self.patch_size = patch_size
        self.stride = stride
        self.is_positive = is_positive

    def forward(self, state: ImageVisionState, *args, **kwargs) -> List[np.ndarray]:
        if state.image is None or state.mask is None:
            return []
            
        mask_view = extract_sliding_windows(state.mask, self.patch_size, self.stride)
        img_view = extract_sliding_windows(state.image, self.patch_size, self.stride)
        
        has_overlap = np.any(mask_view, axis=(-2, -1))
        valid_mask = has_overlap if self.is_positive else ~has_overlap
        
        patches = img_view[valid_mask]
        
        result = []
        for i in range(patches.shape[0]):
            result.append(patches[i])
        return result
