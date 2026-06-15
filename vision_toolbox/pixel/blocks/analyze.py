import cv2
import numpy as np
from vision_toolbox.pipeline.core import Base_Step
from vision_toolbox.pixel.state import ImageVisionState
from vision_toolbox.pixel.functional.measure import find_bbox, calculate_iou

class UpdateBBox(Base_Step):
    """이진 마스크를 기반으로 Bounding Box를 계산하여 상태에 저장합니다."""
    def forward(self, state: ImageVisionState, *args, **kwargs) -> ImageVisionState:
        if state.mask is not None:
            state.roi = find_bbox(state.mask)
        return state

class HistogramStop(Base_Step):
    """이미지의 밝기 분포를 분석하여 특정 조건 충족 시 중단 플래그를 설정합니다."""
    def __init__(self, ratio_thresh: float = 0.5, std_multiplier: float = 0.5, name: str = "HistogramStop"):
        super().__init__(name=name)
        self.ratio_thresh = ratio_thresh
        self.std_multiplier = std_multiplier

    def forward(self, state: ImageVisionState, *args, **kwargs) -> ImageVisionState:
        if state.image.size == 0:
            state.is_done = True
            return state
        mean_val = np.mean(state.image)
        std_val = np.std(state.image)
        thresh_val = mean_val + self.std_multiplier * std_val
        _, bin_img = cv2.threshold(state.image, thresh_val, 255, cv2.THRESH_BINARY)
        bright_ratio = cv2.countNonZero(bin_img) / state.image.size
        if bright_ratio >= self.ratio_thresh:
            state.is_done = True
        return state

class IoUStop(Base_Step):
    """이전 마스크와 현재 마스크의 IoU를 계산하여 변화가 없으면 중단 플래그를 설정합니다."""
    def __init__(self, threshold: float = 0.95, name: str = "IoUStop"):
        super().__init__(name=name)
        self.threshold = threshold
        self.prev_mask = None

    def forward(self, state: ImageVisionState, *args, **kwargs) -> ImageVisionState:
        if state.mask is None:
            return state
        
        if self.prev_mask is not None:
            iou = calculate_iou(self.prev_mask, state.mask)
            if iou >= self.threshold:
                state.is_done = True
        
        self.prev_mask = state.mask.copy()
        return state
