import cv2
import numpy as np
from typing import Union

from vision_toolbox.pipeline.core import Base_Step
from vision_toolbox.pixel.state import ImageVisionState
from vision_toolbox.pixel.functional.math import normalize
from vision_toolbox.pixel.functional.morphology import extract_blob_mask

class ThresholdMask(Base_Step):
    """이미지 이진화를 수행합니다."""
    def __init__(self, value: Union[int, float] = 1.0, name: str = "ThresholdMask"):
        super().__init__(name=name)
        self.value = value

    def forward(self, state: ImageVisionState, *args, **kwargs) -> ImageVisionState:
        if state.image is None or state.image.size == 0:
            return state

        norm_img = normalize(state.image)
        if isinstance(self.value, int):
            threshold_val = float(self.value)
        else:
            threshold_val = np.mean(norm_img) + self.value * np.std(norm_img)

        _, thresh = cv2.threshold(norm_img, threshold_val, 255, cv2.THRESH_BINARY)
        state.mask = thresh
        return state

class ExtractBlob(Base_Step):
    """이미지를 축소하여 밀도가 높은 씨앗(Seed)을 찾고 다시 복원하여 큰 덩어리를 찾는 단계."""
    def __init__(self, top_k: int = 1, name: str = "ExtractBlob"):
        super().__init__(name=name)
        self.top_k = top_k

    def forward(self, state: ImageVisionState, *args, **kwargs) -> ImageVisionState:
        if state.mask is not None:
            state.mask = extract_blob_mask(state.mask, self.top_k)
        return state
