import cv2
import numpy as np
from typing import Union

from vision_toolbox.pipeline.core import Base_Step
from vision_toolbox.pixel.state import ImageVisionState
from vision_toolbox.pixel.functional.filters import frangi_filter

class BlurFilter(Base_Step):
    """이미지에 가우시안 블러를 적용하여 노이즈를 제거합니다."""
    def __init__(self, kernel_size: int = 5, sigma: float = 0, name: str = "BlurFilter"):
        super().__init__(name=name)
        self.ksize = (kernel_size, kernel_size) if kernel_size % 2 == 1 else (kernel_size + 1, kernel_size + 1)
        self.sigma = sigma

    def forward(self, state: ImageVisionState, *args, **kwargs) -> ImageVisionState:
        if state.image is None or state.image.size == 0:
            return state
        state.image = cv2.GaussianBlur(state.image, self.ksize, self.sigma)
        return state

class FrangiFilter(Base_Step):
    """관상 구조(tubular structures)를 감지하기 위한 Frangi Vesselness 필터입니다."""
    def __init__(
        self,
        sigmas: Union[float, int, tuple] = (1.0, 5.0),
        scale_step: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 15.0,
        black_ridges: bool = False,
        name: str = "FrangiFilter"
    ):
        super().__init__(name)
        if isinstance(sigmas, (int, float)):
            self.sigmas = np.array([float(sigmas)])
        else:
            self.sigmas = np.arange(sigmas[0], sigmas[1], scale_step)
            if len(self.sigmas) == 0:
                self.sigmas = np.array([float(sigmas[0])])
                
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.black_ridges = black_ridges

    def forward(self, state: ImageVisionState, *args, **kwargs) -> ImageVisionState:
        if state.image is None or state.image.size == 0:
            return state
        state.image = frangi_filter(
            state.image, self.sigmas, self.alpha, self.beta, self.gamma, self.black_ridges
        )
        return state
