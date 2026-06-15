"""이미지 처리 파이프라인 블록 모듈.

이미지 처리를 위한 다양한 파이프라인 블록들을 모아놓은 모듈입니다.
"""
from typing import Any

from vision_toolbox.pipeline.core import Sequential_Step
from vision_toolbox.pipeline.config import Step_Config, get_pipeline_summary, get_config_hash

from .preprocess import BlurFilter, FrangiFilter
from .segment import ThresholdMask, ExtractBlob
from .refine import MorphologyClean, KeepLargestContour, KMeansCluster, RefineLightArea
from .analyze import UpdateBBox, HistogramStop, IoUStop
from .transform import CropToROI, ResizeView, ExtractPatches

__all__ = [
    "BlurFilter", "FrangiFilter",
    "ThresholdMask", "ExtractBlob",
    "MorphologyClean", "KeepLargestContour", "KMeansCluster", "RefineLightArea",
    "UpdateBBox", "HistogramStop", "IoUStop",
    "CropToROI", "ResizeView", "ExtractPatches",
    "Get_step_generator", "build_pipeline_and_meta"
]

# 각 단계 이름과 해당 클래스를 매핑하는 딕셔너리
STEP_MAP = {
    "blur": BlurFilter,
    "frangi": FrangiFilter,
    "threshold": ThresholdMask,
    "blob": ExtractBlob,
    "morphology": MorphologyClean,
    "largest_contour": KeepLargestContour,
    "cluster": KMeansCluster,
    "light_area": RefineLightArea,
    "bbox": UpdateBBox,
    "hist_stop": HistogramStop,
    "iou_stop": IoUStop,
    "crop": CropToROI,
    "resize": ResizeView,
    "extract_patches": ExtractPatches
}

def Get_step_generator(config: Step_Config):
    """설정에 따라 파이프라인 단계 생성기를 반환.

    Args:
        config: 파이프라인 단계 생성 설정 정보.

    Returns:
        설정에 맞는 파이프라인 단계 객체.

    Raises:
        ValueError: 설정에 'steps' 또는 'step_type'이 올바르지 않은 경우.
    """
    # 중첩된 하위 단계가 있는 경우 연속된 단계 생성
    if config.steps:
        return Sequential_Step(
            config.name,
            (Get_step_generator(_cfg) for _cfg in config.steps)
        )

    # 등록된 단일 단계 인스턴스 생성
    if config.step_type in STEP_MAP:
        return STEP_MAP[config.step_type](**config.params)

    # 유효하지 않은 설정 예외 처리
    raise ValueError(
        f"Invalid config at '{config.name}': Node must have 'steps' or 'step_type'."
    )

def build_pipeline_and_meta(
    step_configs_raw: list[dict[str, Any]], name: str = "Engine"
) -> tuple[Sequential_Step, str, str, str]:
    """순수 데이터(Config 딕셔너리 리스트)를 받아 파이프라인을 조립하고 메타데이터를 반환합니다.

    Args:
        step_configs_raw: 파이프라인 단계를 정의하는 원시 딕셔너리 리스트.
        name: 생성될 Sequential_Step의 이름.

    Returns:
        tuple: (조립된 파이프라인, 파이프라인 요약 이름, 파이프라인 해시, 파라미터 해시)
    """
    step_configs = [Step_Config(**cfg) for cfg in step_configs_raw]
    steps = [Get_step_generator(cfg) for cfg in step_configs]
    pipeline = Sequential_Step(f"{name}_Pipeline", steps)
    
    pipelines_name, pipeline_hash = get_pipeline_summary(step_configs)
    params_hash = get_config_hash(step_configs)
    
    return pipeline, pipelines_name, pipeline_hash, params_hash

