from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, ClassVar

from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset

from python_toolbox.project import Base_Config

from ..typing import Mode


@dataclass
class Dataset_Config(Base_Config):
    """데이터셋 생성에 필요한 설정.

    data_kwargs는 Extract() 시 언패킹되어 Builder()에 개별 키워드 인자로 전달된다.

    Attributes:
        config_type: CFGS 레지스트리 조회 키.
        object_type: DATASETS 레지스트리 조회 키.
        data_dir: 데이터셋 루트 디렉토리 경로.
        name: 데이터셋 이름.
        category: 사용할 카테고리 서브셋.
        data_kwargs: 서브클래스별 추가 설정. Extract() 시 언패킹됨.
    """

    __unpack_extract__: ClassVar[set[str]] = {"data_kwargs"}

    config_type: str = "Dataset_Config"
    object_type: str = "Base_Dataset"

    data_dir: str = "./datasets"
    name: str = "no_data"
    category: str = "not_use"
    data_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataloader_Config(Base_Config):
    """DataLoader 생성에 필요한 설정.

    데이터셋은 dataset_meta dict로만 보유한다.
    Build_dataset 단계에서 CFGS 레지스트리를 통해 Dataset_Config를 생성한다.
    collate_fn은 레지스트리 키 문자열로 저장하고, Build_dataloader에서 함수로 변환한다.

    Attributes:
        batch_size: 배치 크기.
        num_workers: DataLoader 워커 수.
        shuffle: 에폭마다 데이터 셔플 여부. DDP 환경에서는 무시된다.
        drop_last: 마지막 불완전 배치 제거 여부.
        pin_memory: 핀 메모리 사용 여부.
        collate_fn: DATALOADER_FN 레지스트리 키. None이면 기본 collate 사용.
        pk_sampler: PK 샘플러 설정 딕셔너리. 설정 시 단일 GPU TRAIN에서 적용.
        dataset_meta: Dataset_Config 생성에 사용하는 raw dict.
    """

    __exclude_extract__: ClassVar[set[str]] = {"pk_sampler", "dataset_meta"}

    batch_size: int = 1
    num_workers: int = 0
    shuffle: bool = True
    drop_last: bool = False
    pin_memory: bool = False
    collate_fn: str | None = None
    pk_sampler: dict[str, Any] | None = None
    dataset_meta: dict[str, Any] = field(default_factory=dict)


class Custom_Dataset(Dataset):
    """프레임워크 데이터셋 추상 기반 클래스.

    서브클래스는 Builder(), __len__(), __getitem__()을 구현한다.
    ONNX export가 필요한 경우 Info_for_onnx()도 구현한다.

    Attributes:
        layout: 데이터 레이아웃 설명 (예: "NCHW"). ONNX export 메타데이터용.
        data_format: 데이터 포맷 설명 (예: "RGB_uint8"). ONNX export 메타데이터용.
        shape_profile: TensorRT 최적화 프로파일. {"min": [...], "opt": [...], "max": [...]}.
        mode: 현재 데이터셋의 실행 mode (train / val / test).
    """

    layout: str = ""
    data_format: str = ""
    shape_profile: dict[str, list[int]] = {"min": [], "opt": [], "max": []}

    def __init__(self, mode: Mode, **kwargs):
        self.mode = mode
        self.Builder(**kwargs)

    def Builder(self, data_dir: str, name: str, category: str, **kwargs):
        """데이터셋을 초기화한다.

        Dataset_Config.Extract()의 결과가 그대로 전달된다.

        Args:
            data_dir: 데이터셋 루트 디렉토리 경로.
            name: 데이터셋 이름 (예: "imagenet").
            category: 사용할 카테고리 서브셋 (예: "all").
            **kwargs: Dataset_Config의 data_kwargs에서 언패킹된 추가 인자.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index) -> Any:
        raise NotImplementedError

    def Info_for_onnx(self) -> tuple[
        nn.Module | None,
        tuple[Tensor, ...],
        dict[str, Any],
        dict[str, Any],
    ]:
        """ONNX export에 필요한 정보를 반환한다.

        Returns:
            tuple:
                - preprocess_layer: 모델 앞에 융합할 전처리 레이어. 없으면 None.
                - dummy_inputs: torch.onnx.export에 전달할 더미 입력 텐서 튜플.
                - onnx_kwargs: torch.onnx.export에 전달할 추가 키워드 인자
                  (input_names, output_names, dynamic_axes 등).
                - runtime_kwargs: TensorRT 런타임 설정 딕셔너리
                  (precision, workspace_size 등).
        """
        raise NotImplementedError
