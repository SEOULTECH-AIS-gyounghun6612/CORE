""" ### 학습을 위한 기본 구조 처리 모듈

------------------------------------------------------------------------
### Requirement
    - torch >=2.2
    - torchsummary
    - python_ex

### Structure
    - Mode: 학습 과정을 표현하기 위한 열거형
    - LearningProcess: End to End 학습을 구현한 기본 구조

"""
from __future__ import annotations
from enum import auto
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, TypeVar, Callable, Generic
)

from torch import load, save, device
from torch.autograd.grad_mode import no_grad
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from python_ex.system import Path, String
from python_ex.project import Template

from torch_ex.dataset import (DATASET, Config as Dataset_Config)
from torch_ex.neural_network import (MODEL, Config as Neural_Network_Config)


class Mode(String.String_Enum):
    """ ### 학습 과정을 표현하기 위한 열거형

    ---------------------------------------------------------------------
    ### Args
    - None

    ### Attributes
    - `TRAIN`
    - `VALIDATION`
    - `TEST`

    ### Structure
    - None

    """

    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()


class Process_Config():
    @dataclass
    class Observer():
        data_ct: int = 0
        loss: dict[int, dict[str, list[float]]] = field(default_factory=dict)

    OBSERVER = TypeVar(
        "OBSERVER",
        bound=Observer
    )

    @dataclass
    class Optim():
        def Get_optimizer_params(self) -> dict[str, Any]:
            raise NotImplementedError

    CONFIG_OPTIM = TypeVar(
        "CONFIG_OPTIM",
        bound=Optim
    )

    @dataclass
    class End_to_End(
        Generic[
            Dataset_Config.CONFIG_DATALOADER,
            Neural_Network_Config.CONFIG_NEURAL_NETWORK,
            CONFIG_OPTIM
        ]
    ):
        project_name: str
        max_epoch: int
        this_epoch: int
        gpus: list[int]
        holder: Process_Config.OBSERVER

        dataloader_cfg: dict[str, Dataset_Config.CONFIG_DATALOADER]
        neural_network_cfg: Neural_Network_Config.CONFIG_NEURAL_NETWORK
        optimizer_cfg: Process_Config.CONFIG_OPTIM

    CONFIG_END2END = TypeVar(
        "CONFIG_END2END",
        bound=End_to_End
    )


class End_to_End(Template, ABC, Generic[DATASET, MODEL]):
    """ ### End to End 학습을 구현한 기본 구조
    각 프로젝트에서 요구되는 End to End 학습 구조를 구성하기 위하여
    기본적인 구조를 구성한 모듈.\n
    상속 후 세부적인 사항의 작성이 필요함.

    ---------------------------------------------------------------------
    ### Args
    - Super
        - `project_name`: 프로젝트 이름
        - `category`: 프로젝트 구분
        - `result_dir`: 프로젝트 결과 저장 최상위 경로를 생성하기 위한 경로
    - This
        - `apply_mode`: 학습에 사용할 진행 과정
        - `max_epoch`: 학습 최대 반복 횟수
        - `last_epoch`: 이전 학습에서 완료한 마지막 반복 횟수
        - `gpus`: 학습 과정에 사용하고자 하는 GPU 장치 번호

    ### Attributes
    - `project_name`: 프로젝트 이름
    - `result_root`: 프로젝트 결과 저장 최상위 경로
    - `apply_mode`: 학습에 사용할 진행 과정
    - `max_epoch`: 학습 최대 반복 횟수
    - `last_epoch`: 이전 학습에서 완료한 마지막 반복 횟수
    - `gpus`: 학습 과정에 사용하고자 하는 GPU 장치 번호
    - `holder`: 주요 상수 정보를 저장하기 위한 보관 자료형
    - `loss`: 학습 과정에 따른 loss 변화를 저장하기 위한 보관 자료형
    - `eval_holder`: 학습 과정에 따른 평가 인자 변화를 저장하기 위한 보관 자료형

    ### Structure
    - Set_dataloader: 학습 과정에 사용할 데이터를 처리하는 Dataloader 생성 함수
    - Set_eval_holder: 학습 과정 중 생성되는 평가 인자 보관 자료형 구성 함수
    - Get_output: 입력 데이터에 따른 모델의 결과를 생성하는 함수
    - Save_weight: 학습 과정의 주요 인자값을 저장하는 함수
    - Load_weight: 이전 학습 과정에서 생성된 주요 인자값을 불러오는 함수
    - Main_work: 전체 학습을 진행하는 함수
    - Run: 학습 실행 함수

    ### To do
    - Set_dataset: 학습 과정에 사용하기 위한 데이터셋 구성 함수
    - Set_model: 학습 대상 모델을 구성하는 함수
    - Set_optimizer: 학습 대상 모델에 할당되는 optimizer와 scheduler을 구성하는 함수
    - Set_loss: 학습 과정 중 사용하고자 하는 loss 구조와 값 보관 자료형 구성 함수
    - Set_intput_n_target: Dataloader를 통해 처리된 데이터의 후처리 과정 함수
    - Core: 실질적인 학습을 진행하는 함수
    - Decision_for_learning: 반복 학습 결과를 확인하고, 그에 따른 결정을 내리는 함수


    """
    def __init__(
        self,
        project_name: str,
        max_epoch: int, last_epoch: int = -1,
        observer: Process_Config.OBSERVER = Process_Config.Observer()
    ):
        super().__init__(project_name)

        # base info
        self.max_epoch = max_epoch
        self.last_epoch = last_epoch

        self.gpus: list[int] = []

        # debug info
        self.observer: Process_Config.OBSERVER = observer

    @abstractmethod
    def Set_optimizer(
        self, model: MODEL, **kwarg
    ) -> tuple[Optimizer, LRScheduler | None]:
        """ ### 학습 대상 모델에 할당되는 optimizer와 scheduler을 구성하는 함수

        ------------------------------------------------------------------
        ### Args
        - `model`: 학습 대상 모델
        - `optimizer_prams`: optimizer와 scheduler을 구성을 위해 사용되는 설정 값

        ### Returns
        - `Optimizer`: 학습에 사용되는 optimizer
        - `LRScheduler` or `None`: 학습에 사용되는 scheduler. 설정이 없을 경우 None.

        ### Raises
        - `NotImplementedError`: 해당 함수는 구성이 필요함

        """
        raise NotImplementedError

    @abstractmethod
    def Jump_to_cuda(self, data: Any, device_info: device) -> Any:
        """ ### Dataloader를 통해 처리된 데이터의 후처리 과정 함수

        ------------------------------------------------------------------
        ### Args
        - `device_info`: 학습 모델에 사용될 장치
        - `data`: Dataloader를 통해 구성한 학습용 데이터

        ### Returns
        - `Input Data`: 학습 모델 입력 데이터
        - `Target Data`: 지도 학습 과정 중 사용하는 정답 데이터
        - `Additional info`: 각 데이터의 추가 정보

        ### Raises
        - `NotImplementedError`: 해당 함수는 구성이 필요함

        """
        raise NotImplementedError

    @abstractmethod
    def Core(
        self,
        epoch: int,
        mode: Mode,
        model: Module,
        loss_fn: list[Callable],
        optim: Optimizer,
        **data
    ) -> int:
        """ ### 실질적인 학습을 진행하는 함수
        학습 과정에 따라 학습에 필요한 과정(ex, backpropagation, 평가값 갱신 등)수행

        ------------------------------------------------------------------
        ### Args
        - `epoch`: 현재 학습 반복 횟수
        - `mode`: 현재 학습 과정
        - `model`: 학습 모델
        - `output`: 입력 데이터에 따른 출력 데이터
        - `target`: 입력 데이터에 따른 정답 데이터
        - `optim`: 학습에 사용되는 optimizer
        - `data_count`: 지금까지 사용된 데이터 갯수

        ### Returns or Yields
        - `this_data_count`: 현재까지 사용된 데이터 갯수

        ### Raises
        - `NotImplementedError`: 해당 함수는 구성이 필요함

        """
        raise NotImplementedError

    def Save_weight(
        self,
        model: Module,
        optim: Optimizer,
        scheduler: LRScheduler | None,
        file_name: str,
        file_dir: list[str] | None = None
    ):
        """ ### 학습 과정의 주요 인자값을 저장하는 함수

        ------------------------------------------------------------------
        ### Args
        - `model`: 학습 대상 모델
        - `optim`: 학습에 사용되는 optimizer
        - `scheduler`: 학습에 사용되는 scheduler. 없는 경우 저장하지 않음.
        - `file_name`: 인자가 저장되는 파일
        - `file_dir`: 저장되는 파일이 위치할 디렉토리

        ### Returns
        - None

        ### Raises
        - None

        """
        _scheduler = None if scheduler is None else scheduler.state_dict()
        _file_dir = [] if file_dir is None else file_dir
        _file_dir += [f"{file_name}.h5"]

        save(
            {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scheduler": _scheduler
            },
            Path.Join(_file_dir, self.result_root)
        )

    def Load_weight(
        self,
        model: Module,
        optim: Optimizer,
        scheduler: LRScheduler | None,
        file_name: str,
        file_dir: list[str] | None = None
    ) -> tuple[Module, Optimizer, LRScheduler | None]:
        """ ### 이전 학습 과정에서 생성된 주요 인자값을 불러오는 함수

        ------------------------------------------------------------------
        ### Args
        - `model`: 학습 대상 모델
        - `optim`: 학습에 사용되는 optimizer
        - `scheduler`: 학습에 사용되는 scheduler. 사용하지 않는 경우 불러오지 않음.
        - `file_name`: 인자가 저장되어 있는 파일
        - `file_dir`: 저장되어 있는 파일이 위치한 디렉토리

        ### Returns or Yields
        - `model`: 이전 학습으로 훈련 된 모델
        - `optim`: 학습에 사용되는 optimizer
        - `scheduler`: 학습에 사용되는 scheduler. 정보가 없는 경우 불러오지 않음.

        ### Raises
        - None

        """
        _file_dir = [] if file_dir is None else file_dir
        _file_dir += [f"{file_name}.h5"]
        _state_dict = load(Path.Join(_file_dir, self.result_root))

        model.load_state_dict(_state_dict["model"])
        optim.load_state_dict(_state_dict["optim"])
        if scheduler is not None and _state_dict["scheduler"] is not None:
            scheduler.load_state_dict(_state_dict["scheduler"])

        return model, optim, scheduler

    @abstractmethod
    def Decision_to_learning_stop(self, **kwarg) -> bool:
        """ ### 반복 학습 결과를 확인하고, 그에 따른 결정을 내리는 함수
        해당 반복학습 결과의 경우 Attributes `self.loss`, `self.holder`,
        `self.eval_holder` 정보를 바탕으로 처리하게 됨.\n

        ------------------------------------------------------------------
        ### Args
        - `epoch`: 현재 학습 반복 횟수
        - `model`: 학습 모델
        - `optim`: 학습에 사용되는 optimizer
        - `scheduler`: 학습에 사용되는 scheduler. 없는 경우 저장하지 않음.

        ### Returns
        - None

        ### Raises
        - `NotImplementedError`: 해당 함수는 구성이 필요함

        """
        raise NotImplementedError

    def Main_work(
        self,
        thred_num: int,
        cfg: Process_Config.End_to_End
    ):
        """ ### 전체 학습을 진행하는 함수

        ------------------------------------------------------------------
        ### Args
        - `thred_num`: 해당 학습 과정 구분 번호
        - `prams`: 학습에 사용되는 인자값

        ### Returns
        - None

        ### Raises
        - None

        """
        # preprocess for learning
        # set dataloader
        _dataloader = dict(
            (
                Mode(_k),
                DataLoader(
                    _v.dataset_config.Build_dataset(_k),
                    **_v.Get_Dataloader_params()
                )
            ) for _k, _v in cfg.dataloader_cfg.items() if _k in list(Mode)
        )

        # set model and loss function
        _model, _loss_fns = cfg.neural_network_cfg.Build_model_n_loss()

        # set optim and scheduler
        _optim, _scheduler = self.Set_optimizer(
            _model, **cfg.optimizer_cfg.Get_optimizer_params())

        _gpu_list = self.gpus
        _gpu_id = _gpu_list[thred_num] if _gpu_list else -1
        _device = device(f"cuda:{_gpu_id}" if _gpu_list else "cpu")

        # load pretrained weight
        if self.last_epoch:
            _model, _optim, _scheduler = self.Load_weight(
                _model, _optim, _scheduler, "last"
            )

        for _epoch in range(self.last_epoch, self.max_epoch):
            for _mode, _loader in _dataloader.items():
                for _data in _loader:
                    self.Core(
                        _epoch, _mode,
                        _model, _loss_fns, _optim,
                        **self.Jump_to_cuda(_data, _device)
                    )
            # make decision for learning in each epoch
            if self.Decision_to_learning_stop():
                break

            # !!! learning is continue !!!
            if _scheduler is not None:
                _scheduler.step()

    def Run(self, config):
        """ ### 학습 실행 함수

        ------------------------------------------------------------------
        ### Args
        - `prams`: 학습에 사용되는 인자값

        ### Returns
        - None

        ### Raises
        - None

        """
        # not use distribute
        # in later add code, that use distribute option
        self.Main_work(0, config)

    @staticmethod
    def get_output(mode: Mode, model: Module, **intpu_data):
        if mode is Mode.TRAIN:
            return model(**intpu_data)
        with no_grad():
            return model(**intpu_data)
