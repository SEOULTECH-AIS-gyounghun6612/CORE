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
from enum import Enum
from typing import Tuple, Dict, List, Any

from torch import Tensor, load, save, device
from torch.autograd.grad_mode import no_grad
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader

from python_ex.system import Path
from python_ex.project import Template


PARAMS = Dict[str, Any]


class Mode(Enum):
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

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class LearningProcess(Template):
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
    - Set_dataset: 학습 과정에 사용하기 위한 데이터셋 구성 함수
    - Set_dataloader: 학습 과정에 사용할 데이터를 처리하는 Dataloader 생성 함수
    - Set_model: 학습 대상 모델을 구성하는 함수
    - Set_optimizer: 학습 대상 모델에 할당되는 optimizer와 scheduler을 구성하는 함수
    - Set_loss: 학습 과정 중 사용하고자 하는 loss 구조와 값 보관 자료형 구성 함수
    - Set_eval_holder: 학습 과정 중 생성되는 평가 인자 보관 자료형 구성 함수
    - Set_intput_n_target: Dataloader를 통해 처리된 데이터의 후처리 과정 함수
    - Get_output: 입력 데이터에 따른 모델의 결과를 생성하는 함수
    - Core: 실질적인 학습을 진행하는 함수
    - Save_weight: 학습 과정의 주요 인자값을 저장하는 함수
    - Load_weight: 이전 학습 과정에서 생성된 주요 인자값을 불러오는 함수
    - Decision_for_learning: 반복 학습 결과를 확인하고, 그에 따른 결정을 내리는 함수
    - Main_work: 전체 학습을 진행하는 함수
    - Run: 학습 실행 함수

    """
    def __init__(
        self,
        project_name: str, apply_mode: List[Mode], max_epoch: int,
        last_epoch: int = -1, gpus: List[int] | None = None,
        category: str | None = None, result_root: str | None = None
    ):
        super().__init__(project_name, category, result_root)

        # base info
        self.max_epoch = max_epoch
        self.last_epoch = last_epoch + 1
        self.apply_mode = apply_mode

        self.gpus = [] if gpus is None else gpus

        # debug info
        self.holder: Dict[Mode, Dict[str, int]] = {}
        self.loss: Dict[Mode, Dict[str, List[float]]] = {}
        self.eval_holder: Dict[Mode, Dict[str, List[float]]] = {}

    def Set_dataset(
        self, mode: Mode, dataset_prams: PARAMS
    ) -> Dataset:
        """ ### 학습 과정에 사용하기 위한 데이터셋 구성 함수

        ------------------------------------------------------------------
        ### Args
        - `mode`: 해당 데이터셋을 사용하는 학습 과정
        - `dataset_prams`: 데이터셋 구성을 위한 설정 값

        ### Returns
        - `Dataset`: 학습에 사용하고자 하는 데이터 셋

        ### Raises
        - `NotImplementedError`: 해당 함수는 구성이 필요함

        """
        raise NotImplementedError

    def Set_dataloader(
        self, dataset: Dataset, dataloader_prams: PARAMS
    ) -> DataLoader:
        """ ### 학습 과정에 사용할 데이터를 처리하는 Dataloader 생성 함수

        ------------------------------------------------------------------
        ### Args
        - `dataset`: 학습 과정에 사용되는 데이터 셋
        - `dataloader_prams`: Dataloader 구성을 위한 설정 값

        ### Returns
        - `DataLoader`: 학습에 사용되는 데이터를 구성하는 Dataloader

        ### Raises
        - None

        """
        return DataLoader(dataset, **dataloader_prams)

    def Set_model(
        self, device_info: device, model_prams: PARAMS
    ) -> Module:
        """ ### 학습 대상 모델을 구성하는 함수

        ------------------------------------------------------------------
        ### Args
        - `device_info`: 학습 모델에 사용될 장치
        - `model_prams`: 모델 구성을 위한 설정 값

        ### Returns
        - `Module`: 학습 대상 모델

        ### Raises
        - `NotImplementedError`: 해당 함수는 구성이 필요함

        """
        raise NotImplementedError

    def Set_optimizer(
        self, model: Module, optimizer_prams: PARAMS
    ) -> Tuple[Optimizer, LRScheduler | None]:
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

    def Set_loss(self, loss_prams: PARAMS):
        """ ### 학습 과정 중 사용하고자 하는 loss 구조와 값 보관 자료형 구성 함수
        해당 함수에서 학습이 진행되는 과정에서 발생된 loss를 기록하는 Attributes
        `self.loss`의 초기화도 같이 진행할 필요가 있음.

        ------------------------------------------------------------------
        ### Args
        - `loss_prams`: loss를 구성하기 위해 사용되는 설정 값

        ### Returns
        - None

        ### Raises
        - `NotImplementedError`: 해당 함수는 구성이 필요함

        """
        raise NotImplementedError

    def Set_eval_holder(self, debug_prams: PARAMS):
        """ ### 학습 과정 중 생성되는 평가 인자 보관 자료형 구성 함수

        ------------------------------------------------------------------
        ### Args
        - `debug_prams`: 평가 인자 설정 값

        ### Returns
        - None

        ### Raises
        - None

        """
        _holder_name: List[str] = debug_prams["holder_name"]
        self.eval_holder = dict((
            _mode,
            dict((
                _name,
                [0.0 for _ in range(self.last_epoch, self.max_epoch)]
            ) for _name in _holder_name)
        ) for _mode in self.apply_mode)

    def Set_intput_n_target(
        self, device_info: device, data: Tensor | List[Tensor]
    ) -> Tuple[Tensor | List[Tensor], Tensor | List[Tensor], Any]:
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

    def Get_output(
        self,
        mode: Mode,
        model: Module,
        intput_data: Tensor | List[Tensor]
    ) -> Tensor | List[Tensor]:
        """ ### 입력 데이터에 따른 모델의 결과를 생성하는 함수

        ------------------------------------------------------------------
        ### Args
        - `mode`: 현재 학습 과정
        - `model`: 학습 모델
        - `intput_data`: 입력 데이터

        ### Returns
        - `output_data`: 입력 데이터에 따른 출력 데이터

        ### Raises
        - `error_type`: Method of handling according to error issues

        """
        if mode is Mode.TRAIN:
            return model(intput_data)
        with no_grad():
            return model(intput_data)

    def Core(
        self,
        epoch: int,
        mode: Mode,
        model: Module,
        output: Tensor | List[Tensor],
        target: Tensor | List[Tensor],
        optim: Optimizer,
        data_count: int,
        **kwarg
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
        file_dir: List[str] | None = None
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
        file_dir: List[str] | None = None
    ) -> Tuple[Module, Optimizer, LRScheduler | None]:
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

    def Decision_for_learning(
        self,
        epoch: int,
        model: Module,
        optim: Optimizer,
        scheduler: LRScheduler | None
    ):
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

    def Main_work(self, thred_num: int, prams: PARAMS):
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
        _device = device(
            f"cuda:{self.gpus[thred_num]}" if self.gpus else "cpu"
        )

        # set holder for debug and decision
        self.Set_eval_holder(prams["debug"])

        # dataset and dataloader
        _data_proc_prams: Dict[Mode, Dict[str, PARAMS]] = prams["data_proc"]
        _loaders = dict((
            _mode,
            self.Set_dataloader(
                self.Set_dataset(_mode, _prams["dataset_prams"]),
                _prams["dataloader_prams"]
            )) for _mode, _prams in _data_proc_prams.items()
        )
        # model and optimizer (with scheduler)
        _model = self.Set_model(_device, prams["model"])
        _optim, _scheduler = self.Set_optimizer(_model, prams["optimizer"])

        # load pretrained weight
        if self.last_epoch:
            _model, _optim, _scheduler = self.Load_weight(
                _model, _optim, _scheduler, "last"
            )

        # set loss
        self.Set_loss(prams["loss"])

        for _epoch in range(self.last_epoch, self.max_epoch):
            for _mode in self.apply_mode:
                _dataloader = _loaders[_mode]
                _data_ct = 0

                for _data in _dataloader:
                    _in, _target, _ = self.Set_intput_n_target(_device, _data)
                    _out = self.Get_output(_mode, _model, _in)

                    _data_ct = self.Core(
                        _epoch, _mode, _model, _out, _target, _optim, _data_ct
                    )

            if _scheduler is not None:
                _scheduler.step()

            # make decision for learning in each epoch
            self.Decision_for_learning(_epoch, _model, _optim, _scheduler)

    def Run(self, prams: PARAMS):
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
        self.Main_work(0, prams)
