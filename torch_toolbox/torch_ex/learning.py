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
from typing import Any, Callable
from datetime import datetime

from torch import load, save, device
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from python_ex.system import Path, String, File, Time
from python_ex.project import Template, Config, Debuging

from torch_ex.dataset import (Dataloader_Config)
from torch_ex.neural_network import (Model_Basement, Neural_Network_Config)


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


@dataclass
class Observer():
    observe_param: list[str] = field(default_factory=list)

    this_epoch: int = 0
    this_mode: str = "train"
    holder: dict[int,                           # epoch
                 dict[str,                      # train mode
                      dict[str,                 # parameter_name
                           tuple[int,           # data count
                                 list[float]    # log
                                 ]]]] = field(default_factory=dict)

    def Save_observe(self, file_name: str, file_dir: str):
        _this_epoch = self.this_epoch
        _save_dir = Path.Join(str(_this_epoch), file_dir)
        File.Json.Write(file_name, _save_dir, self.holder[_this_epoch])

    def Holder_init(self, max_epoch: int, mode_list: list[str]):
        self.holder = dict((
            _ep,
            dict((
                _mode,
                dict((_name, (0, [])) for _name in self.observe_param)
            ) for _mode in mode_list)
        ) for _ep in range(max_epoch))

    def Set_log(self, **kwarg: tuple[int, float | list[float]]):
        _param_list = self.observe_param
        _this_holder = self.holder[self.this_epoch][self.this_mode]

        _log_text = ""

        for _p_name, _logging_data in kwarg.items():
            if _p_name in _param_list:
                _data_ct, _log = _this_holder[_p_name]
                _this_ct, _this_value = _logging_data
                _data_total = _data_ct + _this_ct

                if isinstance(_this_value, list):
                    _log += _this_value
                else:
                    _log.append(_this_value)

                _this_holder[_p_name] = (_data_total, _log)
                _value = sum(_log) / _data_total
                _log_text += f"{_p_name}: {_value:0>6.3f} "

        return _log_text[:-1]

    def Get_log(self, parameter_name: str):
        _this_holder = self.holder[self.this_epoch][self.this_mode]

        if parameter_name in _this_holder:
            return _this_holder[parameter_name]
        return (0, [])


class Process_Config():
    @dataclass
    class End_to_End(Config):
        project_name: str = "default_name"

        # epoch
        max_epoch: int = 50
        this_epoch: int = 0
        is_save_each_epoch: bool = True

        # optim and scheduler
        learning_rate: float = 0.005

        dataloader: dict[str, Dataloader_Config] = field(
            default_factory=lambda: {
                "train": Dataloader_Config(),
                "validation": Dataloader_Config(),
            })

        neural_network: Neural_Network_Config = field(
            default_factory=Neural_Network_Config)

        gpus: list[int] = field(default_factory=lambda: [0])
        observe_param: list[str] = field(default_factory=lambda: ["loss"])

        def Build_observe(self):
            _mode_list = list(self.dataloader.keys())
            _observer = Observer(
                self.observe_param, self.this_epoch, _mode_list[0])
            _observer.Holder_init(self.max_epoch, list(self.dataloader.keys()))

            return _observer

        def Build_learning(self) -> End_to_End:
            raise NotImplementedError

        def Get_summation(self):
            _data_str_info = "_".join([
                "_".join(
                    [_k, ] + _v.Get_summation()
                ) for _k, _v in self.dataloader.items()
            ])

            _model_info_str = "_".join(self.neural_network.Get_summation())

            return [
                self.project_name,
                _model_info_str,
                f"lr_{self.learning_rate}",
                _data_str_info
            ]


class End_to_End(Template, ABC):
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
    @abstractmethod
    def Set_optimizer(
        self, model: Model_Basement, learning_rate: float, **kwarg
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
        model: Model_Basement,
        loss_fn: list[Callable],
        optim: Optimizer,
        **data
    ) -> tuple[int, dict[str, tuple[int, float | list[float]]]]:
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

    def _log_display(
        self,
        mode: Mode,
        epoch: int, max_epoch: int,
        this_data_ct: int, data_ct: int, total_ct: int,
        st_time: datetime,
        log_str: str,
    ):
        data_ct += this_data_ct
        _prefix = f"[{str(mode)}] "
        _prefix += f"[epoch]: {String.Count_auto_aligning(epoch, max_epoch)} "
        _prefix += f"[data]: {String.Count_auto_aligning(data_ct, total_ct)}"

        _each = (Time.Get_term(st_time) / this_data_ct)
        _remain_str = Time.Make_text_from(
            st_time + (_each * ((max_epoch - epoch - 2) * total_ct - data_ct)),
            '%Y-%m-%d %H:%M:%S'
        )
        _suffix = f"finish at {_remain_str} {log_str}"

        Debuging.Progress_bar(data_ct, total_ct, _prefix, _suffix, length=10)

        return data_ct

    @abstractmethod
    def Decision_to_learning_stop(self, observer: Observer, **kwarg) -> bool:
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

    def Save_weight(
        self,
        model: Model_Basement,
        optim: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        file_dir: str | None = None
    ):
        """ ### 학습 과정의 주요 인자값을 저장하는 함수

        ------------------------------------------------------------------
        ### Args
        - `model`: 학습 대상 모델
        - `optim`: 학습에 사용되는 optimizer
        - `scheduler`: 학습에 사용되는 scheduler. 없는 경우 저장하지 않음.
        - `file_dir`: 저장되는 파일이 위치할 디렉토리

        ### Returns
        - None

        ### Raises
        - None

        """
        save(
            {"model": model.state_dict(), },
            Path.Join(f"{model.model_name}.h5", file_dir)
        )

        if optim is not None:
            save(
                {"optim": optim.state_dict(), },
                Path.Join(f"{model.model_name}_optim.h5", file_dir)
            )

        if scheduler is not None:
            save(
                {"scheduler": scheduler.state_dict(), },
                Path.Join(f"{model.model_name}_scheduler.h5", file_dir)
            )

    def Load_weight(
        self,
        model: Model_Basement,
        optim: Optimizer,
        scheduler: LRScheduler | None,
        file_dir: list[str] | None = None
    ) -> tuple[Model_Basement, Optimizer, LRScheduler | None]:
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
        _model_dir = _file_dir + [f"{model.model_name}.h5"]
        _model_state_dict = load(Path.Join(_model_dir, self.result_root))
        model.load_state_dict(_model_state_dict["model"])

        _optim_dir = _file_dir + [f"{model.model_name}_optim.h5"]
        _optim_dict = load(Path.Join(_optim_dir, self.result_root))
        if len(_optim_dict):
            optim.load_state_dict(_optim_dict["optim"])

        _scheduler_dir = _file_dir + [f"{model.model_name}_scheduler.h5"]
        _scheduler_dict = load(Path.Join(_scheduler_dir, self.result_root))
        if len(_scheduler_dict) and scheduler is not None:
            scheduler.load_state_dict(_scheduler_dict["scheduler"])

        return model, optim, scheduler

    def _Main_work(
        self,
        thred_num: int,
        cfg: Process_Config.End_to_End,
        observer: Observer
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
                Mode(_k.upper()),
                _v.Builde_dataloder(_k)
            ) for _k, _v in cfg.dataloader.items() if _k in Mode.list()
        )

        _gpu_list = cfg.gpus
        _gpu_id = _gpu_list[thred_num] if _gpu_list else -1
        _device = device(f"cuda:{_gpu_id}" if _gpu_list else "cpu")

        # set model and loss function
        _model, _loss_fns = cfg.neural_network.Build_model_n_loss(_device)

        # set optim and scheduler
        _optim, _scheduler = self.Set_optimizer(_model, cfg.learning_rate)

        # load pretrained weight
        if cfg.this_epoch:
            _model, _optim, _scheduler = self.Load_weight(
                _model, _optim, _scheduler
            )

        _max_e = cfg.max_epoch

        # learning
        for _epoch in range(cfg.this_epoch, _max_e):
            observer.this_epoch = _epoch

            for _mode, _loader in _dataloader.items():
                observer.this_mode = str(_mode)

                _data_ct = 0
                _max_data_ct = len(_loader.dataset)
                _st = Time.Stamp()
                for _data in _loader:
                    _this_data_ct, _mini_result = self.Core(
                        _epoch, _mode,
                        _model, _loss_fns, _optim,
                        **self.Jump_to_cuda(_data, _device)
                    )
                    _log_txt = observer.Set_log(**_mini_result)
                    _data_ct = self._log_display(
                        _mode,
                        _epoch, _max_e,
                        _this_data_ct, _data_ct, _max_data_ct,
                        _st,
                        _log_txt
                    )
                    _st = Time.Stamp()

            # save the each epoch result
            self.Save_weight(
                _model,
                _optim,
                _scheduler,
                Path.Make_directory(str(_epoch), self.result_root)
            )
            observer.Save_observe("log.json", self.result_root)

            # make decision for learning in each epoch
            if self.Decision_to_learning_stop(observer):
                break

            # !!! learning is continue !!!
            if _scheduler is not None:
                _scheduler.step()

    def _Set_result_dir(self, config: Process_Config.End_to_End):
        self.result_root = Path.Join(config.Get_summation(), self.result_root)

    def Run(self, config: Process_Config.End_to_End, observer: Observer):
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
        self._Set_result_dir(config)
        self._Main_work(0, config, observer)

        config.Write_to("config.json", self.result_root)
