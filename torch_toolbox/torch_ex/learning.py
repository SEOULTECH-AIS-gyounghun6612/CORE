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

from torch import load, save, device, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from python_ex.system import Path, String, File, Time
from python_ex.project import Template, Config, Debuging

from torch_ex.dataset import (Dataloader_Config, DataLoader)
from torch_ex.neural_network import (
    Model_Basement, Neural_Network_Config, Module)


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

    def __post_init__(self):
        self.max_epoch: int = 0
        self.max_data_ct_in_epoch: int = 0

    def Save_observer(
        self,
        file_name: str,
        file_dir: str,
        save_at_each_epoch: bool = True
    ):
        if save_at_each_epoch:
            _this_e = self.this_epoch
            _save_dir = Path.Join(str(_this_e), file_dir)
            _save_data = self.holder[_this_e]
        else:
            _save_dir = file_dir
            _save_data = self.holder
        File.Json.Write(file_name, _save_dir, _save_data)

    def Holder_init(self, max_epoch: int, mode_list: list[str]):
        self.max_epoch = max_epoch
        self.holder = dict((
            _ep,
            dict((
                _mode,
                dict((_name, (0, [])) for _name in self.observe_param)
            ) for _mode in mode_list)
        ) for _ep in range(max_epoch))

    def Set_log(
        self,
        standard_time: datetime,
        data_ct: int,
        **kwarg: tuple[int, float | list[float]]
    ):
        _mode = self.this_mode

        _max_ct = self.max_data_ct_in_epoch
        _max_e = self.max_epoch
        _this_e = self.this_epoch

        _str_e = String.Count_auto_aligning(
            _this_e, _max_e)
        _str_d = String.Count_auto_aligning(
            data_ct, _max_ct)

        _prefix = f"[{_mode[:5]}] [epoch]: {_str_e} [data]: {_str_d}"

        _sp_t = Time.Get_term(standard_time)
        _e_t = _sp_t / data_ct * _max_ct * (_max_e - _this_e)
        _finish_time = Time.Make_text_from(
            standard_time + _e_t - _sp_t,
            "%Y-%m-%d %H:%M:%S"
        )

        _param_list = self.observe_param
        _this_holder = self.holder[_this_e][_mode]
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

        _suffix = f"end to {_finish_time} {_log_text[:-1]}"

        Debuging.Progress_bar(
            data_ct, _max_ct, _prefix, _suffix, length=10)

    def Get_log(self, parameter_name: str):
        _this_holder = self.holder[self.this_epoch][self.this_mode]

        if parameter_name in _this_holder:
            return _this_holder[parameter_name]
        return (0, [])


@dataclass
class End_to_End_Config(Config):
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
        return [
            self.project_name,
            f"lr_{self.learning_rate}",
            "_".join([
                "_".join(
                    [_k, ] + _v.Get_summation()
                ) for _k, _v in self.dataloader.items()
            ]),
            "_".join(self.neural_network.Get_summation())
        ]


@dataclass
class Reinforcement_Based_on_Policy_Config(End_to_End_Config):
    policy_network: Neural_Network_Config = field(
        default_factory=Neural_Network_Config)

    def Build_learning(self) -> End_to_End:
        raise NotImplementedError

    def Get_summation(self):
        return super().Get_summation().append(
            "_".join(self.policy_network.Get_summation()))


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

    # build program obj in learning preprocee
    @abstractmethod
    def _Biuild_optimizer(
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

    def _preprocess(
        self,
        cfg: End_to_End_Config,
        device_info: device
    ) -> tuple[
        dict[Mode, tuple[int, DataLoader]],
        Model_Basement,
        list[Module],
        Optimizer,
        LRScheduler | None,
        dict[str, Any]
    ]:
        _dataloader = dict(
            (
                Mode(_k.upper()),
                _v.Builde_dataloder(_k)
            ) for _k, _v in cfg.dataloader.items() if _k in Mode.list()
        )
        _model, _loss_fns = cfg.neural_network.Build_model_n_loss(device_info)

        # set optim and scheduler
        _optim, _scheduler = self._Biuild_optimizer(_model, cfg.learning_rate)

        return (
            _dataloader,
            _model,
            _loss_fns,
            _optim,
            _scheduler,
            {})

    @abstractmethod
    def _Jump_to_cuda(self, data: Any, device_info: device) -> Any:
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
    def _Core(
        self,
        epoch: int,
        mode: Mode,
        data_ct: int,
        model: Model_Basement,
        loss_fn: list[Module],
        optim: Optimizer,
        data: dict[str, Tensor],
        **additional_data
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

    @abstractmethod
    def _Decision_to_learning_stop(self, observer: Observer, **kwarg) -> bool:
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

    def _Save_weight(
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

    def _Load_weight(
        self,
        model: Model_Basement,
        optim: Optimizer,
        scheduler: LRScheduler | None,
        file_dir: str | None = None
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
        _model_state_dict = load(Path.Join(f"{model.model_name}.h5", file_dir))
        model.load_state_dict(_model_state_dict["model"])

        _optim_dict = load(Path.Join(f"{model.model_name}_optim.h5", file_dir))
        if len(_optim_dict):
            optim.load_state_dict(_optim_dict["optim"])

        _scheduler_dict = load(
            Path.Join(f"{model.model_name}_scheduler.h5", file_dir))
        if len(_scheduler_dict) and scheduler is not None:
            scheduler.load_state_dict(_scheduler_dict["scheduler"])

        return model, optim, scheduler

    def _Main_work(
        self,
        thred_num: int,
        cfg: End_to_End_Config,
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
        _gpu_list = cfg.gpus
        _gpu_id = _gpu_list[thred_num] if _gpu_list else -1
        _device = device(f"cuda:{_gpu_id}" if _gpu_list else "cpu")

        (
            _dataloader, _model, _loss_fns, _optim, _scheduler, addtional
        ) = self._preprocess(cfg, _device)
        observer.max_data_ct_in_epoch = sum(
            _v[0] for _v in _dataloader.values())

        _this_e = cfg.this_epoch
        _max_e = cfg.max_epoch

        # load pretrained weight
        if _this_e:
            _file_dir = Path.Join(str(_this_e), self.result_root)
            _model, _optim, _scheduler = self._Load_weight(
                _model, _optim, _scheduler, _file_dir
            )

        # learning
        for _epoch in range(_this_e, _max_e):
            _save_dir = Path.Make_directory(str(_epoch), self.result_root)
            observer.this_epoch = _epoch
            _time_stamp = Time.Stamp()

            _data_ct = 0
            for _mode, (_, _loader) in _dataloader.items():
                observer.this_mode = str(_mode)

                _data_ct_in_mode = 0
                for _data in _loader:
                    _data_ct_in_mode, _mini_result = self._Core(
                        _epoch, _mode, _data_ct_in_mode,
                        _model, _loss_fns, _optim,
                        self._Jump_to_cuda(_data, _device),
                        **addtional
                    )
                    observer.Set_log(
                        _time_stamp,
                        _data_ct + _data_ct_in_mode,
                        **_mini_result
                    )
                _data_ct += _data_ct_in_mode

            # save the each epoch result
            self._Save_weight(
                model=_model,
                optim=_optim,
                scheduler=_scheduler,
                file_dir=_save_dir
            )
            observer.Save_observer(
                "log.json",
                self.result_root,
                cfg.is_save_each_epoch)

            # make decision for learning in each epoch
            if self._Decision_to_learning_stop(observer):
                break

            # !!! learning is continue !!!
            if _scheduler is not None:
                _scheduler.step()

    def _Set_result_dir(self, config: End_to_End_Config):
        self.result_root = Path.Make_directory(
            config.Get_summation(), self.result_root)

    def Run(self, config: End_to_End_Config, observer: Observer):
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
        config.Write_to("config.json", self.result_root)

        self._Main_work(0, config, observer)


class Reinforcement():
    class Base_on_Model(End_to_End):
        @abstractmethod
        def _Biuild_optimizer(
            self,
            model: tuple[Model_Basement, Model_Basement],
            learning_rate: float,
            **kwarg
        ) -> tuple[Optimizer, LRScheduler | None]:
            """ ###

            ------------------------------------------------------------------
            ### Args
            - `model`:
            - `optimizer_prams`:

            ### Returns
            - `Optimizer`:
            - `LRScheduler` or `None`:

            ### Raises
            - `NotImplementedError`:

            """
            raise NotImplementedError

        @abstractmethod
        def __Build_reward_function(
            self,
            cfg: Reinforcement_Based_on_Policy_Config,
            device_info: device
        ) -> Callable[[tuple[Any, ...]], Tensor]:
            raise NotImplementedError

        def _preprocess(
            self,
            cfg: Reinforcement_Based_on_Policy_Config,
            device_info: device
        ) -> tuple[
            dict[Mode, tuple[int, DataLoader]],
            tuple[Model_Basement, Model_Basement],
            list[Module],
            Optimizer,
            LRScheduler | None,
            dict[str, Any]
        ]:
            _dataloader = dict(
                (
                    Mode(_k.upper()),
                    _v.Builde_dataloder(_k)
                ) for _k, _v in cfg.dataloader.items() if _k in Mode.list()
            )
            _model, _loss_fns = cfg.neural_network.Build_model_n_loss(
                device_info)
            _policy, _policy_loss_fns = cfg.policy_network.Build_model_n_loss(
                device_info)
            _reward_func = self.__Build_reward_function(cfg, device_info)

            # set optim and scheduler
            _optim, _scheduler = self._Biuild_optimizer(
                (_model, _policy), cfg.learning_rate)

            return (
                _dataloader,
                (_model, _policy),
                _loss_fns + _policy_loss_fns,
                _optim,
                _scheduler,
                {"reward_func": _reward_func}
            )

        @abstractmethod
        def _Core(
            self,
            epoch: int,
            mode: Mode,
            data_ct: int,
            model: tuple[Model_Basement, Model_Basement],
            loss_fn: list[Module],
            optim: Optimizer,
            data: dict[str, Tensor],
            **additional_data
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

        def _Save_weight(
            self,
            model: tuple[Model_Basement, Model_Basement],
            optim: Optimizer | None = None,
            scheduler: LRScheduler | None = None,
            file_dir: str | None = None
        ):
            _model, _policy = model
            save(
                {"model": _policy.state_dict(), },
                Path.Join(f"{_policy.model_name}.h5", file_dir)
            )

            super()._Save_weight(_model, optim, scheduler, file_dir)

        def _Load_weight(
            self,
            model: tuple[Model_Basement, Model_Basement],
            optim: Optimizer,
            scheduler: LRScheduler | None,
            file_dir: str | None = None
        ):
            _model, _policy = model
            _policy_state_dict = load(
                Path.Join(f"{_policy.model_name}.h5", file_dir))
            _policy.load_state_dict(_policy_state_dict["model"])

            _model, optim, scheduler = super()._Load_weight(
                _model, optim, scheduler, file_dir)

            return (_model, _policy), optim, scheduler

    class Base_on_Value(End_to_End):
        ...
