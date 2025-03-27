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
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable
from datetime import datetime

from torch import Tensor, device
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from python_ex.system import String, Time_Utils
from python_ex.file import Json
from python_ex.project import Config
from python_ex.project import Project_Template

from .dataset import (
    Data_Config, Dataloader_Config, Custom_Dataset, Build_dataloader)
from .neural_network import Custom_Model, Model_Config


class Mode(String.String_Enum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()


@dataclass
class Optimizer_Config(Config.Basement):
    learning_rate: float = 0.005


@dataclass
class Learning_Config(Config.Basement):
    # epoch
    max_epoch: int = 50
    this_epoch: int = 0
    is_save_each_epoch: bool = True

    # multi gpus
    gpus: list[int] = field(default_factory=lambda: [0])

    # dataset
    dataset_cfg_files: dict[str, str | None] = field(default_factory=lambda: {
        "train": "./cfg/dataset/data.json",
        "validation": "./cfg/dataset/data.json",
        "test": None
    })
    dataset_cfg: dict[Mode, Data_Config] = field(init=False)

    # dataloader
    dataloader_arg: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "train": {
            "batch_size": 32, "shuffle": True, "num_workers": 2,
            "collate_fn": None, "drop_last": False},
        "validation": {
            "batch_size": 16, "shuffle": False, "num_workers": 2,
            "collate_fn": None, "drop_last": False},
        "test": {
            "batch_size": 16, "shuffle": False, "num_workers": 2,
            "collate_fn": None, "drop_last": False}
    })
    dataloader_cfg: dict[Mode, Dataloader_Config] = field(init=False)

    # model
    model_cfg_file: str = "./cfg/model/main_model.json"
    model_cfg: Model_Config = field(init=False)

    # optim and scheduler
    optim_arg: dict[str, Any] = field(default_factory=lambda: {

    })
    optim_cfg: Optimizer_Config = field(init=False)

    # logger -> {mode: {epoch: loss_logger(dict)}}
    logger: dict[int, dict[str, dict]] = field(default_factory=dict)

    def __post_init__(self):
        # set data cfg
        _dataset_cfg_files = self.dataset_cfg_files
        _dataset_cfg = {}
        for _k, _file in _dataset_cfg_files.items():
            if not _file:
                continue
            _mode: Mode = getattr(Mode, _k.upper())
            _is_exist, _dataset_arg = Json.Read_from(Path(_file))
            if not _is_exist:
                continue
            _dataset_cfg[_mode] = Data_Config(**_dataset_arg)
        self.dataset_cfg = _dataset_cfg

        # set dataloader cfg
        _dataloader_arg = self.dataloader_arg
        self.dataloader_cfg = dict((
            getattr(Mode, _k.upper()), Dataloader_Config(**_arg)
        ) for _k, _arg in _dataloader_arg.items())

        # set model cfg
        _is_exist, _model_cfg = Json.Read_from(Path(self.model_cfg_file))

        if _is_exist:
            self.model_cfg = Model_Config(**_model_cfg)
        else:
            self.model_cfg = Model_Config()

        # set optim and schedule cfg
        _optim_arg = self.optim_arg
        self.optim_cfg = Optimizer_Config(**(_optim_arg if _optim_arg else {}))

    def Config_to_dict(self) -> dict[str, Any]:
        return {
            "max_epoch": self.max_epoch,
            "this_epoch": self.this_epoch,
            "is_save_each_epoch": self.is_save_each_epoch,
            "gpus": self.gpus,
            "dataset_cfg_files": self.dataset_cfg_files,
            "dataloader_arg": self.dataloader_arg,
            "model_cfg_file": self.model_cfg_file,
            "optim_arg": self.optim_arg,
            "logger": self.logger
        }


LOSSES = tuple[Callable[..., Tensor], ...]


class End_to_End(Project_Template):
    project_cfg: Learning_Config

    def _Get_dataset(self, data_cfg: Data_Config) -> Custom_Dataset:
        raise NotImplementedError

    def _Get_model_n_loss(
        self, model_cfg: Model_Config, device_info: device
    ) -> tuple[Custom_Model, LOSSES]:
        raise NotImplementedError

    def _Get_optim(
        self, model: Custom_Model, optim_cfg: Optimizer_Config
    ) -> tuple[Optimizer, LRScheduler | None]:
        raise NotImplementedError

    def _Learning(
        self, data: Any, model: Custom_Model, loss: LOSSES, optim: Optimizer,
        holder: dict, mode: Mode, epoch: int, st_time: datetime
    ):
        raise NotImplementedError

    def _Lerning_decision(self, holder: dict) -> bool:
        return True

    def _Main_work(self, thred_num: int, cfg: Learning_Config):
        # get the device info
        _gpus = cfg.gpus
        _device = device(
            f"cuda:{_gpus[thred_num]}" if len(_gpus) > thred_num else "cpu")

        # prepare learning
        _logger = cfg.logger
        _d_cfg = cfg.dataset_cfg
        _d_loader_cfg = cfg.dataloader_cfg
        _dataloader = dict((
            _m, Build_dataloader(_d_loader_cfg[_m], self._Get_dataset(_d_cfg))
        ) for _m, _d_cfg in _d_cfg.items())

        _model, _losses = self._Get_model_n_loss(cfg.model_cfg, _device)
        _optim, _scheduler = self._Get_optim(_model, cfg.optim_cfg)

        # use multi gpu
        if len(_gpus) >= 2:
            ...  # not yet

        _this_e = cfg.this_epoch
        _max_e = cfg.max_epoch

        # load pretrained weight
        if _this_e:
            ...  # not yet

        # 학습 구조
        for _epoch in range(_this_e, _max_e):
            _this_path = self.result_path / f"{_this_e:0>6d}"
            _this_path.mkdir(exist_ok=True)

            _epoch_logger: dict[str, dict] = {}
            _st_time = Time_Utils.Stamp()
            for _m, _d in _dataloader.items():
                _this_mode = str(_m)
                _this_looger = {}
                _this_path = _this_path / _this_mode
                _this_path.mkdir(exist_ok=True)

                _model = _model.train() if _m is Mode.TRAIN else _model.eval()

                _debugging = _this_looger, _m, _epoch, _st_time

                # 실제 데이터를 이용한 학습 진행
                for _data in _d:
                    self._Learning(_data, _model, _losses, _optim, *_debugging)

                _epoch_logger[_this_mode] = _this_looger
            _logger[_epoch] = _epoch_logger

            # 현재까지 학습 결과를 바탕으로 학습 진행 여부 결정
            if self._Lerning_decision(_logger):
                break

            # 학습 속행
            if _scheduler is not None:
                _scheduler.step()

    def Run(self):
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
        Json.Write_to(
            self.result_path / "config.json",
            self.project_cfg.Config_to_dict()
        )

        self._Main_work(0, self.project_cfg)
