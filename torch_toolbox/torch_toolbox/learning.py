# learning.py
"""
### 신경망 학습 과정의 기본 구조 및 실행 지원 모듈
학습 설정, 데이터 로딩, 모델 실행, 결과 저장 등 학습 파이프라인 전반을 관리.

------------------------------------------------------------------------
### Requirement
    - torch >=2.2
    - torchsummary
    - python_ex

### Structure
    - Mode: 학습 과정을 표현하기 위한 열거형
    - Optimizer_Config: 옵티마이저 및 학습률 스케줄러 설정 데이터 클래스.
    - Learning_Config: 학습 전반의 주요 설정(시드, 경로, 에폭, GPU 등) 관리 데이터 클래스.
    - FUNC_LOSS: 손실 함수 타입 정의.
    - End_to_End: End-to-End 학습 파이프라인 추상 기본 클래스.
        - Get_result_path: 결과 저장 경로 생성 및 관리.
        - __Get_dataset__: 모드별 데이터셋 및 추가 정보 반환 (하위 클래스에서 구현).
        - __Get_model_n_loss__: 모델 및 손실 함수 반환 (하위 클래스에서 구현).
        - __Get_optim__: 옵티마이저 및 스케줄러 반환 (하위 클래스에서 구현).
        - __Run_one_epoch__: 단일 에폭의 학습/검증/테스트 실행 (하위 클래스에서 구현).
        - __Should_stop_early__: 조기 종료 조건 확인 및 현재 에폭 결과 저장.
        - Load_model: 저장된 모델 가중치를 로드.
        - __Recovey_model__: 지정된 에폭의 모델 가중치 복구.
        - __Model_running__: 전체 에폭에 대한 학습 및 검증/테스트 루프 실행.
        - __Run_train_and_validation__: 학습 및 검증 프로세스 전체 실행 (단일/다중 GPU 지원).
        - Train_with_validation: 학습 및 검증 프로세스 시작점.
        - __Run_test__: 테스트 프로세스 전체 실행 (단일 GPU).
        - Test: 테스트 프로세스 시작점.

"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable
from datetime import datetime

from torch import (
    Tensor, device, distributed as dist, multiprocessing as torch_mp,
    save, load, manual_seed
)
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from torch.nn.parallel import DistributedDataParallel as DDP

from python_ex.system import Time_Utils
from python_ex.file import Utils
from python_ex.project import Config, Project_Template

from . import Mode

from .dataset import (
    Data_Config, Dataloader_Config, Custom_Dataset,
    DataLoader, DistributedSampler, Build_loader
)
from .neural_network import Custom_Model, Model_Config


@dataclass
class Optimizer_Config(Config.Basement):
    """ ### 옵티마이저 및 학습률 스케줄러 설정 데이터 클래스

    학습에 사용될 옵티마이저의 학습률 및 스케줄러 관련 설정을 정의.

    ------------------------------------------------------------------
    ### Args
    - learning_rate (optional): 옵티마이저 학습률. (default=0.005)
    - scheduler_cfg (optional): 학습률 스케줄러 설정 딕셔너리. (default={})
    """
    learning_rate: float = 0.005
    scheduler_cfg: dict = field(default_factory=dict)


@dataclass
class Learning_Config(Config.Basement):
    seed: int = 930110
    # result directory
    result_dir: str = "./result"
    recovery_path: str | None = None

    # epoch
    max_epoch: int = 50
    this_epoch: int = 0
    is_save_each_epoch: bool = True

    # gpus
    gpus: list[int] = field(default_factory=lambda:[0])
    # with multi
    host_name: str = "localhost"  # "localhost" or "127.0.0.1"
    port_num: int = 12355

    world_size: int = 1
    node_rank_offset: int = 0

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

    def __post_init__(self):
        # set data cfg
        _dataset_cfg_files = self.dataset_cfg_files
        _dataset_cfg = {}
        for _k, _file in _dataset_cfg_files.items():
            if not _file:
                continue
            _is_exist, _dataset_arg = Utils.Read_from(Path(_file))
            if not _is_exist:
                continue
            _dataset_cfg[Mode(_k.lower())] = Data_Config(**_dataset_arg)
        self.dataset_cfg = _dataset_cfg

        # set dataloader cfg
        _dataloader_arg = self.dataloader_arg
        self.dataloader_cfg = dict((
            Mode(_k.lower()), Dataloader_Config(**_arg)
        ) for _k, _arg in _dataloader_arg.items())

        # set model cfg
        _is_exist, _model_cfg = Utils.Read_from(Path(self.model_cfg_file))

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
            "optim_arg": self.optim_arg
        }


FUNC_LOSS = Callable[
    [dict[str, Tensor], dict[str, Tensor], dict[str, list[float]]],
    Tensor
]


class End_to_End(Project_Template):
    def Get_result_path(self, config: Learning_Config) -> Path:
        _root_dir = Path(config.result_dir) / self.project_name
        _root_dir /= "_".join([
            f"{_m.capitalize()[:2]}_{_d_cfg.name}" for (
                _m, _d_cfg
            ) in config.dataset_cfg.items()
        ])

        _r_path = config.recovery_path
        if _r_path is not None and (_root_dir / _r_path).is_dir():
            return _root_dir / _r_path

        config.this_epoch = 0
        _r_path = Time_Utils.Make_text_from(
            Time_Utils.Stamp(), "%Y-%m-%dT%H:%M:%S"
        )
        _result_dir = _root_dir / _r_path
        _result_dir.mkdir(exist_ok= True)

        return _result_dir

    def __Get_dataset__(
        self, mode: Mode, data_cfg: Data_Config
    ) -> tuple[Custom_Dataset, Callable | None]:
        raise NotImplementedError

    def __Get_model_n_loss__(
        self, model_cfg: Model_Config, device_info: device
    ) -> tuple[Custom_Model, FUNC_LOSS]:
        raise NotImplementedError

    def __Get_optim__(
        self, model: Custom_Model, optim_cfg: Optimizer_Config
    ) -> tuple[Optimizer, LRScheduler | None]:
        raise NotImplementedError

    def __Run_one_epoch__(
        self,
        epoch: int, st_time: datetime,
        mode: Mode, dataloader: DataLoader, device_info: device,
        model: Custom_Model | DDP, loss: FUNC_LOSS, optim: Optimizer,
        save_path: Path
    ) -> dict[str, list[float]]:
        raise NotImplementedError

    def __Should_stop_early__(
        self,
        holder: dict, model: Custom_Model | DDP, save_path: Path, rank: int
    ) -> bool:
        Utils.Write_to(save_path / f"{rank:0>3d}_result.json", holder)

        if not rank:
            _state: dict[str, Any] = model.module.state_dict() if isinstance(
                model, DDP
            ) else model.state_dict()
            save(_state, save_path / "model.pth")

        return False

    def Load_model(
        self, model: Custom_Model, save_path: Path, device_info: device
    ):
        model.load_state_dict(load(save_path, map_location=device_info))

    def __Recovery_model__(
        self, model: Custom_Model, epoch: int, device_info: device
    ):
        self.Load_model(
            model,
            self.result_path / f"{epoch:0>6d}" / "model.pth",
            device_info
        )

    def __Model_running__(
        self, this_epoch: int, max_epoch: int, rank: int,
        model: Custom_Model | DDP, losses: FUNC_LOSS,
        dataloader: dict[Mode, tuple[DataLoader, DistributedSampler | None]],
        device_info: device,
        optim: Optimizer, scheduler: LRScheduler | None
    ):
        _logger = {}

        for _epoch in range(this_epoch, max_epoch):
            _path = self.result_path / f"{_epoch:0>6d}"

            if not rank:
                _path.mkdir(exist_ok=True)

            _epoch_logger: dict[str, dict] = {}
            _st_time = Time_Utils.Stamp()

            for _m, (_dataset, _sampler) in dataloader.items():
                if _sampler is not None:
                    _sampler.set_epoch(_epoch)

                _m_str = str(_m)
                _m_path = _path / _m_str
                _m_path.mkdir(exist_ok=True)

                # 실제 데이터를 이용한 학습 진행
                _epoch_logger[_m_str] = self.__Run_one_epoch__(
                    _epoch, _st_time,
                    _m, _dataset, device_info,
                    model.train() if _m == "train" else model.eval(),
                    losses, optim,
                    _m_path
                )

            _logger[_epoch] = _epoch_logger

            # 현재까지 학습 결과를 바탕으로 학습 진행 여부 결정
            if self.__Should_stop_early__(_epoch_logger, model, _path, rank):
                break

            # 학습 속행
            if scheduler is not None:
                scheduler.step()
        return _logger

    def __Run_train_and_validation__(
        self, local_rank: int, cfg: Learning_Config
    ):
        manual_seed(cfg.seed)
        # get the device info
        _w_size = cfg.world_size
        _is_mp = _w_size > 1
        _rank = cfg.node_rank_offset + local_rank
        _device = device(f"cuda:{cfg.gpus[local_rank]}")

        # prepare learning
        _model, _losses = self.__Get_model_n_loss__(cfg.model_cfg, _device)
        _optim, _scheduler = self.__Get_optim__(_model, cfg.optim_cfg)

        # load pretrained weight
        _this_e = cfg.this_epoch
        if _this_e and not _rank:
            self.__Recovery_model__(_model, _this_e, _device)

        if _is_mp:
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{cfg.host_name}:{cfg.port_num}",
                world_size=_w_size,
                rank=_rank
            )
            _model = DDP(
                _model, device_ids=[local_rank], output_device=local_rank,
                # find_unused_parameters=True
                # # 모델의 일부 파라미터가 forward에서 사용되지 않을 경우
            )
            print((
                f"Info: [Rank {_rank}/{_w_size}] DDP initialized"
                "Model wrapped with DDP."
            ))
        else:
            print((
                "Info: Model ready for single-process execution on "
                f"device '{_device}'."
            ))

        # ready for train data
        _d_cfg = cfg.dataset_cfg
        _d_loader_cfg = cfg.dataloader_cfg
        _d_dict = dict((
            _m,
            Build_loader(
                _d_loader_cfg[_m],
                *self.__Get_dataset__(_m, _d_cfg),
                is_multi_process=_is_mp,
                world_size=_w_size,
                rank=_rank
            )
        ) for _m, _d_cfg in _d_cfg.items())

        # 학습 시작
        self.__Model_running__(
            _this_e, cfg.max_epoch, _rank, _model, _losses,
            dict((_m, _d) for _m, _d in _d_dict.items() if _m != "test"),
            _device,
            _optim, _scheduler
        )

        if _is_mp and dist.is_initialized():
            dist.destroy_process_group()
            if _rank == 0:
                print("DDP process group destroyed.")

    def Train_with_validation(self, config: Learning_Config):
        """ ### 학습 실행 함수

        ------------------------------------------------------------------
        ### Args
        - `prams`: 학습에 사용되는 인자값

        ### Returns
        - None

        ### Raises
        - None

        """
        Utils.Write_to(
            self.result_path / "config.yaml", config.Config_to_dict()
        )

        # multi process check
        _w_size = config.world_size
        _n_size = len(config.gpus) if config.gpus else 0

        if _n_size > _w_size:
            raise ValueError

        if _n_size:
            if _w_size > 1:  # use multi gpu
                # do multi process
                torch_mp.spawn(
                    self.__Run_train_and_validation__,
                    args=(config),
                    nprocs=_n_size,
                    join=True
                )  

            else:  # use single gpu
                self.__Run_train_and_validation__(0, config)

        else:
            # in this code not use train by CPU
            raise ValueError

    def __Run_test__(
        self, local_rank: int, cfg: Learning_Config
    ):
        # get the device info
        _rank = cfg.node_rank_offset + local_rank
        _device = device(f"cuda:{cfg.gpus[local_rank]}")

        # prepare learning
        _model, _losses = self.__Get_model_n_loss__(cfg.model_cfg, _device)
        _optim, _scheduler = self.__Get_optim__(_model, cfg.optim_cfg)

        # load pretrained weight
        _this_e = cfg.this_epoch
        if _this_e and not _rank:
            self.__Recovery_model__(_model, _this_e, _device)

        print((
            "Info: Model ready for single-process execution on "
            f"device '{_device}'."
        ))

        # ready for train data
        _d_cfg = cfg.dataset_cfg
        _d_loader_cfg = cfg.dataloader_cfg
        _d_dict = dict((
            _m,
            Build_loader(
                _d_loader_cfg[_m],
                *self.__Get_dataset__(_m, _d_cfg),
                is_multi_process=False
            )
        ) for _m, _d_cfg in _d_cfg.items())

        # 학습 시작
        self.__Model_running__(
            _this_e, cfg.max_epoch, _rank, _model, _losses,
            dict((_m, _d) for _m, _d in _d_dict.items() if _m == "test"),
            _device,
            _optim, _scheduler
        )

    def Test(self, config: Learning_Config):
        """ ### 학습 실행 함수

        ------------------------------------------------------------------
        ### Args
        - `prams`: 학습에 사용되는 인자값

        ### Returns
        - None

        ### Raises
        - None

        """
        Utils.Write_to(
            self.result_path / "config.yaml", config.Config_to_dict()
        )

        # multi process check
        _n_size = len(config.gpus) if config.gpus else 0

        if _n_size:
            self.__Run_test__(0, config)
        else:
            # in this code not use train by CPU
            raise ValueError
