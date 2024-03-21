from __future__ import annotations
from typing import Dict, List, Tuple, Type, Any, Callable
from dataclasses import dataclass, field

from enum import Enum
from collections import deque

from torch import distributed, save, load, Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from torch.autograd.grad_mode import no_grad
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from python_ex._System import Path, File
from python_ex._Project import Debuging, Config
from python_ex._Vision import cv2

from ._Utils import System_Utils
from ._Dataset import Data, Data_Config
from ._Model_n_Optim import Model, Optim, Scheduler, Model_and_Optimizer_Config


# -- DEFINE CONSTANT -- #
MAIN_RANK: int = 0


# -- Main code -- #
class Learning():
    class Mode(Enum):
        TRAIN = "train"
        VALIDATION = "val"
        TEST = "test"

    @dataclass
    class Dataloader_Config():
        dataset_process: Data.Process.Basement
        batch_size_per_node: int
        num_worker_per_node: int
        collate_fn: Callable | None = None
        drop_last: bool = True

        def _Get_dataloader(self, sampler: DistributedSampler | None = None):
            return DataLoader(
                dataset=self.dataset_process,
                batch_size=self.batch_size_per_node,
                num_workers=self.num_worker_per_node,
                collate_fn=self.collate_fn,
                sampler=sampler,
                drop_last=self.drop_last
            )

    @dataclass
    class Reward_Opt():
        dataset_process: Data.Process.Basement
        batch_size_per_node: int
        num_worker_per_node: int
        collate_fn: Callable | None = None
        drop_last: bool = True

        def _Get_Dataloader(self, sampler: DistributedSampler | None):
            return DataLoader(
                dataset=self.dataset_process,
                batch_size=self.batch_size_per_node,
                num_workers=self.num_worker_per_node,
                collate_fn=self.collate_fn,
                sampler=sampler,
                drop_last=self.drop_last
            )

    class Process():
        class Basement():
            """
            ### 모델 학습을 위한 기본 모듈

            -------------------------------------------------------------------------------------------
            ## Argument & Parameters
            - project_name : 훈련이 진행되는 프로젝트 이름
            - description : 훈련의 목적 및 세부 설명
            - save_root : 훈련 과정 및 결과를 저장하기 위한 경로
            - max_epoch : 훈련에 적용하고자 하는 최대 epoch
            - last_epoch : 훈련의 시작 epoch
            - display_term : 훈련의 진행 사항 중간 보고 간격.
            -------------------------------------------------------------------------------------------
            """
            def __init__(self, project_name: str, description: str, result_root: str, max_epoch: int, last_epoch: int = -1, display_term: float | int = 0.1):
                self.project_name = project_name
                self.description = description
                self.result_root = result_root

                self.max_epoch = max_epoch
                self.last_epoch = last_epoch

                self.display_term = display_term

                self.is_multi_gpu = False
                self.multi_protocal = None
                self.world_size = 1

                # debugging
                _debug_process_text = f"Set the Learning for {project_name}.\n\n"
                _debug_process_text += f"This Learing work to at {max_epoch} epoch from {last_epoch + 1} epoch.\n"
                print(_debug_process_text)

            #  ----------------- #
            def _Set_dataloader_option(self, dataloader_opt_block: Dict[Learning.Mode, Learning.Dataloader_Config]):
                """
                ### 훈련 데이터 설정 할당

                -------------------------------------------------------------------------------------------
                ## Parameters
                    data_process (Dataset_Process)
                        : 훈련용 데이터 생성 모듈
                    batch_size_per_node (int)
                        : 학습 데이터의 mini batch 크기
                    num_worker_per_node (int)
                        : 학습 데이터 생성 프로세서 할당 수

                ## Returns
                    None

                -------------------------------------------------------------------------------------------
                """
                # dataloader
                self.mode_list: List[Learning.Mode] = list(dataloader_opt_block.keys())
                self.dataloader_param = dataloader_opt_block

                # set the save dir
                _working_day = Debuging.Time._Apply_text_form(Debuging.Time._Stemp(), True, "%Y-%m-%d")
                _trial_num = 0
                while True:
                    _result_dir = Path._Join([self.project_name, _working_day, f"trial_{_trial_num:0>3d}"], self.result_root)
                    if not Path._Exist_check(_result_dir, Path.Type.DIRECTORY):
                        break
                    else:
                        _trial_num += 1

                self.result_root = Path._Make_directory(_result_dir, self.result_root)

                # debug the progress mode list in this learning
                _debug_process_text = "This learning process, that consist of "
                if len(self.mode_list) >= 2:
                    _debug_process_text += f"{', '.join(_mode.value for _mode in self.mode_list[: -1])} and {self.mode_list[-1].value}.\n"
                else:
                    _debug_process_text = f"{self.mode_list[-1].value}.\n"

                # debug the term of display for in process result
                _debug_process_text += "Interim reporting on the process are conducted at "
                if isinstance(self.display_term, float):
                    _debug_process_text += f"intervals of {self.display_term} times the total length of each learning process.\n"
                else:
                    _debug_process_text += f"{self.display_term} intervals for each learning process.\n"
                _debug_process_text = f"Result of this learing, save at root directory: {self.result_root}\n"
                print(_debug_process_text)

            def _Set_model_n_optim(
                self,
                model: Model,
                optim: Optimizer,
                scheduler: Scheduler.Basement | None
            ):
                """
                ### 훈련을 위한 모델과 Optimizer 설정

                -------------------------------------------------------------------------------------------
                ## Parameters
                    model ()
                        : 
                    optim (Optimizer)
                        : 훈련에서 사용되는 Optimizer
                    scheduler (float)
                        : Optimizer의 학습 비율 변경에 사용되는 scheduler

                ## Returns
                    None

                -------------------------------------------------------------------------------------------
                """
                self._model = model
                self._optim = optim
                self._scheduler = scheduler

            def _Set_GPU_option(
                self,
                min_of_memory: float | int | None = None,
                max_gpu_count: int = 1,
                device_num: int = 0,
                world_size: int = 1,
                multi_protocal: str | None = "tcp://127.0.0.1:10001"
            ):
                """
                ### 멀티 프로세서 관련 설정

                ### Set multi process setting

                -------------------------------------------------------------------------------------------
                ## Parameters
                    gpu_count (List[Tuple[int, str]])
                        : 해당 단말에서 사용하고자 하는 최대 GPU 개수
                    device_rank (int)
                        : 해당 코드를 실행 하고자 하는 단말장치의 식별번호 시작값
                    world_size (int)
                        : 훈련에 사용되는 전체 프로세서 갯수
                    multi_protocal (str | None)
                        : 다른 프로세서와 통신 설정

                ## Returns
                    None

                -------------------------------------------------------------------------------------------
                """
                print("Set the multiprocess option.")
                _gpu_info = System_Utils.Cuda._Get_gpu_list(min_of_memory)

                assert _gpu_info, "Check the GPU"

                if max_gpu_count > 2:
                    self.is_multi_gpu = True

                    self.world_size = world_size
                    self.device_num = device_num
                    self.multi_protocal = multi_protocal

                    # in later check it for this code must need
                    cv2.setNumThreads(0)
                    cv2.ocl.setUseOpenCL(False)
                    # in later check it for this code must need

                print(f"In this learning session, {'using multi process' if self.is_multi_gpu else 'using single process'}.")
                print("--------------------------------------------------")

                self._gpu_info: List[int] = []
                for _gpu_name, _gpu_id, _, _, _ in _gpu_info[:max_gpu_count]:
                    print(f"\tGPU device {_gpu_name}: {_gpu_id}")
                    self._gpu_info.append(_gpu_id)
                print("--------------------------------------------------")

            #  ----------------- #
            def _Process_initialize(self):
                ...

            def _Core_of_Learning(
                self,
                learning_info: Tuple[int, Learning.Mode],  # epoch, learning_mode
                gpu_info: int,
                data_info: Tuple[DataLoader, int],
                model: Model | DDP,
                optim: Optimizer,
                logger: SummaryWriter,
                save_dir: str,
                is_main_rank: bool
            ):
                raise NotImplementedError

            def _Process(self, num_of_p_d: int):
                """
                ### 각 프로세서 별 훈련 과정

                -------------------------------------------------------------------------------------------
                ## Parameters
                    num_of_p_d (int)
                        : 현재 장치에서 해당 훈련 과정에 할당된 프로세서 번호
                ## Returns
                    None

                -------------------------------------------------------------------------------------------
                """
                # set processer infomation
                _num_of_p_l = self.device_num + num_of_p_d  # process number in total learning
                _gpu_info: int = self._gpu_info[num_of_p_d]

                # set logger in learning
                _logger_dir = Path._Make_directory(f"process_{_num_of_p_l}", self.result_root)
                _logger = SummaryWriter(_logger_dir)

                # initialize model, optimizer
                _model = self._model
                _model = _model.cuda(_gpu_info)
                _model_name = _model._model_name

                _optim = self._optim
                _scheduler = self._scheduler

                # initialize dataset and data process
                _dataloaders: Dict[Learning.Mode, Tuple[DataLoader, DistributedSampler | None, int]] = {}

                if self.is_multi_gpu:  # Use DistributedDataParallel module for consist multi-process
                    distributed.init_process_group(backend="nccl", init_method=self.multi_protocal, world_size=self.world_size, rank=_num_of_p_l)
                    _model = DDP(_model)
                    for _mode, _opt in self.dataloader_param.items():
                        _this_sampler = DistributedSampler(_opt.dataset_process, rank=num_of_p_d)
                        _dataloaders[_mode] = (_opt._Get_dataloader(_this_sampler), _this_sampler, _opt.dataset_process.__len__())

                    return _model_name, _model, _optim, _scheduler, _dataloaders

                else:
                    for _mode, _opt in self.dataloader_param.items():
                        _dataloaders[_mode] = (_opt._Get_dataloader(None), None, _opt.dataset_process.__len__())

                System_Utils.Base._Print(f"Set Learning model {_model_name}, optimizer, Dataset", num_of_p_d)

                # Do learning
                for _epoch in range(self.last_epoch + 1, self.max_epoch):
                    _epoch_dir = System_Utils.Base._Make_dir(f"{_epoch}", self.result_root, _num_of_p_l)

                    for _active_mode in self.mode_list:
                        self._model.train() if _active_mode == Learning.Mode.TRAIN else _model.eval()

                        _dataloader, _this_sampler, _data_length = _dataloaders[_active_mode]

                        # When use sampler, shuffling
                        if _this_sampler is not None: _this_sampler.set_epoch(_epoch)

                        # Make save directory for each mode process
                        _mode_dir = System_Utils.Base._Make_dir(_active_mode.value, _epoch_dir, _process_num)

                        self._Core_of_Learning((_epoch, _active_mode), _gpu_info, (_dataloader, _data_length), _model, _optim, _logger, _mode_dir, _process_num is MAIN_RANK)

                    if not _process_num: self._Save(_epoch_dir, _model_name, _model, _optim, _scheduler)
                    if _scheduler is not None: _scheduler.step()

                    # save log file check
                    _logger.flush()

                _logger.close()

                print(f"Set Learning process for model {_model_name}is finish")

            def _Work(self):
                """
                ### 설정에 따른 훈련 진행

                -------------------------------------------------------------------------------------------
                ## Parameters
                    None

                ## Returns
                    None

                -------------------------------------------------------------------------------------------
                """
                if self.is_multi_gpu:
                    _gpu_count = len(self._gpu_info)
                    spawn(self._Process, nprocs=_gpu_count)
                else:
                    self._Process(num_of_p_d=0)

            #  ----------------- #
            def _Move_to_gpu(
                self,
                datas: Dict[str, Tensor],
                gpu_info: int
            ) -> Tuple[Tensor | List[Tensor], Tensor | List[Tensor] | None, int, Dict[str, Any]]:
                raise NotImplementedError

            def _Get_loss_n_observe_param(
                self,
                learning_info: Tuple[int, Learning.Mode],  # epoch, learning_mode
                output: Tensor | List[Tensor],
                label: Tensor | List[Tensor] | None,
                logger: SummaryWriter,
                save_dir: str | None = None,
                **data_info
            ) -> Tuple[Tensor, Dict[str, Tensor]]:
                """
                ### 모델의 출력 결과와 정답 이미지를 비교하여 훈련을 위한 loss 구성 및, 훈련 진행 정보를 생성.

                -------------------------------------------------------------------------------------------
                ## Parameters
                    output (Tensor | List[Tensor])
                        : model을 통해 생성한 출력 결과

                    label (Tensor | List[Tensor])
                        : 훈련 설정에 따라 구성된 Dataloader에서 생성된 라벨 데이터

                ## Returns
                    loss (Tensor)
                        : 모델 출력과 label 데이터를 기반으로 생성한 loss

                -------------------------------------------------------------------------------------------
                """
                raise NotImplementedError

            def _Progress_dispaly(
                self,
                learning_info: Tuple[int, Learning.Mode],
                progress_loss: float,
                progress_observe_param: Dict[str, float],
                spend_time: float,
                data_size: int,
                data_length: int,
            ):
                raise NotImplementedError

            def _Save(
                self,
                save_dir: str,
                file_name: str
            ):
                raise NotImplementedError

            def _Load(
                self,
                save_dir: str,
                file_name: str
            ):
                raise NotImplementedError

        class E2E(Basement):
            def _Core_of_Learning(
                self,
                learning_info: Tuple[int, Learning.Mode],  # epoch, learning_mode
                gpu_info: int,
                data_info: DataLoader,
                model: Model | DDP,
                optim: Optimizer,
                logger: SummaryWriter,
                save_dir: str,
                is_main_rank: bool
            ):
                # learning info
                _epoch, _mode = learning_info
                # _batch_pool_size = len(dataloader)
                dataloader, _max_data_length = data_info
                _display_term = int(self.display_term * _max_data_length) if isinstance(self.display_term, float) else self.display_term
                _this_count = 0

                # initialize parameter for observing
                _progress_loss = 0
                _progress_observe_param: Dict[str, float] = {}
                _display_milestone = 0
                _start_time = Debuging.Time._Stemp()

                for _datas in dataloader:
                    _input_datas, _label_data, _data_size, _data_info = self._Move_to_gpu(_datas, gpu_info)
                    _this_count += _data_size

                    if _mode == Learning.Mode.TRAIN:  # for Train
                        _output: Tensor | List[Tensor] = model(*_input_datas)
                    else:  # for validation
                        with no_grad(): _output: Tensor | List[Tensor] = model(*_input_datas)

                    _loss, _observe_param = self._Get_loss_n_observe_param(learning_info, _output, _label_data, logger, save_dir, **_data_info)

                    # doing learning
                    if _mode == Learning.Mode.TRAIN:  # for Train
                        optim.zero_grad()
                        _loss.backward()
                        optim.step()

                    # update learning process observation
                    _progress_loss += _loss.item() * _data_size
                    for _param, _value in _observe_param.items():
                        if _param in _progress_observe_param.keys():
                            _progress_observe_param[_param] += _value.item()
                        else:
                            _progress_observe_param.update({_param: _value.item()})

                    if _this_count >= _display_milestone and is_main_rank:
                        _display_milestone += _display_term
                        self._Progress_dispaly(learning_info, _progress_loss, _progress_observe_param, Debuging.Time._Stemp(_start_time), _this_count, _max_data_length)

                logger.add_scalar(f"Loss/{_mode.value}", _progress_loss / _this_count, _epoch)
                for _param, _value in _progress_observe_param.items(): logger.add_scalar(f"{_param}/{_mode.value}", _value / _this_count, _epoch)            

        class Reinforcement(Basement):
            def _Set_exploration_option(
                self,
                max_step: int,
                exploration_rate: float,
                exploration_discont: float,
                exploration_minimum: float,
                sequence_depth: int,
                reward_discount: float,
                memory_size: int,
                memory_minimum: int,
                reward_model: Callable
            ):
                """
                ### 강화학습 훈련에 사용되는 주요 인자 설정

                -------------------------------------------------------------------------------------------
                ## Argument
                - max_step : 강화학습에 사용되는 시나리오의 최대 크기
                - exploration_rate : actor의 행동 중 탐험 비율
                - exploration_discont : step 진행에 따른 actor의 행동 중 탐험비율 감소율
                - exploration_minimum : step 진행에 따라 감소되는 행동 중 탐험비율의 최소치
                - sequence_depth : 학습에 사용되는 시도 길이
                - reward_discount : step 진행에 따른 actor의 행동 중 탐험비율 감소율
                - memory_size : 훈련에서 사용되는 메모리 저장 최대 개수
                - memory_threshold : 훈련이 시작되기 위한 저장된 메모리의 최소 개수

                ## Returns
                    None

                -------------------------------------------------------------------------------------------
                """
                self._max_step = max_step

                # Parameter for exploration when make the action from actor output
                self._exploration_rate = exploration_rate
                self._exploration_discont = exploration_discont
                self._exploration_minimum = exploration_minimum

                # Parameter for replay
                self._sequence_depth = sequence_depth
                self._reward_discount = reward_discount
                self._reaplay_memory = deque(maxlen=memory_size) if memory_size != -1 else deque(maxlen=1)
                self._memory_minimum = memory_minimum

                self._reward_model = reward_model

            def _Core_of_Learning(
                self,
                learning_info: Tuple[int, Learning.Mode],  # epoch, learning_mode
                gpu_info: int,
                dataloader: DataLoader,
                model: Model | DDP,
                optim: Optimizer,
                logger: SummaryWriter,
                save_dir: str,
                is_main_rank: bool
            ):
                # learning info
                _epoch, _mode = learning_info
                # _batch_pool_size = len(dataloader)
                _max_data_length = self._dataset.__len__()
                _display_term = int(self.display_term * _max_data_length) if isinstance(self.display_term, float) else self.display_term
                _this_count = 0

                # initialize parameter for observing
                _progress_loss = 0
                _progress_observe_param = {}
                _display_milestone = 0
                _start_time = Debuging.Time._Stemp()

                # doing learning
                for _datas in dataloader:
                    _this_state, _label_data, _data_size, _data_info = self._Move_to_gpu(_datas, gpu_info)
                    _this_count += _data_size
                    _is_done = False
                    _next_state = None

                    # step
                    for _ in range(self._max_step):
                        if _mode == Learning.Mode.TRAIN:  # for Train
                            _output: Tensor | List[Tensor] = model(*_this_state)
                        else:  # for validation or test
                            with no_grad(): _output: Tensor | List[Tensor] = model(*_this_state)

                        _next_state, _is_done = self._Step(_this_state, _output)
                        with no_grad(): _next_output: Tensor | List[Tensor] = model(*_next_state)

                        _loss, _observe_param = self._Get_loss_n_observe_param(learning_info, [_output, _next_output], _label_data, _is_done, logger, save_dir, **_data_info)

                        if _mode == Learning.Mode.TRAIN:
                            optim.zero_grad()
                            _loss.backward()
                            # self._average_gradients(model)
                            optim.step()

                        # update learning process observation
                        _progress_loss += _loss.item() * _data_size
                        for _param, _value in _observe_param.items():
                            if _param in _progress_observe_param.keys():
                                _progress_observe_param[_param] += _value.item()
                            else:
                                _progress_observe_param.update({_param: _value.item()})

                        if _is_done:
                            break
                        else:
                            # state update
                            _this_state = _next_state

                    if _this_count >= _display_milestone and is_main_rank:
                        _display_milestone += _display_term
                        self._Progress_dispaly(learning_info, _progress_loss, _progress_observe_param, Debuging.Time._Stemp(_start_time), _this_count, _max_data_length)

                logger.add_scalar(f"Loss/{_mode.value}", _progress_loss, _epoch)
                for _param, _value in _progress_observe_param.items(): logger.add_scalar(f"{_param}/{_mode.value}", _value, _epoch)

            def _Step(self, state: Tensor | List[Tensor], output: Tensor | List[Tensor]):
                raise NotImplementedError

            def _Get_loss_n_observe_param(
                self,
                learning_info: Tuple[int, Learning.Mode],  # epoch, learning_mode
                output: List[Tensor | List[Tensor]],
                label: Tensor | List[Tensor] | None,
                is_done: bool,
                logger: SummaryWriter,
                save_dir: str | None = None,
                **data_info
            ) -> Tuple[Tensor, Dict[str, Tensor]]:
                raise NotImplementedError


class Learning_Config():
    @dataclass
    class E2E(Config):
        # learning information
        # --- project option --- #
        project_name: str = Directory._Relative_root(True)[:-len(Directory._Divider)]
        description: str = ""

        # --- e2e option --- #
        max_epoch: int = 100
        last_epoch: int = -1
        result_root: str = "./runs/"
        display_term: float | int = 0.1

        # --- multi process option --- #
        multi_method: str = "Auto"
        world_size: int = 2
        device_rank: int = 0
        max_gpu_count: int = 2
        multi_protocal: str = "tcp://127.0.0.1:10001"  # local

        # --- addtional config option --- #
        data_config_files : Dict[str, str | None] = field(
            default_factory = lambda: ({
                "train": "./config/dataset/dataloader.json",
                "validation": "./config/dataset/dataloader.json",
                "test": None
            }
        ))
        model_config_file: str = "./config/model/model.json"  # model config file information

        def _Param_for_laerning_process(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            _learning_basement_opt = {
                "project_name": self.project_name,
            }
            _learning_multi_opt= {
                "multi_method": Learning.Multi_Method(self.multi_method),
                "world_size": self.world_size,
                "device_rank": self.device_rank,
                "max_gpu_count": self.max_gpu_count,
                "multi_protocal": self.multi_protocal
            }

            return _learning_basement_opt, _learning_multi_opt

        def _Param_for_data_process(self) -> Dict[Learning.Mode, Learning.Dataloader_Config]:
            # example
            _dataloader_configs: Dict[Learning.Mode, Learning.Dataloader_Config] = {}

            for _mode_name, _config_file in self.data_config_files.items():
                if _config_file is not None:
                    _mode = Learning.Mode(_mode_name)
                    _dataset_config = Data_Config()
                    _dataset_config._Load(*File._Extrect_file_name(_config_file, False))

                    _dataset_name, _dataset_kwarg, _dataloader_kwarg = _dataset_config._Get_parameter()

                    if _dataset_name in Data.Process.__class__.__dict__.keys():
                        _dataloader_configs[_mode] = Learning.Dataloader_Config(
                            Data.Process.__class__.__dict__[_dataset_name](mode=_mode, **_dataset_kwarg),
                            collate_fn=None,
                            **_dataloader_kwarg
                        )

            return _dataloader_configs

        def _Get_model_n_optim_config(self) -> Tuple[Model, Optimizer, Scheduler.Basement | None]:
            # example
            _model_n_optim_config = Model_and_Optimizer_Config()
            _model_n_optim_config._Load(*File._Extrect_file_name(self.model_config_file, False))

            return _model_n_optim_config._Get_parameter(self.last_epoch)

        def _Initialize_project(
            self,
            learning_process: Type[Learning.Process.E2E]
        ):
            # Make learning process
            # --- make basement and set multi process parameter --- #
            _learning_basement_opt, _learning_multi_opt = self._Param_for_laerning_process()
            _learning_process = learning_process(**_learning_basement_opt)
            _learning_process._Set_GPU_option(**_learning_multi_opt)
 
            # --- set model process --- #
            _learning_process._Set_model_n_optim(*self._Get_model_n_optim_config())

            # --- set dataloader process --- #
            _learning_process._Set_dataloader_option(self._Param_for_data_process())

            return _learning_process

    @dataclass
    class Reinforcement(E2E):
        # --- exploration option --- #
        max_step: int = 1
        exploration_rate: float = 0.0
        exploration_discont: float = 0.0
        exploration_minimum: float = 0.0
        sequence_depth: int = 2
        reward_discount: float = 0.9
        memory_size: int = -1
        memory_minimum: int = -1

        # --- addtional config option --- #
        reward_config_file: str = "./config/reward.json"  # reward config file information

        def _Get_parameter(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
            raise NotImplementedError

        def _Get_Reward_model_config(self, config_process: Type[Config]):
            _file_dir, _file_name = File._Extrect_file_name(self.reward_config_file, False)

            _reward_model_config = config_process()
            _reward_model_config._Load(_file_name, _file_dir)
            return _reward_model_config
