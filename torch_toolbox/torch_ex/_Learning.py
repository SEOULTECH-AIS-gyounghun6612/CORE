from typing import Dict, List, Tuple, Type, Any, Callable
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

from python_ex._Base import Directory, File
from python_ex._Project import Debuging
from python_ex._Vision import cv2

from torch_ex._Base import Process_Name

from ._Base import Process_Name, System_Utils
from ._Dataset import Custom_Dataset_Process
from ._Model_n_Optim import Model, Optim, _LRScheduler


# -- DEFINE CONSTANT -- #
MAIN_RANK: int = 0


class Multi_Method(Enum):
    AUTO =  "Auto"
    NOT_USE = "None"
    DDP = "DistributedDataParallel"


# -- Main code -- #
class Learning_Process():
    class Basement():
        """
        ### 모델 학습을 위한 기본 모듈

        -------------------------------------------------------------------------------------------
        ## Argument & Parameters
        - project_name : 훈련이 진행되는 프로젝트 이름
        - description : 훈련의 목적 및 세부 설명
        - save_root : 훈련 과정 및 결과를 저장하기 위한 경로
        - mode_list : 훈련에서 진행하고자 하는 과정 -> Train, Validation, Test
        - max_epoch : 훈련에 적용하고자 하는 최대 epoch
        - last_epoch : 훈련의 초기 epoch
        -------------------------------------------------------------------------------------------
        """
        def __init__(self, project_name: str, description: str, result_root: str, mode_list: List[Process_Name], max_epoch: int, last_epoch: int = -1):
            self._project_name = project_name
            self._description = description
            self._result_root = result_root

            self._max_epoch = max_epoch
            self._last_epoch = last_epoch
            self._mode_list = mode_list

            print(f"Set the Learning for {self._project_name}.\n Result of this Learing, save at {self._result_root}")
            print("Set the basement of learning option.")
            _debug_process_text = f"This Learing work to at {self._max_epoch} epoch from {self._last_epoch + 1} epoch.\n"
            _debug_mode_list_text = f"This learning process, that consist of {', '.join(_mode.value for _mode in self._mode_list[: -1])} and {self._mode_list[-1].value}.\n"
            print(f"\t{_debug_process_text}\n\t{_debug_mode_list_text}")

        #  ----------------- #
        def _Set_dataloader_option(
            self,
            dataset_process: Custom_Dataset_Process,
            batch_size_per_node: int,
            num_worker_per_node: int,
            display_term: int | float
        ):
            """
            ### 훈련 데이터 설정 할당

            -------------------------------------------------------------------------------------------
            ## Parameters
                data_process (Custom_Dataset_Process)
                    : 훈련용 데이터 생성 모듈
                batch_size_per_node (int)
                    : 학습 데이터의 mini batch 크기
                num_worker_per_node (int)
                    : 학습 데이터 생성 프로세서 할당 수

            ## Returns
                None

            -------------------------------------------------------------------------------------------
            """
            # dataset
            self._dataset = dataset_process

            # dataloader
            self._batch_size = batch_size_per_node
            self._num_worker = num_worker_per_node

            # debugging
            _working_day = Debuging.Time._Apply_text_form(Debuging.Time._Stemp(), True, "%Y-%m-%d")
            _result_dir = Directory._Divider.join([self._project_name, dataset_process._organization.__class__.__name__, _working_day])
            self._result_root = Directory._Make(_result_dir, self._result_root)
            self._display_term = display_term

        def _Set_model_n_optim(
            self,
            model_structure: Type[Model],
            model_option: Dict[str, Any],
            optim_name: Optim.Supported,
            initial_lr: float,
            scheduler_name: Optim.Scheduler.Supported | None,
            scheduler_option: Dict[str, Any],
            weight_dir: str | None = None
        ):
            """
            ### 훈련을 위한 모델과 Optimizer 설정

            -------------------------------------------------------------------------------------------
            ## Parameters
                model_structure ()
                    : 훈련 대상 모델 class
                model_option ()
                    : 훈련 대상 모델 생성을 위한 입력 인자
                optim_name (Suport_Optimizer)
                    : 훈련에서 사용되는 Optimizer
                initial_lr (float)
                    : Optimizer 초기 학습 비율
                schedule_name (float)
                    : Optimizer의 학습 비율 변경에 사용되는 scheduler
                schedule_option (float)
                    : scheduler 생성을 위한 입력 인자

            ## Returns
                None

            -------------------------------------------------------------------------------------------
            """
            self._model_structure = model_structure
            self._model_option = model_option

            self._optim_name = optim_name
            self._initial_lr = initial_lr

            self._schedule_name = scheduler_name
            self._schedule_option = scheduler_option

            self._weight_dir= weight_dir

        def _Set_processer_option(
            self,
            multi_method: Multi_Method = Multi_Method.AUTO,
            world_size: int = 2,
            device_rank: int = 0,
            max_gpu_count: int = 2,
            multi_protocal: str | None = "tcp://127.0.0.1:10001"
        ):
            """
            ### 멀티 프로세서 관련 설정

            ### Set multi process setting

            -------------------------------------------------------------------------------------------
            ## Parameters
                multi_method (Multi_Method)
                    : 멀티 프로세서 활동 방법
                world_size (int)
                    : 훈련에 사용되는 전체 프로세서 갯수
                device_rank (int)
                    : 해당 코드를 실행 하고자 하는 단말장치의 식별번호 시작값
                usable_gpus (List[Tuple[int, str]])
                    : 학습에 사용가능한 GPU 리스트. 리스트에 각 데이터는 GPU 식별번호와 GPU 장치 이름으로 구성됨.
                    : List of usable GPUs. Each data consists of a GPU ID and a GPU name
                multi_protocal (str | None)
                    : 다른 프로세서와 통신 설정

            ## Returns
                None

            -------------------------------------------------------------------------------------------
            """
            print("Set the multiprocess option.")
            self._device_rank = device_rank

            _gpu_info = System_Utils.Cuda._Get_useable_gpu_list()
            # self._gpu_info: List[Tuple[int, str]] = []

            if len(_gpu_info) >= 2 and not multi_method is Multi_Method.NOT_USE:  # can use gpu
                print("In this learning session, using multi process.")
                print("--------------------------------------------------")
                if not multi_method.AUTO: _gpu_info = _gpu_info[:max_gpu_count]
                self._gpu_info = []

                for _gpu_id, _gpu_name in _gpu_info:
                    print(f"\tGPU device {_gpu_id}: {_gpu_name}")
                    self._gpu_info.append(_gpu_id)
                print("--------------------------------------------------")

                self._multi_method: Multi_Method = Multi_Method.DDP
                self._multi_protocal = multi_protocal
                self._world_size = world_size if world_size >= (device_rank + max_gpu_count) else (device_rank + max_gpu_count)

                # set cv2 setting
                cv2.setNumThreads(0)
                cv2.ocl.setUseOpenCL(False)

            else:                
                print("In this learning session, using single process.")
                print("--------------------------------------------------")

                assert len(_gpu_info), "THIS DEVICE CNA'T USE GPU. CHECK IT"

                for _gpu_id, _gpu_name in _gpu_info:
                    print(f"\tGPU device {_gpu_id}: {_gpu_name}")
                    self._gpu_info.append(_gpu_id)
                print("--------------------------------------------------")

                self._multi_method: Multi_Method = Multi_Method.NOT_USE
                self._multi_protocal = None
                self._world_size = 1

        #  ----------------- #
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
            if self._multi_method == Multi_Method.DDP:
                _gpu_count = len(self._gpu_info)
                spawn(self._Process, nprocs=_gpu_count)
            else:
                self._Process(process_num=0)

        def _Process(self, process_num: int):
            """
            ### 각 프로세서 별 훈련 과정

            -------------------------------------------------------------------------------------------
            ## Parameters
                processer_num (int)
                    : 현재 장치에서 해당 훈련 과정에 할당된 프로세서 번호
                share_block (multiprocessing.Queue | None)
                    : 멀티 프로세서 사용시 각 결과를 교환하기 위한 공유 블럭

            ## Returns
                None

            -------------------------------------------------------------------------------------------
            """
            # set processer infomation
            _process_num = self._device_rank + process_num
            _is_this_main = _process_num is MAIN_RANK
            _gpu_info: int = self._gpu_info[process_num]

            # set logger in learning
            _logger_dir = Directory._Make(f"_rank_{_process_num}_gpu_{_gpu_info}", self._result_root)
            _logger = SummaryWriter(_logger_dir)

            # initialize model, optimizer, dataset and data process
            _model, _model_name, _optim, _scheduler, _dataloader, _sampler = self._Process_init(_process_num, _gpu_info)

            # initialize parameter for best performing
            # _best_epoch = 0
            # _best_loss = float("inf")
            # _best_acc = float("-inf")

            # Do learning
            for _epoch in range(self._last_epoch + 1, self._max_epoch):
                _epoch_dir = System_Utils.Base._Make_dir(f"{_epoch}", self._result_root, _process_num)

                for _active_mode in self._mode_list:
                    _model.train() if _active_mode == Process_Name.TRAIN else _model.eval()
                    self._dataset._Set_active_mode_from(_active_mode)

                    # When use sampler, shuffling
                    if _sampler is not None: _sampler.set_epoch(_epoch)

                    # Make save directory for each mode process
                    _mode_dir = System_Utils.Base._Make_dir(_active_mode.value, _epoch_dir, _process_num)

                    self._Learning_core((_epoch, _active_mode), _gpu_info, _dataloader, _model, _optim, _logger, _mode_dir, _is_this_main)

                if _is_this_main: self._Save(_epoch_dir, _model_name, _model, _optim, _scheduler)
                if _scheduler is not None: _scheduler.step()

                # save log file check
                _logger.flush()

            _logger.close()

            print(f"Set Learning process for model {_model_name}is finish")
            # print(f"best epoch is {_best_epoch}; loss {_best_loss}, acc {_best_acc}")

        def _Process_init(self, process_num: int, gpu_info: int) -> tuple[DDP | Model, str, Optimizer, _LRScheduler | None, DataLoader, DistributedSampler | None]:
            # initialize model, optimizer, dataset and data process
            _model = self._model_structure(**self._model_option)
            _model = _model.cuda(gpu_info)
            _model_name = _model._model_name

            _optim, _scheduler = Optim._build(
                optim_name=self._optim_name,
                model=_model,
                initial_lr=self._initial_lr,
                schedule_name=self._schedule_name,
                last_epoch=self._last_epoch,
                **self._schedule_option)

            if (self._last_epoch + 1) and (self._weight_dir is not None):
                _model, _optim, _scheduler = self._Load(self._weight_dir, _model_name, _model, _optim, _scheduler)
            else:
                self._last_epoch = -1

            # initialize multi process or not
            if self._multi_method == Multi_Method.DDP:  # Use DistributedDataParallel module for consist multi-process
                distributed.init_process_group(backend="nccl", init_method=self._multi_protocal, world_size=self._world_size, rank=process_num)

                _model = DDP(_model)
                _sampler = DistributedSampler(self._dataset, rank=process_num)
                _dataloader = DataLoader(dataset=self._dataset, batch_size=self._batch_size, num_workers=self._num_worker, sampler=_sampler, drop_last=True)

            else:  # not consist multi-process
                _sampler = None
                _dataloader = DataLoader(dataset=self._dataset, batch_size=self._batch_size, num_workers=self._num_worker, shuffle=True, drop_last=True)

            print(f"Set Learning model {_model_name}, optimizer, Dataset")

            return _model, _model_name, _optim, _scheduler, _dataloader, _sampler

        def _Save(self, save_dir: str, file_name: str, model: Model | DDP, optim: Optimizer | None = None, schedule: _LRScheduler | None = None):
            save(model.state_dict(), f"{save_dir}{file_name}_model.h5")

            if optim is not None:
                _optim_and_schedule = {
                    "optimizer": optim.state_dict(),
                    "schedule": None if schedule is None else schedule.state_dict()}
                save(_optim_and_schedule, f"{save_dir}{file_name}_optim.h5")  # save optim and schedule state

        def _Load(self, save_dir: str, file_name: str, model: Model, optim: Optimizer, schedule: _LRScheduler | None = None):
            _save_dir = Directory._Divider_check(save_dir)
            if File._Exist_check(_save_dir, f"{file_name}_model.h5"):
                model.load_state_dict(load(f"{_save_dir}{file_name}_model.h5"))
            else:
                raise FileExistsError(f"model file _model.h5 is not exist in {save_dir}. Please check it")

            if File._Exist_check(_save_dir, f"{file_name}_optim.h5"):
                _optim_and_schedule = load(f"{_save_dir}{file_name}_optim.h5")
                optim.load_state_dict(_optim_and_schedule["optimizer"])
                if schedule is not None:
                    schedule.load_state_dict(_optim_and_schedule["schedule"])

            return model, optim, schedule

        #  ----------------- #
        def _Move_to_gpu(
            self,
            datas: Dict[str, Tensor],
            gpu_info: int
        ) -> Tuple[Tensor | List[Tensor], Tensor | List[Tensor] | None, int, Dict[str, Any]]:
            raise NotImplementedError

        def _Learning_core(
            self,
            learning_info: Tuple[int, Process_Name],  # epoch, learning_mode
            gpu_info: int,
            dataloader: DataLoader,
            model: Model | DDP,
            optim: Optimizer,
            logger: SummaryWriter,
            save_dir: str,
            is_main_rank: bool
        ):
            raise NotImplementedError

        def _Get_loss_n_observe_param(
            self,
            learning_info: Tuple[int, Process_Name],  # epoch, learning_mode
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
            learning_info: Tuple[int, Process_Name],
            progress_loss: float,
            progress_observe_param: Dict[str, float],
            spend_time: float,
            data_size: int,
            data_length: int,
        ):
            raise NotImplementedError

    class E2E(Basement):
        def _Learning_core(
            self,
            learning_info: Tuple[int, Process_Name],  # epoch, learning_mode
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
            _display_term = int(self._display_term * _max_data_length) if isinstance(self._display_term, float) else self._display_term
            _this_count = 0

            # initialize parameter for observing
            _progress_loss = 0
            _progress_observe_param: Dict[str, float] = {}
            _display_milestone = 0
            _start_time = Debuging.Time._Stemp()

            for _datas in dataloader:
                _input_datas, _label_data, _data_size, _data_info = self._Move_to_gpu(_datas, gpu_info)
                _this_count += _data_size

                if _mode == Process_Name.TRAIN:  # for Train
                    _output: Tensor | List[Tensor] = model(*_input_datas)
                else:  # for validation
                    with no_grad(): _output: Tensor | List[Tensor] = model(*_input_datas)
                
                _loss, _observe_param = self._Get_loss_n_observe_param(learning_info, _output, _label_data, logger, save_dir, **_data_info)

                # doing learning
                if _mode == Process_Name.TRAIN:  # for Train
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
        def _Set_reinforcement_option(
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

        def _Learning_core(
            self,
            learning_info: Tuple[int, Process_Name],  # epoch, learning_mode
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
            _display_term = int(self._display_term * _max_data_length) if isinstance(self._display_term, float) else self._display_term
            _this_count = 0

            # initialize parameter for observing
            _progress_loss = 0
            _progress_observe_param = {}
            _display_milestone = 0
            _start_time = Debuging.Time._Stemp()

            for _datas in dataloader:
                _this_state, _label_data, _data_size, _data_info = self._Move_to_gpu(_datas, gpu_info)
                _this_count += _data_size
                _is_done = False
                _next_state = None

                # doing learning
                for _ in range(self._max_step):
                    if _mode == Process_Name.TRAIN:  # for Train
                        _output: Tensor | List[Tensor] = model(*_this_state)
                    else:  # for validation or test
                        with no_grad(): _output: Tensor | List[Tensor] = model(*_this_state)

                    _next_state, _is_done = self._Step(_this_state, _output)
                    with no_grad(): _next_output: Tensor | List[Tensor] = model(*_next_state)

                    _loss, _observe_param = self._Get_loss_n_observe_param(learning_info, [_output, _next_output], _label_data, _is_done, logger, save_dir, **_data_info)

                    if _mode == Process_Name.TRAIN:
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
            learning_info: Tuple[int, Process_Name],  # epoch, learning_mode
            output: List[Tensor | List[Tensor]],
            label: Tensor | List[Tensor] | None,
            is_done: bool,
            logger: SummaryWriter,
            save_dir: str | None = None,
            **data_info
        ) -> Tuple[Tensor, Dict[str, Tensor]]:
            raise NotImplementedError


class Learning_Config():
    class Base():
        def __init__(self, file_name: str, file_dir: str = "./config/"):
            _, self._file_name = File._Extension_check(file_name, ["json"], True)
            self.file_dir = Directory._Divider_check(file_dir)

            self._options = self._Load()

        # Freeze function
        def _Set_default_option(self):
            return {
                # --- basement option --- #
                "project_name": Directory._Relative_root(True)[:-len(Directory._Divider)],
                "description": "",
                "result_root": "./runs/",
                "learning_plan": {
                    "train" : {
                        "amplification": 1,
                        "augmentations": {
                            "apply_mathod": "Albumentations",
                            "output_size": [256, 256],
                            # "rotate_limit": 0,
                            # "hflip_rate": 0.0,
                            # "vflip_rate": 0.0,
                            # "is_norm": True,
                            # "norm_mean": [0.485, 0.456, 0.406],
                            # "norm_std": [0.229, 0.224, 0.225],
                            # "apply_to_tensor": True,
                            # "group_parmaeter": None
                        }
                    },
                    "val" : {
                        "amplification": 1,
                        "augmentations": {
                            "apply_mathod": "Albumentations",
                            "output_size": [256, 256],
                            # "rotate_limit": 0,
                            # "hflip_rate": 0.0,
                            # "vflip_rate": 0.0,
                            # "is_norm": True,
                            # "norm_mean": [0.485, 0.456, 0.406],
                            # "norm_std": [0.229, 0.224, 0.225],
                            # "apply_to_tensor": True,
                            # "group_parmaeter": None
                        }
                    }
                },
                "max_epoch": 100,
                "last_epoch": -1,

                # --- dataloader option --- #
                # dataset
                "data_root": "./data/",
                # "data_name": "",

                # dataloader
                "batch_size_per_node": 512,
                "num_worker_per_node":8,
                "display_term": 0.01,

                # --- scheduler option --- #
                "optim_name": "Adam",
                "scheduler_name": "Cosin_Annealing",
                "term": 0.1,
                "term_amp": 1,
                "maximum": 0.0001,
                "minimum": 0.00005,
                "decay": 1,  # -> ?

                # --- multi process option --- #
                "multi_method": "Auto",
                "world_size": 2,
                "device_rank": 0,
                "max_gpu_count": 2,
                "multi_protocal": "tcp://127.0.0.1:10001"  # local
            }

        def _Save(self, config_file: Dict[str, Any]):
            File._Json(self.file_dir, self._file_name, True, config_file)

        def _Load(self):
            if File._Exist_check(self.file_dir, self._file_name):
                # load config
                return File._Json(self.file_dir, self._file_name)
            else:  # load template
                print(f"config file {self.file_dir}{self._file_name} not exsit. \nyou must config data initialize before use it")
                return self._Set_default_option()

        def _Make_learning_process(self, Learning_process: Type[Learning_Process.Basement]):
            # Make learning process
            _learning_opt = dict((_key, self._options[_key]) for _key in ["project_name", "description", "result_root", "learning_plan", "max_epoch", "last_epoch"])
            _learning_opt.update({"mode_list": [Process_Name(_mode) for _mode in self._options["learning_plan"].keys()]})
            _learning_process = Learning_process(**_learning_opt)

            # set multi process 
            _processer_opt = dict((_key, self._options[_key]) for _key in ["world_size", "device_rank", "max_gpu_count", "multi_protocal"])
            _processer_opt.update({"multi_method": Multi_Method(self._options["multi_method"])})
            _learning_process._Set_processer_option(**_processer_opt)
            _learning_process._Set_model_n_optim(**self._Make_model_n_optim_option())
            _learning_process._Set_dataloader_option(**self._Make_dataloader_option())

            return _learning_process

        def _Make_model_info(self) -> Tuple[Type[Model], Dict[str, Any]]:
            raise NotImplementedError

        def _Make_model_n_optim_option(self) -> Dict[str, Any]:
            _model_structure, _model_option = self._Make_model_info()
            _scheduler_option = dict((_key, self._options[_key]) for _key in ["term_amp", "maximum", "minimum", "decay"])

            _term = self._options["term"]
            _scheduler_option.update({"term": round(self._options["max_epoch"] * _term) if isinstance(_term, float) else _term})

            return {
                "model_structure": _model_structure,
                "model_option": _model_option,
                "optim_name": Optim.Supported(self._options["optim_name"]),
                "initial_lr": self._options["maximum"],
                "scheduler_name": Optim.Scheduler.Supported.Cosin_Annealing,
                "scheduler_option": _scheduler_option
            }

        def _Make_dataset(self) -> Custom_Dataset_Process:
            raise NotImplementedError

        def _Make_dataloader_option(self) -> Dict[str, Any]:
            _dataloader_option = {"dataset_process": self._Make_dataset()}
            _dataloader_option.update(dict((_key, self._options[_key]) for _key in ["batch_size_per_node", "num_worker_per_node", "display_term"]))
            return _dataloader_option