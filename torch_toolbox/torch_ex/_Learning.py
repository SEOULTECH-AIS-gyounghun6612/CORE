from typing import Dict, List, Tuple, Type, Any
from enum import Enum

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

from ._Base import Process_Name, System_Utils
from ._Dataset import Custom_Dataset_Process
from ._Model_n_Optim import Model, Optim, _LRScheduler


# -- DEFINE CONSTANT -- #
MAIN_RANK: int = 0


class Multi_Method(Enum):
    NOT_USE = "None"
    DP = "DataParallel"
    DDP = "DistributedDataParallel"


# -- Main code -- #
class End_to_End():
    """
    ### 모델 학습을 위한 End_to_End 학습 과정

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
    def __init__(self, project_name: str, description: str, save_root: str, mode_list: List[Process_Name], max_epoch: int, last_epoch: int = -1):
        self._project_name = project_name
        self._description = description
        self._save_root = self._Make_save_root(save_root)

        self._max_epoch = max_epoch
        self._last_epoch = last_epoch
        self._mode_list = mode_list
        self._active_mode = mode_list[0]

        print(f"Set the Learning for {self._project_name}.\n Result of this Learing, save at {self._save_root}")
        print("Set the basement of learning option.")
        _debug_process_text = f"This Learing work to at {self._max_epoch} epoch from {self._last_epoch + 1} epoch.\n"
        _debug_mode_list_text = f"This learning process, that consist of {', '.join(_mode.value for _mode in self._mode_list[: -1])} and {self._mode_list[-1].value}.\n"
        print(f"\t{_debug_process_text}\n\t{_debug_mode_list_text}")

    # --- initialize function --- #
    def _Make_save_root(self, save_root: str):
        _working_day = Debuging.Time._Apply_text_form(Debuging.Time._Stemp(), True, "%Y-%m-%d")
        return Directory._Make(save_root, Directory._Divider.join([self._project_name, _working_day]))

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
        self._dataset = dataset_process

        # - dataloader
        self._batch_size = batch_size_per_node
        self._num_worker = num_worker_per_node
        self._display_term = display_term

    def _Set_model_n_optim(
        self,
        model_structure: Type[Model],
        model_option: Dict[str, Any],
        optim_name: Optim.Supported,
        initial_lr: float,
        scheduler_name: Optim.Scheduler.Supported | None,
        scheduler_option: Dict[str, Any]
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

    def _Set_processer_option(
        self,
        multi_method: Multi_Method = Multi_Method.NOT_USE,
        world_size: int = 1,
        device_rank: int = 0,
        gpu_count: int = 1,
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
                : 해당 코드를 실행 하고자 하는 컴퓨터 식별번호
            usable_gpus (List[Tuple[int, str]])
                : 학습에 사용가능한 GPU 리스트. 리스트에 각 데이터는 GPU 식별번호와 GPU 장치 이름으로 구성됨.
                : List of usable GPUs. Each data consists of a GPU ID and a GPU name
            multi_protocal (str | None)
                : 멀티 프로세서 사이의 통신 설정

        ## Returns
            None

        -------------------------------------------------------------------------------------------
        """
        print("Set the multiprocess option.")
        self._device_rank = device_rank
        self._multi_protocal = multi_protocal
        self._gpu_info: List[Tuple[int, str]] = []

        if multi_method is Multi_Method.NOT_USE:  # NOT use multi process
            # In this flow, parameter _world_size, _multi_protocal in class is not use.
            self._multi_method: Multi_Method = multi_method
            print("In this learning session, using single process.")
            print("--------------------------------------------------")
            if gpu_count:  # using one gpu
                _gpu_info = System_Utils.Cuda._Get_useable_gpu_list()[0]
                print(f"\tGPU device {_gpu_info[0]}: {_gpu_info[1]}")
                self._gpu_info.append(_gpu_info)
            else:  # just use cpu
                print("\tCPU device")

            print("--------------------------------------------------")
        else:  # use multi process
            assert world_size >= device_rank + gpu_count,\
                f"paramerter {world_size} is wrong value; in this device process id num overflow the world size"  # In later, change to some function in python_ex._debug.py

            print("\tIn this learning session, using multi process.")
            # set parameter
            self._multi_method: Multi_Method = multi_method
            self._multi_protocal = multi_protocal
            self._world_size = world_size

            # set cv2 setting
            cv2.setNumThreads(0)
            cv2.ocl.setUseOpenCL(False)

            print("--------------------------------------------------")
            # if use gpu, set the gpu ids
            self._gpu_info = System_Utils.Cuda._Get_useable_gpu_list()
            for _info in self._gpu_info:  # if set multi gpu infomations,
                print(f"\tUsing GPU device {_info[0]}: {_info[1]}")
            print("--------------------------------------------------")

    # --- learning function --- #
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
            self._Process(processer_num=0)

    def _Process(self, processer_num: int):
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
        _process_num = self._device_rank + processer_num
        _is_this_main = _process_num is MAIN_RANK
        _gpu_info = self._gpu_info[processer_num] if len(self._gpu_info) else None

        # set logger in learning
        _logger = SummaryWriter(
            self._save_root,
            f"_rank_{_process_num}_cpu" if _gpu_info is None else f"_rank_{_process_num}_gpu_{_gpu_info[0]}_{_gpu_info[1]}")

        # initialize model, optimizer, dataset and data process
        _model, _model_name, _optim, _scheduler, _dataloader, _sampler = self._Process_init(_process_num, _gpu_info)

        # initialize parameter for best performing

        # _best_epoch = 0
        # _best_loss = float("inf")
        # _best_acc = float("-inf")

        # Do learning
        for _epoch in range(self._last_epoch + 1, self._max_epoch):
            _epoch_dir = Directory._Make(f"{_epoch}", self._save_root) if _is_this_main\
                else Directory._Divider_check(Directory._Divider.join([self._save_root, f"{_epoch}"]))

            for _active_mode in self._mode_list:
                self._Apply_active_mode(_active_mode, _model)

                # When use sampler, shuffling
                _sampler.set_epoch(_epoch) if _sampler is not None else ...

                # Make save directory for each mode process
                _mode_dir = Directory._Make(_active_mode.value, _epoch_dir) if _is_this_main\
                    else Directory._Divider_check(Directory._Divider.join([_epoch_dir, _active_mode.value]))

                self._Learning_core(_epoch, _gpu_info, _active_mode, _dataloader, _model, _optim, _logger, _mode_dir, _is_this_main)

            self._Save(_epoch_dir, _model_name, _model, _optim, _scheduler) if _is_this_main else ...
            _scheduler.step() if _scheduler is not None else ...

            # save log file check
            # _logger.flush()

        _logger.close()

        print(f"Set Learning process for model {_model_name}is finish")
        # print(f"best epoch is {_best_epoch}; loss {_best_loss}, acc {_best_acc}")

    def _Process_init(self, process_num: int, gpu_info: Tuple[int, str] | None):
        # initialize model, optimizer, dataset and data process
        _model = self._model_structure(**self._model_option)

        _model_name = _model._model_name

        _optim, _scheduler = Optim._build(
            optim_name=self._optim_name,
            model=_model,
            initial_lr=self._initial_lr,
            schedule_name=self._schedule_name,
            last_epoch=self._last_epoch,
            **self._schedule_option)

        if self._last_epoch + 1:
            _model, _optim, _scheduler = self._Load(Directory._Divider.join([self._save_root, f"{self._last_epoch}"]), _model_name, _model, _optim, _scheduler)

        # initialize multi process or not
        if self._multi_method == Multi_Method.DDP:  # Use DistributedDataParallel module for consist multi-process
            assert gpu_info is not None
            distributed.init_process_group(backend="nccl", init_method=self._multi_protocal, world_size=self._world_size, rank=process_num)
            _model = DDP(_model.cuda(gpu_info[0]))

            _sampler = DistributedSampler(self._dataset, rank=process_num)
            _dataloader = DataLoader(
                dataset=self._dataset,
                batch_size=self._batch_size,
                num_workers=self._num_worker,
                sampler=_sampler,
                drop_last=True)

        else:  # not consist multi-process
            _sampler = None
            _dataloader = DataLoader(
                dataset=self._dataset,
                batch_size=self._batch_size,
                num_workers=self._num_worker,
                shuffle=True,
                drop_last=True)

        print(f"Set Learning model {_model_name}, optimizer, Dataset")

        return _model, _model_name, _optim, _scheduler, _dataloader, _sampler

    def _Apply_active_mode(self, mode: Process_Name, model: Model | DDP):
        model.train() if mode == Process_Name.TRAIN else model.eval()
        self._dataset._Set_active_mode_from(mode)

    def _Move_to_gpu(self, datas, gpu_info: Tuple[int, str] | None) -> Tuple[List[Tensor], List[Tensor] | None, int, Dict[str, Any]]:
        raise NotImplementedError

    def _Learning_core(
        self,
        epoch: int,
        gpu_info: Tuple[int, str] | None,
        mode: Process_Name,
        dataloader: DataLoader,
        model: Model | DDP,
        optim: Optimizer,
        logger: SummaryWriter,
        save_dir: str,
        is_main_rank: bool
    ):
        # initialize parameter for observing
        _progress_loss = 0
        _progress_observe_param = 0
        _data_count = 0
        _display_milestone = 0
        _display_term = int(dataloader.__len__() * self._display_term) if isinstance(self._display_term, float) else self._display_term
        _start_time = Debuging.Time._Stemp()

        for _datas in dataloader:
            _input_datas, _label_data, _data_size, _data_info = self._Move_to_gpu(_datas, gpu_info)
            _data_count += _data_size

            # doing learning
            if mode == Process_Name.TRAIN:  # for Train
                _output: Tensor | List[Tensor] = model(*_input_datas)
                _loss, _observe_param = self._Get_loss_n_observe_param(epoch, mode, _output, _label_data, logger)

                optim.zero_grad()
                _loss.backward()
                # self._average_gradients(model)
                optim.step()

            else:  # for validation
                with no_grad():
                    _output: Tensor | List[Tensor] = model(*_input_datas)
                    _loss, _observe_param = self._Get_loss_n_observe_param(epoch, mode, _output, _label_data, logger, save_dir, **_data_info)

            # update learning process observation
            _progress_loss += _loss.item()
            _progress_observe_param += _observe_param.item()

            if _data_count >= _display_milestone:
                _display_milestone += _display_term
                self._Progress_dispaly(epoch, mode, _progress_loss, _progress_observe_param, Debuging.Time._Stemp(_start_time), _data_count, dataloader.__len__()) if is_main_rank else ...

    def _Get_loss_n_observe_param(
        self,
        epoch: int,
        mode: Process_Name,
        output: Tensor | List[Tensor],
        label: Tensor | List[Tensor] | None,
        logger: SummaryWriter,
        save_dir: str | None = None,
        **data_info
    ) -> Tuple[Tensor, Tensor]:
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

    # def _Average_gradients(self, model: Custom_Model):
    #     size = float(distributed.get_world_size())
    #     for param in model.parameters():
    #         if param.grad is not None:
    #             distributed.all_reduce(param.grad.data, op=distributed.ReduceOp.SUM)
    #             param.grad.data /= size

    def _Progress_dispaly(self, epoch: int, mode: Process_Name, progress_loss: float, progress_observe_param: float, spend_time: float, data_size: int, data_length: int):
        raise NotImplementedError

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


# class Reinforcement():
#     class Basement(End_to_End):
#         # --- initialize function --- #
#         def _Set_reinforcement_option(
#             self,
#             max_step: int,
#             exploration_rate: float,
#             exploration_discont: float,
#             exploration_minimum: float,
#             reward_value_list: List[float],
#             learning_range: int,
#             reward_discount: float,
#             memory_size: int,
#             memory_threshold: int
#         ):
#             """
#             ### 강화학습 훈련에 사용되는 주요 인자 설정

#             -------------------------------------------------------------------------------------------
#             ## Parameters
#                 max_step (int)
#                     : 강화학습에 사용되는 시나리오의 최대 크기
#                 exploration_rate (float)
#                     : actor의 행동 중 탐험 비율
#                 exploration_discont (float)
#                     : step 진행에 따른 actor의 행동 중 탐험비율 감소율
#                 exploration_minimum (float)
#                     : step 진행에 따라 감소되는 행동 중 탐험비율의 최소치
#                 reward_value_list (List[float])
#                     : 적용하고자 하는 reward 구성
#                 total_memory (int)
#                     : 훈련에서 사용되는 메모리 저장 최대 개수
#                 memory_threshold (int)
#                     : 훈련이 시작되기 위한 저장된 메모리의 최소 개수
#                 learning_range (int)
#                     : 학습에 사용되는 각각의 메모리 크기
#                 reward_discount (float)
#                     : step 진행에 따른 actor의 행동 중 탐험비율 감소율

#             ## Returns
#                 None

#             -------------------------------------------------------------------------------------------
#             """
#             self._max_step = max_step

#             # Parameter for exploration when make the action from actor output
#             self._exploration_rate = exploration_rate
#             self._exploration_discont = exploration_discont
#             self._exploration_minimum = exploration_minimum

#             # Parameter for make to reward
#             self._reward_value_list = reward_value_list

#             # Parameter for replay
#             self._learning_range = learning_range
#             self._reward_discount = reward_discount
#             self._reaplay_memory = deque(maxlen=memory_size)
#             self._memory_threshold = memory_threshold

#         def _Set_reward_option(
#             self,
#             reward_milestone: List[float],
#             save_dir: str | None = None,
#             reward_model_structure: Type[Custom_Model] | None = None,
#             **model_parameters
#         ):
#             """
#             ### 강화학습 훈련에 보상 모델 관련 설정

#             -------------------------------------------------------------------------------------------
#             ## Parameters
#                 reward_milestone (List[float])
#                     : 결과에 따른 reward 구성 별 할당 범위, reward = reward_value_list[ct], (reward_milestone[ct] <= result < reward_milestone[ct+1] 일때)
#                 save_dir (str)
#                     : 보상 모델이 저장된 경로
#                 reward_model_structure (Type[Custom_Model])
#                     : 보상 모델 구조

#             ## Returns
#                 None

#             -------------------------------------------------------------------------------------------
#             """
#             # Parameter for make to reward
#             self._reward_milestone = reward_milestone

#             if save_dir is not None and reward_model_structure is not None:
#                 # Parameter for reward model
#                 self._reward_model = reward_model_structure(**model_parameters)
#                 _save_dir = Directory._Divider_check(save_dir)

#                 if File._Exist_check(_save_dir, f"{self._reward_model._model_name}_model.h5"):
#                     self._reward_model.load_state_dict(load(f"{_save_dir}{self._reward_model._model_name}_model.h5"))
#                 else:
#                     raise FileExistsError(f"reward model file {self._reward_model._model_name}_model.h5 is not exist in {save_dir}. Please check it")
#                 self._reward_model.eval()

#         def _Process(self, processer_num: int):
#             """
#             ### 각 프로세서 별 훈련 과정

#             -------------------------------------------------------------------------------------------
#             ## Parameters
#                 processer_num (int)
#                     : 현재 장치에서 해당 훈련 과정에 할당된 프로세서 번호
#                 share_block (multiprocessing.Queue | None)
#                     : 멀티 프로세서 사용시 각 결과를 교환하기 위한 공유 블럭

#             ## Returns
#                 None

#             -------------------------------------------------------------------------------------------
#             """
#             _process_num = self._this_rank + processer_num
#             _gpu_id = self._gpu_list[processer_num] if len(self._gpu_list) else -1

#             _agent_model, _model_name, _optim, _scheduler, _stateloader, _sampler = self._Process_initialize(_process_num, _gpu_id)

#             # Do learning process
#             for _epoch in range(self._last_epoch + 1, self._max_epoch):
#                 _epoch_dir = System_Utils._Make_directory(f"{_epoch}", _process_num, self._save_root)

#                 for _active_mode in self._mode_list:
#                     # get stateloader
#                     self._Apply_active_mode(_active_mode, [_agent_model, ])

#                     # - When use sampler, shuffling
#                     _sampler.set_epoch(_epoch) if _sampler is not None else ...
#                     _mode_dir = System_Utils._Make_directory(f"{_active_mode.value}", _process_num, _epoch_dir)

#                     _state_count = 0
#                     _display_milestone = self._display_term
#                     _start_time = Debuging.Time._Stemp()

#                     for _datas in _stateloader:
#                         _this_state, _goal_state, _etc_data, _load_data_size = self._Data_preprocess(_datas)
#                         _state_count += _load_data_size

#                         while True:
#                             if _active_mode == Process_Name.TRAIN:  # for Train
#                                 _raw_action: Tensor | List[Tensor] = _agent_model(*_this_state)
#                                 _action = self._Make_action(_raw_action)

#                                 _next_state, _reward, _value, is_endtime = self._Play(_action, _goal_state)
#                                 _backward_loss = self._Replay(_this_state, _action, _next_state, _reward, _value)

#                                 _optim.zero_grad()
#                                 _backward_loss.backward()
#                                 # self._average_gradients(model)
#                                 _optim.step()

#                             else:  # for validation, test
#                                 with no_grad():
#                                     _raw_action: Tensor | List[Tensor] = _agent_model(*_this_state)
#                                     _action = self._Make_action(_raw_action)
#                                     _next_state, _reward, _value, is_endtime = self._Play(_action, _goal_state)

#                             self._Logging(
#                                 _epoch,
#                                 _active_mode,
#                                 Debuging.Time._Stemp(_start_time),
#                                 {"action": _action, "reward": _reward, "value": _value, },
#                                 _etc_data, _mode_dir)

#                             if is_endtime:
#                                 break

#                             _this_state = _next_state

#                         if _state_count >= _display_milestone:
#                             _display_milestone += self._display_term
#                             self._Progress_dispaly(_epoch, _active_mode, Debuging.Time._Stemp(_start_time))
#                         _start_time = Debuging.Time._Stemp()

#                 _scheduler.step() if _scheduler is not None else ...

#                 # save log file
#                 if _process_num is MAIN_RANK:
#                     self._logger._Insert({"Last_epoch": _epoch}, self._logger._Annotation)
#                     self._logger._Insert({"Learning_rate": _scheduler.get_lr()}, self._logger._Annotation, False) if _scheduler is not None else ...
#                     self._logger._Save(self._save_root, "Learning_Log.json")

#                     # save model
#                     self._Save(_epoch_dir, _model_name, _agent_model, _optim, _scheduler)

#         def _Make_action(self, raw_data: Tensor | List[Tensor]) -> TYPE_NUMBER | List[TYPE_NUMBER]:
#             raise NotImplementedError

#         def _Play(
#             self,
#             action: TYPE_NUMBER | List[TYPE_NUMBER],
#             goal_state: Tensor | List[Tensor] | None
#         ) -> Tuple[List[Any], TYPE_NUMBER | List[TYPE_NUMBER], TYPE_NUMBER | List[TYPE_NUMBER], bool]:
#             raise NotImplementedError

#         def _Replay(self, this_state: List[Any], action, next_state, reward, value) -> Tensor:
#             raise NotImplementedError

#     class Actor_Critic(Basement):
#         def _Set_value_model_n_optim(
#             self,
#             model_structure: Type[Custom_Model],
#             model_option: Dict[str, Any],
#             optim_name: Suport_Optimizer,
#             initial_lr: float,
#             scheduler_name: Suport_Schedule | None,
#             scheduler_option: Dict[str, Any]
#         ):
#             """
#             ### 훈련에 사용되는 값 모델과 Optimizer 설정

#             -------------------------------------------------------------------------------------------
#             ## Parameters
#                 model_structure ()
#                     : 훈련 대상 모델 class
#                 model_option ()
#                     : 훈련 대상 모델 생성을 위한 입력 인자
#                 optim_name (Suport_Optimizer)
#                     : 훈련에서 사용되는 Optimizer
#                 initial_lr (float)
#                     : Optimizer 초기 학습 비율
#                 schedule_name (float)
#                     : Optimizer의 학습 비율 변경에 사용되는 scheduler
#                 schedule_option (float)
#                     : scheduler 생성을 위한 입력 인자

#             ## Returns
#                 None

#             -------------------------------------------------------------------------------------------
#             """
#             self._v_model_structure = model_structure
#             self._v_model_option = model_option

#             self._optim_for_value = optim_name
#             self._initial_lr_for_value = initial_lr

#             self._schedule_for_v_name = scheduler_name
#             self._schedule_for_v_option = scheduler_option

#         def _Process(self, processer_num: int):
#             """
#             ### 각 프로세서 별 훈련 과정

#             -------------------------------------------------------------------------------------------
#             ## Parameters
#                 processer_num (int)
#                     : 현재 장치에서 해당 훈련 과정에 할당된 프로세서 번호

#             ## Returns
#                 None

#             -------------------------------------------------------------------------------------------
#             """
#             _process_num = self._this_rank + processer_num
#             _gpu_id = self._gpu_list[processer_num] if len(self._gpu_list) else -1

#             [_agent, _value_model], [_agent_name, _value_model_name], [_agnet_optim, _value_optim], [_agent_scheduler, _value_scheduler], _dataloader, _sampler\
#                 = self._Process_initialize(_process_num, _gpu_id)

#             # Do learning process
#             for _epoch in range(self._last_epoch + 1, self._max_epoch):
#                 _epoch_dir = System_Utils._Make_directory(f"{_epoch}", _process_num, self._save_root)

#                 for _active_mode in self._mode_list:
#                     # get stateloader
#                     self._Apply_active_mode(_active_mode, [_agent, _value_model])

#                     # - When use sampler, shuffling
#                     _sampler.set_epoch(_epoch) if _sampler is not None else ...
#                     _mode_dir = System_Utils._Make_directory(f"{_active_mode.value}", _process_num, _epoch_dir)

#                     _state_count = 0
#                     _display_milestone = self._display_term
#                     _start_time = Debuging.Time._Stemp()

#                     for _datas in _dataloader:
#                         _this_state, _goal_state, _etc_data, _load_data_size = self._Data_preprocess(_datas)
#                         _state_count += _load_data_size

#                         while True:
#                             if _active_mode == Process_Name.TRAIN:  # for Train
#                                 _raw_action: Tensor | List[Tensor] = _agent(*_this_state)
#                                 _action = self._Make_action(_raw_action)

#                                 _next_state, _reward, _value, is_endtime = self._Play(_action, _goal_state)
#                                 [_agent_loss, _value_loss] = self._Replay(_this_state, _action, _next_state, _reward, _value)

#                                 _agnet_optim.zero_grad()
#                                 _agent_loss.backward()
#                                 # self._average_gradients(_agent)
#                                 _agnet_optim.step()

#                                 _value_optim.zero_grad()
#                                 _value_loss.backward()
#                                 # self._average_gradients(value_model)
#                                 _value_optim.step()

#                             else:  # for validation, test
#                                 with no_grad():
#                                     _raw_action: Tensor | List[Tensor] = _agent(*_this_state)
#                                     _action = self._Make_action(_raw_action)
#                                     _next_state, _reward, _value, is_endtime = self._Play(_action, _goal_state)

#                             self._Logging(
#                                 _epoch,
#                                 _active_mode,
#                                 Debuging.Time._Stemp(_start_time),
#                                 {"action": _action, "reward": _reward, "value": _value, },
#                                 _etc_data, _mode_dir)

#                             if is_endtime:
#                                 break

#                             _this_state = _next_state

#                         if _state_count >= _display_milestone:
#                             _display_milestone += self._display_term
#                             self._Progress_dispaly(_epoch, _active_mode, Debuging.Time._Stemp(_start_time))
#                         _start_time = Debuging.Time._Stemp()

#                 _agent_scheduler.step() if _agent_scheduler is not None else ...
#                 _value_scheduler.step() if _value_scheduler is not None else ...

#                 # save log file
#                 if _process_num is MAIN_RANK:
#                     self._logger._Insert({"Last_epoch": _epoch}, self._logger._Annotation)
#                     self._logger._Insert({
#                         "Learning_rate": {
#                             "model": _agent_scheduler.get_lr() if _agent_scheduler is not None else self._initial_lr,
#                             "value": _value_scheduler.get_lr() if _value_scheduler is not None else self._initial_lr_for_value
#                         }
#                     }, self._logger._Annotation, False)
#                     self._logger._Save(self._save_root, "Learning_Log.json")

#                     # save model
#                     self._Save(_epoch_dir, _agent_name, _agent, _agnet_optim, _agent_scheduler)
#                     self._Save(_epoch_dir, _value_model_name, _value_model, _value_optim, _value_scheduler)

#         def _Process_initialize(
#             self,
#             process_id: int,
#             gpu_id: int
#         ) -> tuple[List[Custom_Model | DDP], List[str], List[Optimizer], List[_LRScheduler | None], DataLoader, DistributedSampler | None]:
#             """
#             ### 각 프로세서에 할당된 별 작업 초기화 과정

#             -------------------------------------------------------------------------------------------
#             ## Parameters
#                 process_id (int)
#                     : 해당 프로세서가 진행되는 프로세서의 번호

#                 gpu_id (int)
#                     : 해당 프로세서에서 사용되는 GPU 장치의 번호. -1 의 경우 해당 프로세서에서는 GPU를 사용하지 않음.

#             ## Returns
#                 model ( Custom_Model | DDP )
#                     : 모델 구조체
#                 model_name ( str )
#                     : 모델 구조체의 별칭
#                 optimizer ( Optimizer )
#                     : 모델 인자값 갱신을 위한 optimizer
#                 scheduler ( _LRScheduler | None)
#                     : optimizer에 적용된 scheduler
#                 dataloader (dict[Process_Name, DataLoader])
#                     : 모델의 입력으로 사용하기 위한 dataloader
#                 sampler (dict[Process_Name, DistributedSampler | None])
#                     : 분산 구조 적용시, 데이터 샘플링을 위한 구조.

#             -------------------------------------------------------------------------------------------
#             """
#             # set model
#             _agent = self._model_structure(**self._model_option)
#             _agent_name = _agent._model_name

#             _optim, _scheduler = _Optimizer_build(
#                 optim_name=self._optim_name,
#                 model=_agent,
#                 initial_lr=self._initial_lr,
#                 schedule_name=self._schedule_name,
#                 last_epoch=-1 if self._last_epoch else self._last_epoch,
#                 **self._schedule_option)
#             print(f"Set Learning model {_agent_name} and optimizer")

#             _value_model = self._v_model_structure(**self._v_model_option)
#             _value_model_name = _value_model._model_name

#             _optim_for_v, _scheduler_for_v = _Optimizer_build(
#                 optim_name=self._optim_for_value,
#                 model=_value_model,
#                 initial_lr=self._initial_lr_for_value,
#                 schedule_name=self._schedule_for_v_name,
#                 last_epoch=-1 if self._last_epoch else self._last_epoch,
#                 **self._schedule_for_v_option)
#             print(f"Set value model {_value_model_name} and optimizer")

#             # set process, dataset, sampler
#             if self._multi_method == Multi_Method.DDP:  # Use DistributedDataParallel module for consist multi-process
#                 distributed.init_process_group(backend="nccl", init_method=self._multi_protocal, world_size=self._world_size, rank=process_id)
#                 _group_agent = distributed.new_group()
#                 _agent = DDP(_agent, device_ids=[process_id if gpu_id == -1 else gpu_id], process_group=_group_agent, broadcast_buffers=True)
#                 _group_value = distributed.new_group()
#                 _value_model = DDP(_value_model, device_ids=[process_id if gpu_id == -1 else gpu_id], process_group=_group_value, broadcast_buffers=True)

#                 _sampler = DistributedSampler(self._dataset, rank=process_id)
#                 _dataloader = DataLoader(
#                     dataset=self._dataset,
#                     batch_size=self._batch_size,
#                     num_workers=self._num_worker,
#                     sampler=_sampler,
#                     drop_last=True)

#             else:  # not consist multi-process
#                 _sampler = None
#                 _dataloader = DataLoader(
#                     dataset=self._dataset,
#                     batch_size=self._batch_size,
#                     num_workers=self._num_worker,
#                     shuffle=True)

#             print("Set Dataset")

#             return [_agent, _value_model], [_agent_name, _value_model_name], [_optim, _optim_for_v], [_scheduler, _scheduler_for_v], _dataloader, _sampler

#         def _Replay(self, this_state: List[Any], action, next_state, reward, value) -> Tuple[Tensor, Tensor]:
#             raise NotImplementedError
