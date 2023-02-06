from typing import Dict, List, Tuple, Union, Type, Optional
from enum import Enum

from torch import distributed, cuda, save, load, multiprocessing
from torch.multiprocessing.spawn import spawn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.autograd.grad_mode import no_grad
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from python_ex._base import Directory, File, Utils
from python_ex._vision import cv2

if __package__ == "":
    # if this file in local project
    from torch_ex._torch_base import Learning_Mode, Tracking, Parameter_Type
    from torch_ex._label import Label
    from torch_ex._dataset import Custom_Dataset, Augment
    from torch_ex._layer import Custom_Model
    from torch_ex._optimizer import _LRScheduler, Suport_Optimizer, Suport_Schedule, _Optimizer_build
else:
    # if this file in package folder
    from ._torch_base import Learning_Mode, Tracking, Parameter_Type
    from ._label import Label
    from ._dataset import Custom_Dataset, Augment
    from ._layer import Custom_Model
    from ._optimizer import _LRScheduler, Suport_Optimizer, Suport_Schedule, _Optimizer_build


# -- DEFINE CONSTNAT -- #
# type hint
MODEL = Union[Custom_Model, DDP]
MAIN_RANK: int = 0


class Multi_Method(Enum):
    NONE = None
    DP = "DataParallel"
    DDP = "DistributedDataParallel"


# -- Mation Function -- #
class Learning_Process():
    class End_to_End():
        def __init__(
            self,
            max_epoch: int,
            learning_mode: List[Learning_Mode],
            batch_size_per_node: int,
            num_worker_per_node: int,
            last_epoch: int = -1,
            project_name: Optional[str] = None,
            description: Optional[str] = None,
            save_root: str = "./",
            world_size: int = 1,
            this_rank: int = 0,
            gpu_ids: List[int] = list(range(cuda.device_count())),
            multi_method: Multi_Method = Multi_Method.NONE,
            multi_protocal: Optional[str] = "tcp://127.0.0.1:10001",
        ):
            # Information about this learning
            self._project_name = project_name
            self._description = description

            # Setting about file I/O
            self._save_root = Directory._make(Directory._divider_check(f"{project_name}/{description}/"), save_root)

            # Setting about learning
            # - base
            self._max_epoch = max_epoch
            self._last_epoch = last_epoch
            self._learning_mode = learning_mode

            # - dataloader
            self._batch_size = batch_size_per_node
            self._num_worker = num_worker_per_node

            # - gpu & multiprocess (default -> Not use)
            if len(gpu_ids) > 1 and distributed.is_nccl_available():
                # - use mult gpu and multi process
                self._gpu_list = gpu_ids
                cv2.setNumThreads(0)
                cv2.ocl.setUseOpenCL(False)
                self._multi_method = Multi_Method.DDP
                self._world_size = world_size * len(self._gpu_list) if world_size == 1 else world_size
            elif len(gpu_ids) == 1:
                # - use only one gpu
                self._gpu_list = gpu_ids
                self._multi_method = Multi_Method.NONE
                self._world_size = 1
            else:
                # - not use gpu but can use multi process
                self._gpu_list = []
                self._multi_method = Multi_Method.NONE if multi_method is Multi_Method.DDP else multi_method
                self._world_size = 1 if self._multi_method is Multi_Method.NONE else world_size

            self._this_rank = this_rank
            self._multi_protocal = multi_protocal

        # Freeze function
        # --- for init function --- #
        def _Set_dataset(
                self,
                dataset_class: Type[Custom_Dataset],
                label_process: Label.Process.Basement,
                data_process: Label.File_IO.Basement,
                amplification: Dict[Learning_Mode, int],
                augmentation: Dict[Learning_Mode, Union[Augment.Basement, List[Augment.Basement]]]
        ):
            self._dataset: Dict[Learning_Mode, Custom_Dataset] = {}
            for _mode, _amp in amplification.items():
                self._dataset.update({
                    _mode: dataset_class(label_process, data_process._Get_file_profiles(_mode), _amp, augmentation[_mode])
                })

            # in later make the code, that data info update to tracker's annotation
            self._save_root = Directory._make(Directory._divider_check(f"{label_process._lable_name.value}/"), self._save_root)

        def _Set_model_parameter(self, model_structure: Type[Custom_Model], **model_parameter):
            self._model_structure = model_structure
            self._model_parameters = model_parameter

        def _Set_optim_schedule_prameter(self, optim_name: Suport_Optimizer, initial_lr: float, schedule_name: Suport_Schedule, **schedule_parmeter):
            self._optim_name = optim_name
            self._initial_lr = initial_lr

            self._schedule_name = schedule_name
            self._schedule_option = schedule_parmeter

        def _Set_tracker(
            self,
            tracking_param: Dict[Learning_Mode, Dict[Parameter_Type, List[str]]],
            observing_param: Dict[Learning_Mode, Dict[Parameter_Type, Optional[List[str]]]]
        ):
            self._tracker = Tracking.To_Process(tracking_param, observing_param)
            # insert learning info
            # self._tracker._insert()

        # --- for work function --- #
        def _Set_learning_model(self, gpu_id: int) -> Tuple[MODEL, Optimizer, Optional[_LRScheduler]]:
            _model = self._model_structure(**self._model_parameters)
            _model = _model.cuda(gpu_id) if gpu_id != -1 else _model
            _optim, _scheduler = _Optimizer_build(self._optim_name, _model, self._initial_lr, self._schedule_name, last_epoch=self._last_epoch, **self._schedule_option)

            return _model, _optim, _scheduler

        def _Active_mode_change_to(
                self,
                mode: Learning_Mode,
                model: MODEL,
                dataloader: Dict[Learning_Mode, DataLoader],
                sampler: Dict[Learning_Mode, Optional[DistributedSampler]]):

            self._tracker._Set_activate_mode(mode)
            model.train() if mode == Learning_Mode.TRAIN else model.eval()
            _this_dataloader = dataloader[mode]
            _this_sampler = sampler[mode]

            return _this_dataloader, _this_sampler

        # in later this function remove
        def _Save_weight(self, save_dir: str, model: MODEL, optim: Optional[Optimizer] = None, schedule: Optional[_LRScheduler] = None):
            save(model.state_dict(), f"{save_dir}_model.h5")

            if optim is not None:
                _optim_and_schedule = {
                    "optimizer": optim.state_dict(),
                    "schedule": None if schedule is None else schedule}
                save(_optim_and_schedule, f"{save_dir}optim.h5")  # save optim and schedule state

        # in later this function remove
        def _Load_weight(self, save_dir: str, model: MODEL, optim: Optional[Optimizer] = None, schedule: Optional[_LRScheduler] = None):
            _model_file = f"{save_dir}_model..h5"

            if File._exist_check(_model_file):
                model.load_state_dict(load(_model_file))
            else:
                raise FileExistsError(f"model file _model..h5 is not exist in {save_dir}. Please check it")

            _optim_file = f"{save_dir}optim.h5"
            if File._exist_check(_model_file) and optim is not None:
                _optim_and_schedule = load(_optim_file)
                optim.load_state_dict(_optim_and_schedule["optimizer"])
                if schedule is not None:
                    schedule.load_state_dict(_optim_and_schedule["schedule"])

            return model, optim, schedule

        def _Progress_dispaly(
                self,
                mode: Learning_Mode,
                epoch: int,
                decimals: int = 1,
                length: int = 25,
                fill: str = '█'):
            _epoch_board = Utils._progress_board(epoch, self._max_epoch)

            _max_data_len = self._dataset[mode].__len__()
            _data_count = self._tracker._Get_observing_length(epoch)
            _data_board = Utils._progress_board(_data_count, _max_data_len)

            _batch_size = self._batch_size

            _allocated_len = round(_max_data_len / self._world_size)
            _max_batch_ct = _allocated_len // _batch_size + int((_allocated_len % _batch_size) > 0)

            _this_time = self._tracker._Get_progress_time(epoch)
            _this_time_str = Utils.Time._apply_text_form(sum(_this_time), text_format="%H:%M:%S")
            _max_time_str = Utils.Time._apply_text_form(_max_batch_ct * sum(_this_time) / len(_this_time), text_format="%H:%M:%S")

            _pre = f"{mode.value} {_epoch_board} {_data_board} {_this_time_str}/{_max_time_str} "
            _suf = self._tracker._Learning_observing(epoch)

            Utils._progress_bar(_data_count, _max_data_len, _pre, _suf, decimals, length, fill)

        def _Process(self, processer_num: int, share_block: Optional[multiprocessing.Queue] = None):
            _this_node = self._this_rank + processer_num
            _this_gpu_id = self._gpu_list[processer_num] if len(self._gpu_list) else -1

            _dataloader: Dict[Learning_Mode, DataLoader] = {}
            _sampler: Dict[Learning_Mode, Optional[DistributedSampler]] = {}

            # Set process init
            # - Initialize distributed
            if self._multi_method == Multi_Method.DDP:
                distributed.init_process_group(backend="nccl", init_method=self._multi_protocal, world_size=self._world_size, rank=_this_node)
                # Set dataloader and sampler
                for _this_mode in self._learning_mode:
                    _sampler[_this_mode] = DistributedSampler(self._dataset[_this_mode], rank=_this_node, shuffle=(_this_mode == Learning_Mode.TRAIN))
                    _dataloader[_this_mode] = DataLoader(
                        dataset=self._dataset[_this_mode],
                        batch_size=self._batch_size,
                        num_workers=self._num_worker,
                        sampler=_sampler[_this_mode])
                # Set model optim and scheduler
                _model, _optim, _scheduler = self._Set_learning_model(_this_gpu_id)
                _model = DDP(_model, device_ids=[_this_node if _this_gpu_id == -1 else _this_gpu_id])
            # - Not use multi-process
            else:
                # Set dataloader and sampler
                for _this_mode in self._learning_mode:
                    _sampler[_this_mode] = None
                    _dataloader[_this_mode] = DataLoader(
                        dataset=self._dataset[_this_mode],
                        batch_size=self._batch_size,
                        num_workers=self._num_worker,
                        shuffle=_this_mode == Learning_Mode.TRAIN)
                # Set model optim and scheduler
                _model, _optim, _scheduler = self._Set_learning_model(_this_gpu_id)

            # Do learning process
            for _epoch in range(self._last_epoch + 1, self._max_epoch):
                _epoch_dir = Directory._make(f"{_epoch}", self._save_root) if _this_node is MAIN_RANK\
                    else f"{self._save_root}{Directory._Divider}{_epoch}{Directory._Divider}"

                for _this_mode in self._learning_mode:
                    _this_dataloader, _this_sampler = self._Active_mode_change_to(_this_mode, _model, _dataloader, _sampler)

                    # - When use sampler, shuffling
                    _this_sampler.set_epoch(_epoch) if _this_sampler is not None else ...

                    _mode_dir = Directory._make(f"{_this_mode.value}", _epoch_dir) if _this_node is MAIN_RANK\
                        else f"{_epoch_dir}{Directory._Divider}{_this_mode.value}{Directory._Divider}"
                    if _this_mode == Learning_Mode.TRAIN:
                        self._Learning(processer_num, _epoch, _this_mode, _this_dataloader, _model, _optim, _mode_dir, share_block)
                    else:
                        with no_grad():
                            self._Learning(processer_num, _epoch, _this_mode, _this_dataloader, _model, _optim, _mode_dir, share_block)

                _scheduler.step() if _scheduler is not None else ...

                # save log file
                if _this_node is MAIN_RANK:
                    self._tracker._insert({"Last_epoch": _epoch}, self._tracker._Annotation)
                    self._tracker._save(self._save_root, "trainer_log.json")

                    # save model
                    self._Save_weight(_epoch_dir, _model, _optim, _scheduler)

        def _Average_gradients(self, model: Custom_Model):
            size = float(distributed.get_world_size())
            for param in model.parameters():
                if param.grad is not None:
                    distributed.all_reduce(param.grad.data, op=distributed.ReduceOp.SUM)
                    param.grad.data /= size

        def _Work(self):
            _this_date = Utils.Time._apply_text_form(Utils.Time._stemp(), is_local=True, text_format="%Y-%m-%d")
            self._tracker._insert({"Date": _this_date}, self._tracker._Annotation)
            self._save_root = Directory._make(Directory._divider_check(f"{_this_date}/"), self._save_root)

            if self._multi_method == Multi_Method.DDP:
                _share_block = multiprocessing.Manager().Queue()
                _gpu_count = len(self._gpu_list)
                spawn(self._Process, nprocs=_gpu_count, args=(_share_block, ))
            else:
                self._Process(processer_num=0)

        # Un-Freeze function
        def _Learning(
                self,
                this_rank: int,
                epoch: int,
                mode: Learning_Mode,
                dataloader: DataLoader,
                model: MODEL,
                optim: Optimizer,
                save_dir: str,
                share_block: Optional[multiprocessing.Queue]):
            raise NotImplementedError

# class Reinforcment():
#     play_memory = namedtuple("play_memory", ["state", "action", "reward", "next_state", "ep_done"])

#     class basement(Deeplearnig.basement):
#         learning_opt: opt._learning.reinforcement = None

#         def __init__(self, learning_opt: opt._learning.reinforcement, log_opt: opt._log, data_opt: opt._data):
#             super().__init__(learning_opt, log_opt, data_opt)

#             self.memory: Deque[Reinforcment.play_memory] = deque([], maxlen=learning_opt.Memory_size)

#         def fit(self, epoch: int, mode: str = "train", is_display: bool = False, save_root: str = None):
#             super().fit(epoch, mode, is_display, save_root)
#             _data_loader = self.dataloader[mode]
#             _data_num = 0

#             save_root = directory._make(f"{mode}/", f"{save_root}")
#             save_dir = directory._make(f"{epoch}/", f"{save_root}")

#             for _state in _data_loader:  # minibatch
#                 _state, _ep_done, _state_ct = self.data_jump_to_gpu(_state)

#                 for _step_ct in range(self.learning_opt.Max_step):  # step
#                     _action = self.act(_state)
#                     _next_state, _ep_done = self.play(mode, _data_num, _step_ct, _action, _state, _ep_done, is_display, save_dir)

#                     if mode != "test":
#                         self.replay(mode)
#                         self.learning_opt.Exploration_threshold *= self.learning_opt.Exploration_discount

#                     _state = _next_state

#                 _data_num = self.log.progress_bar(epoch, mode, _data_num, _state_ct)

#             # result save
#             self.result_save(mode, epoch, save_root)

#         def act(self, state: Tensor) -> Tuple[Tensor, Tensor]:
#             pass

#         def get_reward(self, state, ep_done):
#             pass

#         def play(
#               self, mode: str, data_num: int, step_ct: int, action: Tensor, state: Tensor, _ep_done: Tensor, is_display: bool, save_root: str)
#  -> Tuple[Tensor, Tensor]:
#             pass

#         def dump_to_memory(self, state, action, reward, next_state, ep_done):
#             self.memory.append(Reinforcment.play_memory(state, action, reward, next_state, ep_done))

#         def replay(self):
#             pass

#         def result_save(self, mode: str, epoch: int, save_root: str):
#             pass

#     class DQN(basement):
#         def __init__(self, learnig_opt: opt._learning.reinforcement, log_opt: opt._log, data_opt: opt._data):
#             super().__init__(learnig_opt, log_opt, data_opt)

#             self.back_up_model: model.custom = None

#         def get_action(self, input):
#             pass

#         def get_reward(self, action):
#             pass

#         def replay(self):
#             pass

#     class A2C(basement):
#         def __init__(self, learnig_opt: opt._learning.reinforcement, log_opt: opt._log, data_opt: opt._data):
#             super().__init__(learnig_opt, log_opt, data_opt)

#             self.model: List[model.custom] = []  # [actor, critic]
#             self.optim: List[Optimizer] = []  # [actor, critic]

#         def set_model_n_optim(self, models: List[model.custom], optims: List[Optimizer], file_dir: List[str] = None):
#             _learning_rates = self.learning_opt.Learning_rate
#             _learning_rates = _learning_rates if isinstance(_learning_rates, list) else [_learning_rates for _ct in range(len(models))]
#             _file_dir = file_dir if isinstance(file_dir, list) else [file_dir for _ct in range(len(models))]

#             for _model, _optim, _lr, _file in zip(models, optims, _learning_rates, _file_dir):
#                 [model, optim] = super().set_model_n_optim(_model, _optim, _lr, _file)
#                 self.model.append(model)
#                 self.optim.append(optim)

#     class A3C(Deeplearnig):
#         def __init__(self, thred, action_size, is_cuda, Learning_rate, discount=0.9):
#             super().__init__(is_cuda, Learning_rate)
#             self.thred = thred
#             self.action_size = action_size
#             self.discount = discount

#             self.model = []  # [actor, critic]
#             self.optim = []
#             self.update_threshold = inf

#             self.log = log(
#                 factor_name=["Action_reward", "Action_eval"],
#                 num_class=2)

#         def set_model_optim(self, model_structure, optim, file_dir=None):
#             for _ct, [_model, _optim] in enumerate(zip(model_structure, optim)):
#                 self.model[_ct] = _model.cuda() if self.is_cuda else _model
#                 self.optim[_ct] = _optim[_ct](self.model[_ct].parameters(), self.LR)

#             if file_dir is not None:
#                 for _ct, _file in enumerate(file_dir):
#                     check_point = self.model[_ct]._load_from(_file)
#                     if check_point["optimizer_state_dict"] is not None:
#                         self.optim[_ct].load_state_dict(check_point["optimizer_state_dict"])

#         def model_update(self):
#             pass

#         def play(self, dataloader):
#             for data in dataloader:
#                 pass

#             pass
