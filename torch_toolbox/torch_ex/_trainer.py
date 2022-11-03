from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Union, Type

from torch import distributed, cuda, save, load, multiprocessing
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from python_ex._base import Directory, File, Utils, OS_Style
from python_ex._label import Label_Process_Config, Label_Style, Input_Style, IO_Style, Label_Process


if __package__ == "":
    # if this file in local project
    from torch_ex._torch_base import Learning_Mode, Debug, Log_Config
    from torch_ex._dataloader import Custom_Dataset, Dataset_Config
    from torch_ex._layer import Custom_Model, Custom_Model_Config
    from torch_ex._optimizer import _LRScheduler, Scheduler_Config, Custom_Scheduler
else:
    # if this file in package folder
    from ._torch_base import Learning_Mode, Debug, Log_Config
    from ._layer import Custom_Model, Custom_Model_Config
    from ._optimizer import _LRScheduler, Scheduler_Config, Custom_Scheduler


# -- DEFINE CONFIG -- #
@dataclass
class Learning_Config():
    @dataclass
    class E2E(Utils.Config):
        ### config; log, dataloader, schedule
        _Log_config: Log_Config
        _Dataset_config: Dataset_Config
        _Schedule_config: Scheduler_Config

        ### Infomation about learning
        _Project_Name: str = "End_to_End_learning"
        _Detail: str = "Empty"
        _Date: str = Utils._time_stemp(is_text=True, is_local=True, text_format="%Y-%m-%d")
        _Save_root: str = Directory._relative_root()

        ### About Learning type and style
        _Batch_size: int = 4
        _Num_workers: int = 8

        _Max_epochs: int = 100
        _Last_epoch: int = -1
        _Learning_list: List[Learning_Mode] = field(default_factory=lambda: [Learning_Mode.TRAIN, Learning_Mode.VALIDATION])

        ### About GPU using
        _Num_of_node: int = 1
        _GPU_list: List[int] = field(default_factory=lambda: list(range(cuda.device_count())))
        _Host_address: str = "tcp://127.0.0.1:10001"
        _This_node_rank: int = 0

        def _get_parameter(self) -> Dict[str, Any]:
            return super()._get_parameter()

        def _convert_to_dict(self) -> Dict[str, Union[Dict, str, int, float, bool, None]]:
            _dict = {
                "_Project_Name": self._Project_Name,
                "_Detail": self._Detail,
                "_Date": self._Date,

                "_Log_config": self._Log_config._convert_to_dict(),
                "_Dataset_config": self._Dataset_config._convert_to_dict(),
                "_Optimizer_config": self._Optimizer_config._convert_to_dict(),
                "_Schedule_config": self._Schedule_config._convert_to_dict(),

                "_Max_epochs": self._Max_epochs,
                "_Start_epoch": self._Last_epoch,
                "_Activate_mode": [_mode.value for _mode in self._Learning_list]}
            return _dict

        def _restore_from_dict(self, data: Dict[str, Union[Dict, str, int, float, bool, None]]):
            self._Log_config._restore_from_dict(data["_Log_config"])
            self._Dataset_config._restore_from_dict(data["_Dataloader_config"])
            self._Optimizer_config._restore_from_dict(data["_Optimizer_config"])
            self._Schedule_config._restore_from_dict(data["_Schedule_config"])

            self._Project_Name = data["_Project_Name"]
            self._Detail = data["_Detail"]
            self._Date = data["_Date"]

            self._Max_epochs = data["_Max_epochs"]
            self._Last_epoch = data["_Start_epoch"]
            self._Learning_list = data["_Activate_mode"]

        def _load_config_from_file(self, config_directory: str, restore_file: str):
            _config_data = File._json(config_directory, restore_file)
            self._restore_from_dict(_config_data)

    # in later fix it
    @dataclass
    class Reinforcement(E2E):
        # reinforcement train option
        Max_step: int = 100

        Q_discount: float = 0.99

        Reward_threshold: List[float] = field(default_factory=list)
        Reward_value: List[float] = field(default_factory=list)
        Reward_fail: float = -10
        # Reward_relation_range: int = 80

        # action option
        Action_size: List[Union[int, List[int]]] = field(default_factory=list)
        Action_range: List[List[int]] = field(default_factory=list)  # [[Max, Min]]

        # replay option
        Memory_size: int = 1000
        Minimum_memroy_size: int = 100
        Exploration_threshold: int = 1.0
        Exploration_discount: float = 0.99
        Exploration_Minimum: int = 1.0


# -- Mation Function -- #
class Learning_process():
    class End_to_End():
        def __init__(self, learning_config: Learning_Config.E2E, ) -> None:
            self._Learning_option = learning_config

            self._Learning_list = learning_config._Learning_list

            # result save dir
            _project_result_dir = f"{learning_config._Project_Name}{Directory._Divider}"
            _project_result_dir += f"{learning_config._Date}{Directory._Divider}"
            self._Save_root = Directory._make(_project_result_dir, learning_config._Save_root)

            # learning option
            self._Last_epoch = learning_config._Last_epoch
            self._Batch_size = learning_config._Batch_size
            self._Num_worker = learning_config._Num_workers

            # distribute option
            self._Word_size = learning_config._Num_of_node * len(learning_config._GPU_list)
            self._This_node_rank = learning_config._This_node_rank
            self._GPU_list = learning_config._GPU_list
            self._Use_distribute = (Directory._OS_THIS == OS_Style.OS_UBUNTU.value) and (len(learning_config._GPU_list) >= 2)

            self._Model_stemp: Type[Custom_Model] = None
            self._Model_config: Custom_Model_Config = None

        # Freeze function
        # --- for init function --- #
        def _set_log(self, log_opt: Log_Config):
            self._Log = Debug.Learning_Log(log_opt)
            self._Log._insert(self._Learning_option._convert_to_dict())

        def _set_dataset(
                self, label_config: Label_Process_Config, label_style: Label_Style, label_io: IO_Style, input_style: Input_Style, input_io: IO_Style,
                amplification: int, augmentation: Dict[Learning_Mode, Dict[str, List[Utils.Config]]]):

            self._Dataset = {}
            for _mode in self._Learning_list:
                _label_process = Label_Process._build(**label_config._get_parameter())
                _label_process._set_learning_mode(_mode)
                _augmentation = augmentation[_mode]

                self._Dataset[_mode] = Custom_Dataset(_label_process, label_style, label_io, input_style, input_io, amplification, _augmentation)

        def _set_model_n_optim_config(self, model_stemp: Type[Custom_Model], model_config: Custom_Model_Config, schedule_config: Scheduler_Config):
            self._Model_stemp = model_stemp
            self._Model_config = model_config
            self._Schedule_config = schedule_config

        # --- for work function --- #
        def _set_process(self, word_size: int, this_rank: int, method: str):
            def _print_lock(is_master: bool):
                import builtins as __builtin__
                builtin_print = __builtin__.print

                def print(*args, **kwargs):
                    force = kwargs.pop('force', False)
                    if is_master or force:
                        builtin_print(*args, **kwargs)
                __builtin__.print = print

            # _print_lock(not this_rank)
            distributed.init_process_group(backend="nccl", init_method=method, world_size=word_size, rank=this_rank)

        def _set_dataloader(self, this_rank: int = 0):
            _dataloader: Dict[Learning_Mode, DataLoader] = {}
            _sampler: Dict[Learning_Mode, DistributedSampler] = {}

            #  Use distribute or Not
            for _mode in self._Learning_option._Learning_list:
                if self._Use_distribute:
                    _sampler[_mode] = DistributedSampler(self._Dataset[_mode], rank=this_rank, shuffle=(_mode == Learning_Mode.TRAIN))
                else:
                    _sampler[_mode] = None

                # set dataloader in each learning mode
                _dataloader[_mode] = DataLoader(dataset=self._Dataset[_mode], batch_size=self._Batch_size, num_workers=self._Num_worker, sampler=_sampler[_mode])

            return _sampler, _dataloader

        def _set_learning_model(self, this_gpu: int = 0, is_cuda: bool = False) -> Tuple[Custom_Model, Optimizer, _LRScheduler]:
            _model = self._Model_stemp(**self._Model_config._get_parameter())
            _model = _model.cuda(this_gpu) if is_cuda else _model
            _optim, _schedule = Custom_Scheduler._build(**self._Schedule_config._get_parameter(_model))

            return _model, _optim, _schedule

        def _set_activate_mode(self, mode: Learning_Mode):
            # set log state
            self._Active_mode = mode
            self._Log._set_activate_mode(mode)

        def _save_model(self, save_dir: str, model: Custom_Model, optim: Optimizer = None, schedule: _LRScheduler = None):
            save(model.state_dict(), f"{save_dir}model.h5")  # save model state

            if optim is not None:
                _optim_and_schedule = {
                    "optimizer": None if optim is None else optim.state_dict(),
                    "schedule": None if schedule is None else schedule}
                save(_optim_and_schedule, f"{save_dir}optim.h5")  # save optim and schedule state

        def _load_model(self, save_dir: str, model: Custom_Model, optim: Optimizer = None, schedule: _LRScheduler = None):
            _model_file = f"{save_dir}model.h5"
            if File._exist_check(_model_file):
                model.load_state_dict(load(_model_file))

            _optim_file = f"{save_dir}optim.h5"
            if File._exist_check(_model_file) and optim is not None:
                _optim_and_schedule = load(_optim_file)
                optim.load_state_dict(_optim_and_schedule["optimizer"])
                if schedule is not None:
                    schedule.load_state_dict(_optim_and_schedule["schedule"])

            return model, optim, schedule

        def _progress_dispaly(self, mode: Learning_Mode, epoch: int, data_count: int, max_data_count: int, decimals: int = 1, length: int = 25, fill: str = '█'):
            _epoch_board = Utils._progress_board(epoch, self._Learning_option._Max_epochs)
            _data_board = Utils._progress_board(data_count, max_data_count)

            _batch_size = self._Learning_option._Batch_size
            _batch_ct = data_count // _batch_size + int(data_count % _batch_size)
            _max_batch_ct = max_data_count // _batch_size + int(max_data_count % _batch_size)

            _this_time, _max_time = self._Log._get_learning_time(_batch_ct, _max_batch_ct)
            _this_time_str = Utils._time_stemp(_this_time, False, True, "%H:%M:%S")
            _max_time_str = Utils._time_stemp(_max_time, False, True, "%H:%M:%S")

            _pre = f"{self._Active_mode.value} {_epoch_board} {_data_board} {_this_time_str}/{_max_time_str} "
            _suf = self._Log._learning_tracking(_batch_ct, data_count)

            Utils._progress_bar(data_count, max_data_count, _pre, _suf, decimals, length, fill)

        def _process(self, gpu: int = 0, gpu_per_node: int = 1):
            _is_cuda = len(self._GPU_list)
            _this_rank = self._This_node_rank * gpu_per_node + gpu
            _this_gpu_id = self._GPU_list[gpu] if _is_cuda else gpu

            if self._Use_distribute:
                self._set_process(self._Word_size, _this_rank, self._Learning_option._Host_address)

            _sampler, _dataloader = self._set_dataloader(_this_rank)
            _model, _optim, _schedule = self._set_learning_model(_this_gpu_id, _is_cuda)

            if self._Use_distribute:
                _model = _model = DistributedDataParallel(_model, device_ids=[_this_gpu_id])

            # Do learning process
            for _epoch in range(self._Last_epoch + 1, self._Learning_option._Max_epochs):
                _epoch_dir = Directory._make(f"{_epoch}/", self._Save_root)
                self._learning(_epoch, _sampler, _dataloader, _model, _optim)

                if _schedule is not None:
                    _schedule.step()

                # save log file
                self._Log._insert({"_Last_epoch": _epoch})
                self._Log._save(self._Save_root, "learning_log.json")

                # save model
                self._save_model(_epoch_dir)

        def _work(self):
            if self._Use_distribute:
                multiprocessing.spawn(self._process, nprocs=len(self._GPU_list), )
            else:
                self._process()

        # Un-Freeze function
        def _learning(
                self,
                epoch: int,
                sampler: Dict[Learning_Mode, DistributedSampler],
                dataloader: Dict[Learning_Mode, DataLoader],
                model: Custom_Model,
                optim: Optimizer):
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

#         def play(self, mode: str, data_num: int, step_ct: int, action: Tensor, state: Tensor, _ep_done: Tensor, is_display: bool, save_root: str) -> Tuple[Tensor, Tensor]:
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
