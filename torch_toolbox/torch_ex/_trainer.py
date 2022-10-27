# from random import sample
# from math import inf
# from collections import deque, namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Any, Union
from torch import cuda, save, load, Tensor

from python_ex._base import Directory, Utils


if __package__ == "":
    # if this file in local project
    from torch_ex._torch_base import Learning_Mode, Debug, Log_Config
    from torch_ex._data_process import Dataloder_Config, DataLoader
    from torch_ex._layer import Custom_Module
    from torch_ex._optimizer import Optimizer, _LRScheduler, Scheduler_Config, Custom_Scheduler, Optimizer_Config
else:
    # if this file in package folder
    from ._torch_base import Learning_Mode, Debug, Log_Config
    from ._data_process import Dataloder_Config, DataLoader
    from ._layer import Custom_Module
    from ._optimizer import Optimizer, _LRScheduler, Scheduler_Config, Custom_Scheduler, Optimizer_Config


# -- DEFINE CONFIG -- #
class Learning_Config():
    @dataclass
    class E2E(Utils.Config):
        # config; log, dataloader, schedule
        _Log_config: Log_Config
        _Dataloader_config: Dataloder_Config
        _Optimizer_config: Optimizer_Config
        _Schedule_config: Scheduler_Config

        # Infomation about learning
        _Project_Name: str = "End_to_End_learning"
        _Anotation: str = "Empty"
        _Date: str = Utils._time_stemp(is_text=True, is_local=True)

        # About Learning type and style
        _Max_epochs: int = 100
        _Start_epoch: int = 0
        _Activate_mode: List[Learning_Mode] = field(default_factory=lambda: [Learning_Mode.TRAIN, Learning_Mode.VALIDATION])

        # About GPU using
        _Use_cuda: bool = cuda.is_available()

        def _convert_to_dict(self) -> Dict[str, Any]:
            _dict = {
                "_Project_Name": self._Project_Name,
                "_Anotation": self._Anotation,
                "_Date": self._Date,

                "_Log_config": self._Log_config._convert_to_dict(),
                "_Dataloader_config": self._Dataloader_config._convert_to_dict(),
                "_Optimizer_config": self._Optimizer_config._convert_to_dict(),
                "_Schedule_config": self._Schedule_config._convert_to_dict(),

                "_Max_epochs": self._Max_epochs,
                "_Start_epoch": self._Start_epoch,
                "_Activate_mode": [__mode.value for __mode in self._Activate_mode],
                "_Use_cuda": self._Use_cuda}
            return _dict

        def _restore_from_dict(self, data: Dict[str, Any]):
            self._Log_config._restore_from_dict(data["_Log_config"])
            self._Dataloader_config._restore_from_dict(data["_Dataloader_config"])
            self._Optimizer_config._restore_from_dict(data["_Optimizer_config"])
            self._Schedule_config._restore_from_dict(data["_Schedule_config"])

            self._Project_Name = data["_Project_Name"]
            self._Anotation = data["_Anotation"]
            self._Date = data["_Date"]

            self._Max_epochs = data["_Max_epochs"]
            self._Start_epoch = data["_Start_epoch"]
            self._Activate_mode = data["_Activate_mode"]
            self._Use_cuda = data["_Use_cuda"]

        @staticmethod
        def _restore(restore_directory: str, restore_file: str):
            ...

    # in later fix it
    @dataclass
    class reinforcement(E2E):
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
        def __init__(self, learning_config: Learning_Config.E2E) -> None:
            self._Learning_option = learning_config
            self._set_log()
            self._set_dataloader()

            # model and optim
            self._Model: Custom_Module.Model
            self._Optim: Optimizer
            self._Schedule: _LRScheduler

        # Freeze function
        def _set_log(self):
            self._Log = Debug.Learning_Log(self._Learning_option._Log_config)
            self._Log._insert(self._Learning_option._convert_to_dict())

            __project_name = f"{self._Learning_option._Project_Name}{Directory._Divider}{self._Learning_option._Date}"
            __save_root = self._Learning_option._Log_config._Save_root

            self._Save_root = Debug._make_result_directory(__project_name, __save_root)

        def _set_dataloader(self):
            self._Dataloader: Dict[Learning_Mode, DataLoader] = {}
            # dataloader dict
            for _learning_mode in self._Learning_option._Activate_mode:
                # set dataloader in each learning mode
                self._Dataloader[_learning_mode] = self._Learning_option._Dataloader_config._make_dataloader(_learning_mode)

        def _set_learning_model(self, model: Custom_Module.Model):
            self._Model = model.cuda() if self._Learning_option._Use_cuda else model
            self._Optim = self._Learning_option._Optimizer_config._make_optim(self._Model)
            self._Schedule = Custom_Scheduler._build(self._Learning_option._Schedule_config, self._Optim, (self._Learning_option._Start_epoch - 1))

        def _set_activate_mode(self, mode: Learning_Mode):
            # set log state
            self._Log._set_activate_mode(mode)

            # set model state
            if mode == Learning_Mode.TRAIN:
                self._Model.train()
            else:
                self._Model.eval()

        def _save_model(self, save_dir: str):
            save(self._Model.state_dict(), f"{save_dir}model.h5")  # save model state

            __optim_and_schedule = {
                "optimizer": self._Optim.state_dict(),
                "schedule": None if self._Schedule is None else self._Schedule.state_dict()}
            save(__optim_and_schedule, f"{save_dir}optim.h5")  # save optim and schedule state

        def _restore(self, save_dir: str):
            self._Model.load_state_dict(load(f"{save_dir}model.h5"))

            __optim_and_schedule = load(f"{save_dir}optim.h5")
            self._Optim.load_state_dict(__optim_and_schedule["optimizer"])
            self._Schedule.load_state_dict(__optim_and_schedule["schedule"])

        def fit(self):
            # Do learning process
            for __epoch in range(self._Learning_option._Start_epoch, self._Learning_option._Max_epochs):
                _epoch_dir = Directory._make(f"{__epoch}/", self._Save_root)
                self._process(__epoch, _epoch_dir)

                if self._Schedule is not None:
                    self._Schedule.step()

                # save log file
                self._Log._save()

                # save model
                self._save_model(_epoch_dir)

        # Un-Freeze function
        def _process(self, mode: Learning_Mode, epoch_dir: str, is_display: bool = True) -> str:
            raise NotImplementedError

        def _get_loss(self, output: Union[Tensor, List[Tensor]], label: Union[Tensor, List[Tensor]]):
            ...

        def _result_process(self, data: Union[Tensor, List[Tensor]], save_dir: str):
            ...


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
