# from random import sample
# from math import inf
# from collections import deque, namedtuple
from typing import Dict, List, Tuple
from torch import Tensor

from python_ex._base import directory, file


if __package__ == "":
    # if this file in local project
    from torch_ex._torch_base import learing_mode, opt, debug, Optimizer
    from torch_ex._structure import module  # , _Loss
    from torch_ex._data_process import dataset, DataLoader, make_dataloader
else:
    # if this file in package folder
    from ._torch_base import learing_mode, opt, debug, Optimizer
    from ._structure import module  # , _Loss
    from ._data_process import dataset, DataLoader, make_dataloader


class Learning_process():
    class End_to_End():
        def __init__(self, learning_opt: opt._learning.E2E, dataloader_opt: opt._dataloader, restore_file: str = None) -> None:
            is_restore = file._exist_check(restore_file)

            # set log
            if is_restore:
                # restore trainer opts from log
                self.learning_opt, self.dataloader_opt = self.trainer_restore()
                self.save_root = self.learning_opt.make_save_directorty()

            else:
                # set learning base option
                self.learning_opt = learning_opt
                self.dataloader_opt = dataloader_opt

                # set base parameter
                self.save_root = self.learning_opt.make_save_directorty()

                # make new log
                self.log = debug.process_log(self.learning_opt.Logging_parameters, save_dir=self.save_root, file_name=self.learning_opt.Log_file)

            # set dataloader
            self.set_data_process()

            # model and optim
            self.model: List[module.custom_module] = []
            self.optim: List[Optimizer] = []

            # set learning mode -> default: First mode in learning_opt
            self.set_learning_mode(self.learning_opt.Learning_mode[0])

        def trainer_restore(self, restore_file) -> Tuple[opt._learning.E2E, opt._dataloader]:
            _file_dir, _file_name = file._name_from_path(restore_file, False)

            # get log data from log file
            self.log = debug.process_log({}, save_dir=_file_dir, file_name=_file_name, is_restore=True)
            return self.log.get_restore_opt()  # make it get_restore_opt

        def set_learning_mode(self, mode: learing_mode):
            # set log state
            self.log.set_logging_mode(mode)

            # set model state
            if mode == learing_mode.TRAIN:
                [_model.train() for _model in self.model]
            else:
                [_model.eval() for _model in self.model]

        def set_data_process(self):
            # dataloader dict
            self.dataloaders: Dict[learing_mode, DataLoader] = {}

            for _learning_mode in self.learning_opt.Learning_mode:
                # set dataloader in each learning mode
                _dataloader = make_dataloader(self.dataloader_opt, _learning_mode, dataset.basement)
                self.dataloaders[_learning_mode] = _dataloader

                # Update log; count of data in each mode
                self.log.info_update("dataloader", {_learning_mode: {"data_count": _dataloader.dataset.__len__(), "batch_size": _dataloader.batch_size}})

        # --- additional editing be optinary, when except make a new learning_trainer --- #
        def set_learning_model(self, block_list: List[Tuple[module.custom_module, opt._optim_opt]], is_resotre: bool):
            for _count, [_module, _optim] in enumerate(block_list):
                self.model.append(_module.cuda() if self.learning_opt.Use_cuda else _module)

                if is_resotre:
                    ...

                else:
                    self.optim.append(_optim.make(self.model[_count], self.learning_opt.LR_initail[_count]), )

        def save_learning_moduel(self, epoch: int):
            # in later make save folder function in log (epoch (int) -> save folder (str))
            save_folder = directory._make(f"{epoch}/", self.save_root)
            for _ct, _model in enumerate(self.model):
                _model._save_to(save_dir=save_folder, epoch=epoch, optim=self.optim[_ct])

        def save_log(self):
            self.log.save()
        # --- -------------------------------------------------------------------------- --- #

        # --- must edit function, when before use it --- #
        def data_jump_to_gpu(self, data_list: List[Tensor]):
            # if use gpu dataset move to datas
            ...

        def fit(self, epoch: int = 0, mode: learing_mode = learing_mode.TRAIN, is_display: bool = True, is_debug_save: bool = True):
            ...

        def result_save(self, mode: str, epoch: int):
            ...
        # --- -------------------------------------- --- #


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
