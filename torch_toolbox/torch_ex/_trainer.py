# from random import sample
from dataclasses import asdict
from math import inf
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Union, Deque

from python_ex._base import utils, directory


if __package__ == "":
    # if this file in local project
    from torch_ex._base import opt, log
    from torch_ex._structure import custom_module, Tensor, Optimizer  # , _Loss
    from torch_ex import _data_process
else:
    # if this file in package folder
    from ._base import opt, log
    from ._structure import custom_module, Tensor, Optimizer  # , _Loss
    from . import _data_process


class Deeplearnig():
    class basement():
        def __init__(self, learning_opt: opt._learning.base, log_opt: opt._log, data_opt: opt._data):
            # set learning option
            self.learning_opt = learning_opt

            # set log
            self.log: log = log(opt=log_opt, info=None)
            self.log.set_info({"Learning date": utils.time_stemp(is_text=True)})
            self.log.set_info({"Train_opt": asdict(self.learning_opt)})

            # set dataloader
            self.set_data_process(learning_opt.Learnig_style, data_opt)

            # model and optim
            self.model: Union[custom_module, List[custom_module]] = None
            self.optim: Union[Optimizer, List[Optimizer]] = None

        def set_data_process(self, learning_style: List[str], data_opt: opt._data):
            # dataloader from data option
            _batch_size = data_opt.Batch_size
            _num_workers = data_opt.Num_workers

            # dataloader
            self.dataloader: Dict[str, _data_process.DataLoader] = {}

            #  holder for logging
            _dataloader_log = {}

            for _style in learning_style:
                _temp_dataset = _data_process.dataset.basement(data_opt, _style)
                # When learning style is "train", Dataloader parameter shuffle is True. Else set False
                self.dataloader[_style] = _data_process.dataloader._make(_temp_dataset, _batch_size, _num_workers, _style == "train")
                _dataloader_log[_style] = {"data_length": _temp_dataset.__len__()}
            self.log.set_info({"Dataloader": _dataloader_log})

        def set_model_n_optim(self, model: custom_module, optim: Optimizer, learning_rate: float, file_dir: str = None):
            model = model.cuda() if self.learning_opt.is_cuda else model
            optim = optim(model.parameters(), learning_rate)

            if file_dir is not None:
                check_point = model._load_from(file_dir)
                if check_point["optimizer_state_dict"] is not None:
                    optim.load_state_dict(check_point["optimizer_state_dict"])

            return model, optim

        def save_model_and_optim(self, epoch: int, model: custom_module, save_dir: str, optim: Optimizer = None):
            save_dir = directory._slash_check(save_dir)
            model._save_to(save_dir=save_dir, epoch=epoch, optim=optim)

        def data_jump_to_gpu(self, data_list: List[Tensor]):
            # if use gpu dataset move to datas
            pass

        def fit(self, epoch: int, mode: str = "train", is_display: bool = False, save_root: str = None):
            if mode == "train":
                [_model.train() for _model in self.model] if isinstance(self.model, list) else self.model.train()
            else:
                [_model.eval() for _model in self.model] if isinstance(self.model, list) else self.model.eval()

        def result_save(self, mode: str, epoch: int, save_root: str):
            pass


class Reinforcment():
    play_memory = namedtuple("play_memory", ["state", "action", "reward", "next_state", "ep_done"])

    class basement(Deeplearnig.basement):
        learning_opt: opt._learning.reinforcement = None

        def __init__(self, learning_opt: opt._learning.reinforcement, log_opt: opt._log, data_opt: opt._data):
            super().__init__(learning_opt, log_opt, data_opt)

            self.memory: Deque[Reinforcment.play_memory] = deque([], maxlen=learning_opt.Memory_size)

        def fit(self, epoch: int, mode: str = "train", is_display: bool = False, save_root: str = None):
            super().fit(epoch, mode, is_display, save_root)
            _data_loader = self.dataloader[mode]
            _data_num = 0

            save_root = directory._make(f"{mode}/", f"{save_root}")
            save_dir = directory._make(f"{epoch}/", f"{save_root}")

            for _state in _data_loader:  # minibatch
                _state, _ep_done, _state_ct = self.data_jump_to_gpu(_state)

                for _step_ct in range(self.learning_opt.Max_step):  # step
                    _action = self.act(_state)
                    _next_state, _ep_done = self.play(mode, _data_num, _step_ct, _action, _state, _ep_done, is_display, save_dir)

                    if mode != "test":
                        self.replay(mode)
                        self.learning_opt.Exploration_threshold *= self.learning_opt.Exploration_discount

                    _state = _next_state

                _data_num = self.log.progress_bar(epoch, mode, _data_num, _state_ct)

            # result save
            self.result_save(mode, epoch, save_root)

        def act(self, state: Tensor) -> Tuple[Tensor, Tensor]:
            pass

        def get_reward(self, state, ep_done):
            pass

        def play(self, mode: str, data_num: int, step_ct: int, action: Tensor, state: Tensor, _ep_done: Tensor, is_display: bool, save_root: str) -> Tuple[Tensor, Tensor]:
            pass

        def dump_to_memory(self, state, action, reward, next_state, ep_done):
            self.memory.append(Reinforcment.play_memory(state, action, reward, next_state, ep_done))

        def replay(self):
            pass

        def result_save(self, mode: str, epoch: int, save_root: str):
            pass

    class DQN(basement):
        def __init__(self, learnig_opt: opt._learning.reinforcement, log_opt: opt._log, data_opt: opt._data):
            super().__init__(learnig_opt, log_opt, data_opt)

            self.back_up_model: custom_module = None

        def get_action(self, input):
            pass

        def get_reward(self, action):
            pass

        def replay(self):
            pass

    class A2C(basement):
        def __init__(self, learnig_opt: opt._learning.reinforcement, log_opt: opt._log, data_opt: opt._data):
            super().__init__(learnig_opt, log_opt, data_opt)

            self.model: List[custom_module] = []  # [actor, critic]
            self.optim: List[Optimizer] = []  # [actor, critic]

        def set_model_n_optim(self, models: List[custom_module], optims: List[Optimizer], file_dir: List[str] = None):
            _learning_rates = self.learning_opt.Learning_rate
            _learning_rates = _learning_rates if isinstance(_learning_rates, list) else [_learning_rates for _ct in range(len(models))]
            _file_dir = file_dir if isinstance(file_dir, list) else [file_dir for _ct in range(len(models))]

            for _model, _optim, _lr, _file in zip(models, optims, _learning_rates, _file_dir):
                [model, optim] = super().set_model_n_optim(_model, _optim, _lr, _file)
                self.model.append(model)
                self.optim.append(optim)

    class A3C(Deeplearnig):
        def __init__(self, thred, action_size, is_cuda, Learning_rate, discount=0.9):
            super().__init__(is_cuda, Learning_rate)
            self.thred = thred
            self.action_size = action_size
            self.discount = discount

            self.model = []  # [actor, critic]
            self.optim = []
            self.update_threshold = inf

            self.log = log(
                factor_name=["Action_reward", "Action_eval"],
                num_class=2)

        def set_model_optim(self, model_structure, optim, file_dir=None):
            for _ct, [_model, _optim] in enumerate(zip(model_structure, optim)):
                self.model[_ct] = _model.cuda() if self.is_cuda else _model
                self.optim[_ct] = _optim[_ct](self.model[_ct].parameters(), self.LR)

            if file_dir is not None:
                for _ct, _file in enumerate(file_dir):
                    check_point = self.model[_ct]._load_from(_file)
                    if check_point["optimizer_state_dict"] is not None:
                        self.optim[_ct].load_state_dict(check_point["optimizer_state_dict"])

        def model_update(self):
            pass

        def play(self, dataloader):
            for data in dataloader:
                pass

            pass
