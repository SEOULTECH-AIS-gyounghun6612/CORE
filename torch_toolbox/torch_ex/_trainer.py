# from random import sample
from math import inf, log10, floor
from collections import deque
from typing import List
from python_ex._base import utils, directory

if __package__ == "":
    # if this file in local project
    from torch_ex._base import torch_utils, opt, log
    from torch_ex._structure import custom_module
    from torch_ex import _dataloader
else:
    # if this file in package folder
    from ._base import torch_utils, opt, log
    from ._structure import custom_module
    from . import _dataloader


class opimizer():
    pass


class Deeplearnig():
    def __init__(self, learnig_opt: opt._learning.base, log_opt: opt._log, data_opt: opt._data):
        self.learning_opt = learnig_opt

        self.log_opt = log_opt
        self.log: log = log(opt=self.log_opt)
        self.data_opt = data_opt
        self.data_worker = _dataloader.data_worker(self.data_opt)
        self.dataloader = self.data_worker.get_dataloader(self.learning_opt.Modes)

        self.model = None
        self.optim = None

        self.learning_option_logging()

    def set_model_n_optim(self, model: custom_module, optim, file_dir=None):
        self.model = model.cuda() if self.learning_opt.is_cuda else model
        self.optim = optim(self.model.parameters(), self.learning_opt.Learning_rate)

        if file_dir is not None:
            self.load_model_n_optim(file_dir)

    def load_model_n_optim(self, file_dir):
        check_point = self.model._load_from(file_dir)
        if check_point["optimizer_state_dict"] is not None:
            self.optim.load_state_dict(check_point["optimizer_state_dict"])

    def print_Progress_Bar(self, learning_mode, epoch, data_num, decimals=1, length=25, fill='█'):
        _e_num_ct = floor(log10(self.learning_opt.Max_epochs)) + 1
        _epoch = f"{epoch}".rjust(_e_num_ct, " ")
        _epochs = f"{self.learning_opt.Max_epochs}".rjust(_e_num_ct, " ")

        data_size = self.log.log_info["dataloader"][learning_mode]["data_length"]
        _d_num_ct = floor(log10(data_size)) + 1
        _data_num = f"{data_num}".rjust(_d_num_ct, " ")
        _data_size = f"{data_size}".rjust(_d_num_ct, " ")

        _prefix = f"{learning_mode} {_epoch}/{_epochs} {_data_num}/{_data_size}"

        utils.Progress_Bar(data_num, data_size, _prefix, self.log.get_log_display(learning_mode), decimals, length, fill)

    def data_jump_to_gpu(self, datas):
        # if use gpu dataset move to datas
        pass

    def fit(self, mode="train"):
        pass

    def get_loss(self, _epoch, datas):
        # cal loss, update log
        pass

    def learning_option_logging(self):
        self.log.info_update("date", utils.time_stemp(True))
        self.log.info_update("dataloader", self.data_worker.info)


class Reinforcment():
    class DQN():
        def __init__(
            self, action_size, memory_size, is_cuda,
            discount=0.9, explore=[1, 0.1, 0.99]
        ):
            self.is_cuda = is_cuda

            self.action_size = action_size
            self.discount = discount
            self.exp = deque(maxlen=memory_size)

            self.explore_rate, self.explore_min, self.explore_decay = explore

            self.model = None
            self.back_up = None
            self.optimizer = None

        def set_model(self):
            pass

        def get_action(self, input):
            pass

        def get_reward(self, action):
            pass

        def play(self, episode_num, dataloader, is_train_mode=True, display=True, savedir=None):
            pass

        def replay(self):
            pass

        def load_from(self, dir):
            pass

        def save_to(self, dir):
            pass

    class A2C(Deeplearnig):
        def __init__(self, learnig_opt: opt._learning.reinforcement, log_opt: opt._log, dataloader_opt: opt._data):
            super().__init__(learnig_opt, log_opt, dataloader_opt)

            self.model: List[custom_module] = []  # [actor, critic]
            self.optim = []

        def play(self, episode, mode: str = "train", display=True, save_root=None):
            if mode == "train":
                [_model.train() for _model in self.model]
            else:
                [_model.eval() for _model in self.model]

            data_num = 0
            for _state in self.dataloader[mode]:
                data_num += _state[0].shape[0]
                _ep_done = torch_utils._tensor.holder([self.data_opt.Batch_size, 1], True, 0)
                _state = self.data_jump_to_gpu(_state)
                for _step_ct in range(self.learning_opt.Max_step):
                    _action, _ep_done = self.act(_state, _ep_done)
                    _state, _each_loss_list, _ep_done = self.get_loss(_step_ct, _state, _action, _ep_done, mode)

                    if mode == "train":
                        for _ct, _each_loss in enumerate(_each_loss_list):
                            self.optim[_ct].zero_grad()
                            _each_loss.backward()
                            self.optim[_ct].step()

                    if self.data_opt.Batch_size < 2:
                        if _ep_done:
                            break

                    self.print_Progress_Bar(mode, episode, data_num)

            # result save
            save_dir = directory._make(f"{episode}/", self.log_opt.Save_root)

            for _model in self.model:
                _model._save_to(save_dir, episode)

            self.log.save("train_log.json" if mode != "test" else "test_log.json")

        def set_model_n_optim(self, model: List[custom_module], optim, file_dir=None):
            _is_cuda = self.learning_opt.is_cuda
            _lr = self.learning_opt.Learning_rate

            for _ct, [_model, _optim] in enumerate(zip(model, optim)):
                self.model.append(_model.cuda() if _is_cuda else _model)
                self.optim.append(_optim(self.model[_ct].parameters(), _lr))

            if file_dir is not None:
                for _ct, _file in enumerate(file_dir):
                    check_point = self.model[_ct]._load_from(_file)
                    if check_point["optimizer_state_dict"] is not None:
                        self.optim[_ct].load_state_dict(check_point["optimizer_state_dict"])

        def data_jump_to_gpu(self, datas):
            # if use gpu dataset move to datas
            pass

        def act(self, state, ep_done):
            pass

        def get_reward(self, state, action, ep_done):
            # make next state from action and reward
            pass

        def get_loss(self, step, state, action, ep_done, mode):
            # cal loss, update log
            pass

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
