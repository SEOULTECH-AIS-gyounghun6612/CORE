# from random import sample
from dataclasses import asdict
from math import inf
from collections import deque
from typing import List, Tuple
from python_ex._base import utils, directory


if __package__ == "":
    # if this file in local project
    from torch_ex._base import torch_utils, opt, log
    from torch_ex._structure import custom_module, Tensor, Optimizer, _Loss
    from torch_ex import _data_process
else:
    # if this file in package folder
    from ._base import torch_utils, opt, log
    from ._structure import custom_module, Tensor, Optimizer, _Loss
    from . import _data_process


class Deeplearnig():
    def __init__(self, learning_opt: opt._learning.base, log_opt: opt._log):
        self.learning_opt = learning_opt

        # log_opt and log
        self.log_opt = log_opt
        self.log: log = log(opt=self.log_opt)

        self.learning_option_logging()

        # model and optim
        self.model: custom_module = None
        self.optim: Optimizer = None

    def learning_option_logging(self):
        learning_info = {}
        learning_info["date"] = utils.time_stemp(True)
        learning_info = asdict(self.learning_opt)
        self.log.info_update("learning", learning_info)

    def set_data_process(self, data_opt: opt._data):
        # data_opt and dataloader
        _batch_size = data_opt.Batch_size
        _num_workers = data_opt.Num_workers
        self.dataloader = {}
        for train_style in self.learning_opt.Train_style:
            self.dataloader[train_style] = _data_process.dataloader._make(
                _data_process.dataset.basement(data_opt, train_style),
                _batch_size,
                _num_workers,
                train_style == "train")

        # self.log.info_update("dataloader", self.data_worker.info)
        pass

    def set_model_n_optim(self, model: custom_module, optim: Optimizer, file_dir: str = None):
        self.model = model.cuda() if self.learning_opt.is_cuda else model
        self.optim = optim(self.model.parameters(), self.learning_opt.Learning_rate)

        if file_dir is not None:
            self.load_model_n_optim(file_dir)

    def load_model_n_optim(self, file_dir: str):
        check_point = self.model._load_from(file_dir)
        if check_point["optimizer_state_dict"] is not None:
            self.optim.load_state_dict(check_point["optimizer_state_dict"])

    def data_jump_to_gpu(self, datas):
        # if use gpu dataset move to datas
        pass

    def fit(self, mode="train"):
        pass

    def get_loss(self, _epoch, datas):
        # cal loss, update log
        pass


class Reinforcment():
    class DQN(Deeplearnig):
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
            self.learning_opt = learnig_opt
            self.model: List[custom_module] = []  # [actor, critic]
            self.optim: List[Optimizer] = []  # [actor, critic]

        def set_model_n_optim(self, model: List[custom_module], optim: List[Optimizer], file_dir: str = None):
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

        def play(self, epoch: int, mode: str = "train", display: bool = True, save_root: str = None):
            # for display, progressbar
            self.log.this_mode = mode
            self.log.this_epoch = epoch

            if mode == "train":
                [_model.train() for _model in self.model]
            else:
                [_model.eval() for _model in self.model]

            data_num = 0
            for _state in self.dataloader[mode]:
                data_num += _state[0].shape[0]
                _ep_done = torch_utils._tensor.holder([self.learning_opt.Batch_size, 1], True, 0)
                _state = self.data_jump_to_gpu(_state)
                for _step_ct in range(self.learning_opt.Max_step):
                    _action, _ep_done = self.act(_state, _ep_done)
                    _state, _each_loss_list, _ep_done = self.get_loss(_step_ct, _state, _action, _ep_done, mode)

                    if mode == "train":
                        for _ct, _each_loss in enumerate(_each_loss_list):
                            self.optim[_ct].zero_grad()
                            _each_loss.backward()
                            self.optim[_ct].step()

                    if self.learning_opt.Batch_size < 2:
                        if _ep_done:
                            break

                    self.log.progress_bar(data_num)

            # result save
            save_dir = directory._make(f"{epoch}/", self.log_opt.Save_root)

            for _model in self.model:
                _model._save_to(save_dir, epoch)

            self.log.save("train_log.json" if mode != "test" else "test_log.json")

        def reward_converter(self, raw_reward, ep_done):
            _reward_holder = raw_reward * 0.0

            _ths = self.learning_opt.Reward_th
            _values = self.learning_opt.Reward_value

            if not self.learning_opt.Reward_relation_range:
                for _ct, _reward_th in enumerate(_ths):
                    _reward_holder += (raw_reward >= _reward_th) * _values[_ct]
                _reward_holder += (raw_reward < _ths[-1]) * _values[-1]

            else:
                pass

            _under = ep_done * 1.0
            _reward_holder = (_reward_holder * (1 - _under)) + (_values[-1] * _under)

            _reward = torch_utils._tensor.from_numpy(_reward_holder)
            _reward = _reward.cuda() if self.learning_opt.is_cuda else _reward
            ep_done = torch_utils._tensor.from_numpy(ep_done)
            ep_done = ep_done.cuda() if self.learning_opt.is_cuda else ep_done

            return _reward, ep_done

        def data_jump_to_gpu(self, datas: List[Tensor]):
            # if use gpu dataset move to datas
            pass

        def act(self, state: Tensor, ep_done: bool):
            pass

        def get_reward(self, state: Tensor, action: Tensor, ep_done: bool):
            # make next state from action and reward
            pass

        def get_loss(self, step: int, state: Tensor, action: Tensor, ep_done: bool, mode: str) -> Tuple[Tensor, List[_Loss], Tensor]:
            # cal loss, update log
            # return [state, loss, ep_done]
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
