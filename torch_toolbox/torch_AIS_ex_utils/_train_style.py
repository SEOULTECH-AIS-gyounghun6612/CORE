# from random import sample
from math import log10, ceil, inf

from python_AIS_ex_utils import _base
from collections import deque

if __package__ == "":
    # if this file in local project
    import _torch_util
else:
    # if this file in package folder
    from . import _torch_util


class Deeplearnig():
    def __init__(self, is_cuda, Learning_rate):
        self.is_cuda = is_cuda
        self.LR = Learning_rate

        self.model = None
        self.optim = None
        self.loss = None

        self.log = None

    def fit(self, epochs, data_loader, mode="train"):
        self.model.train() if mode == "train" else self.model.eval()
        for _epoch in range(epochs):
            _step_ct = 0
            for _datas in data_loader:  # _datas -> [_input, _label] or [_input,]
                _datas = self.data_jump_to_gpu(_datas, mode) if self.is_cuda else _datas
                _output = self.model(_datas[0])
                _step_ct += _output.shape[0]

                # test
                if mode == "test":
                    return _output

                # train or validation
                _each_loss = self.log_update(_epoch, [_output, _datas[1]])
                if mode == "train":
                    self.optim.zero_grad()
                    _each_loss.backward()
                    self.optim.step()

                _base.etc.Progress_Bar(
                    iteration=_step_ct,
                    total=data_loader.dataset.__len__(),
                    prefix=mode + " epoch: " + str(_epoch).zfill(ceil(log10(epochs))),
                    suffix=self.log.get_last_log(),
                    decimals=3,
                    length=40)

    def data_jump_to_gpu(self, datas):
        # if use gpu dataset move to datas
        pass

    def set_model_optim(self, model_structure, optim, file_dir=None):
        self.model = model_structure.cuda() if self.is_cuda else model_structure
        self.optim = optim(self.model.parameters(), self.LR)

        if file_dir is not None:
            self.load_model_n_optim(file_dir)

    def load_model_n_optim(self, file_dir):
        check_point = self.model._load_from(file_dir)
        if check_point["optimizer_state_dict"] is not None:
            self.optim.load_state_dict(check_point["optimizer_state_dict"])

    def log_update(self, _epoch, datas):
        # cal loss, update log
        pass


class Reinforcment():
    @staticmethod
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

    @staticmethod
    class A2C(Deeplearnig):
        def __init__(self, action_size, is_cuda, Learning_rate, discount=0.9):
            super().__init__(is_cuda, Learning_rate)
            self.action_size = action_size
            self.discount = discount

            self.model = []  # [actor, critic]
            self.optim = []
            self.loss = None

            self.log = _torch_util.log(
                factor_name=["loss", "Action_eval"],
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

        def data_jump_to_gpu(self, datas):
            # if use gpu dataset move to datas
            pass

        def act(self, datas, _ep_done):
            pass

        def get_reward(self, action, datas, ep_done):
            # make next state from action and reward
            pass

        def play(self, state, max_step, mode="train", display=True, save_root=None):
            if mode == "train":
                [self.model[_ct].train() for _ct in range(len(self.model))]
            else:
                [self.model[_ct].eval() for _ct in range(len(self.model))]

            _ep_done = False
            _state = self.data_jump_to_gpu(state) if self.is_cuda else state
            for _step_ct in range(max_step):
                _action, _ep_done = self.act(_state, _ep_done)
                _state, _each_loss_list, _ep_done = self.log_update(_step_ct, [_state, _action, _ep_done])

                if mode == "train":
                    for _ct, _each_loss in enumerate(_each_loss_list):
                        self.optim[_ct].zero_grad()
                        _each_loss.backward()
                        self.optim[_ct].step()

                if _ep_done:
                    break

        def log_update(self, epoch, datas):
            # cal loss, update log
            pass

    @staticmethod
    class A3C(Deeplearnig):
        def __init__(self, thred, action_size, is_cuda, Learning_rate, discount=0.9):
            super().__init__(is_cuda, Learning_rate)
            self.thred = thred
            self.action_size = action_size
            self.discount = discount

            self.model = []  # [actor, critic]
            self.optim = []
            self.update_threshold = inf

            self.log = _torch_util.log(
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
