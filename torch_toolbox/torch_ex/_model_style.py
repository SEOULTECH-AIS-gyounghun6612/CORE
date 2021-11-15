# from random import sample
from collections import deque

if __package__ == "":
    # if this file in local project
    import _torch_util
else:
    # if this file in package folder
    from . import _torch_util


class base_style():
    def __init__(self, is_cuda):
        self.is_cuda = is_cuda

        self.model = None
        self.optim = None

    def set_model(self):
        pass

    def set_optim(self):
        pass

    def get_loss(self):
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
    class A2C(base_style):
        def __init__(
            self, action_size, is_cuda,
            discount=0.9
        ):
            super().__init__(is_cuda)
            self.action_size = action_size
            self.discount = discount

            self.log = _torch_util.log(
                factor_name=["Action_reward", "Action_eval"],
                num_class=2)

        def act(self, episode_num, datas, _ep_done, display, savedir):
            pass

        def play(self, episode_num, dataloader, is_train=True, display=True, savedir=None):
            for _ct, datas in enumerate(dataloader):
                _ep_done = False
                # if train used GPU, set cuda
                datas = [data.cuda() for data in datas] if self.is_cuda else datas

                _loss = self.act(episode_num, datas, _ep_done, display, savedir)

                if is_train:
                    self.optim.zero_grad()
                    _loss.backward()
                    self.optim.step()

                else:
                    # for validation log
                    pass

        def load_from(self, file_dir):
            check_point = self.model._load_from(file_dir)
            None if "optimizer_state_dict" not in check_point.keys() else \
                self.optim.load_state_dict(check_point["optimizer_state_dict"])

        def save_to(self, file_dir, epoch, optim_save=False):
            self.model._save_to(file_dir, epoch, optim=optim_save)
