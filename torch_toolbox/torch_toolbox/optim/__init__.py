from python_toolbox.registry import Registry
from torch.optim.lr_scheduler import LRScheduler

SCHEDULER = Registry[type[LRScheduler]]("scheduler", LRScheduler)
