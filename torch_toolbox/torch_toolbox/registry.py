from typing import Callable, Any

from torch.optim.lr_scheduler import LRScheduler

from python_toolbox.registry import Registry
from python_toolbox.project import Base_Config

from .modules.definition import Composable_Module
from .modules.model.definition import Trainable_Model
from .dataloader.definition import Custom_Dataset
from .metric.definition import Accumulator


CFGS = Registry[type[Base_Config]]("torch_config", Base_Config)
LOSSES = Registry[type[Composable_Module]]("losses", Composable_Module)
MODELS = Registry[type[Trainable_Model]]("models", Trainable_Model)

DATASETS = Registry[type[Custom_Dataset]]("dataset", Custom_Dataset)
DATALOADER_FN = Registry[Callable[[Any], Any]](
    "dataloader_collect_fn", Callable[[Any], Any])

SCHEDULER = Registry[type[LRScheduler]]("scheduler", LRScheduler)
ACCUMULATORS = Registry[type[Accumulator]]("accumulators", Accumulator)
