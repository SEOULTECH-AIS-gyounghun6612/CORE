from enum import Enum
from typing import Tuple, Dict, List, Any

from torch import Tensor, load, save, device
from torch.autograd.grad_mode import no_grad
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader

from python_ex.system import Path
from python_ex.project import Template


class Mode(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class LearningProcess(Template):
    def __init__(
        self,
        project_name: str, apply_mode: List[Mode], max_epoch: int,
        last_epoch: int = -1, gpus: List[int] | None = None,
        description: str | None = None, result_root: str | None = None
    ):
        super().__init__(project_name, description, result_root)

        # base info
        self.max_epoch = max_epoch
        self.last_epoch = last_epoch + 1
        self.apply_mode = apply_mode

        self.gpus = [] if gpus is None else gpus

        # debug info
        self.max_data_ct: Dict[Mode, int] = {}
        self.loss: Dict[Mode, Dict[str, List[float]]] = {}
        self.holder: Dict[Mode, Dict[str, List[float]]] = {}

    def Set_dataset(
        self, mode: Mode, dataset_config: Dict[str, Any]
    ) -> Dataset:
        raise NotImplementedError

    def Set_dataloader(
        self, dataset: Dataset, dataloader_config: Dict[str, Any]
    ) -> DataLoader:
        return DataLoader(dataset, **dataloader_config)

    def Set_model(
        self, device_info: device, model_config: Dict[str, Any]
    ) -> Module:
        raise NotImplementedError

    def Set_optimizer(
        self, model: Module, optimizer_config: Dict[str, Any]
    ) -> Tuple[Optimizer, LRScheduler | None]:
        raise NotImplementedError

    def Set_loss(self, loss_config: Dict[str, Any]):
        raise NotImplementedError

    def Set_holder(self, debug_config: Dict[str, Any]):
        _holder_name: List[str] = debug_config["holder_name"]
        self.holder = dict((
            _mode,
            dict((
                _name,
                [0.0 for _ in range(self.last_epoch, self.max_epoch)]
            ) for _name in _holder_name)
        ) for _mode in self.apply_mode)

    def Set_intput_n_target(
        self, device_info: device, data: Tensor | List[Tensor]
    ) -> Tuple[Tensor | List[Tensor], Tensor | List[Tensor], Any]:
        raise NotImplementedError

    def Get_output(
        self,
        mode: Mode,
        model: Module,
        intput_data: Tensor | List[Tensor]
    ) -> Tensor | List[Tensor]:
        if mode is Mode.TRAIN:
            return model(intput_data)
        with no_grad():
            return model(intput_data)

    def Core(
        self,
        epoch: int,
        mode: Mode,
        model: Module,
        output: Tensor | List[Tensor],
        target: Tensor | List[Tensor],
        optim: Optimizer,
        data_count: int,
        **kwarg
    ) -> int:
        raise NotImplementedError

    def Save_weight(
        self,
        model: Module,
        optim: Optimizer,
        scheduler: LRScheduler | None,
        file_name: str,
        file_dir: List[str] | None = None
    ):
        _scheduler = None if scheduler is None else scheduler.state_dict()
        _file_dir = [] if file_dir is None else file_dir
        _file_dir += [f"{file_name}.h5"]

        save(
            {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scheduler": _scheduler
            },
            Path.Join(_file_dir, self.result_root)
        )

    def Load_weight(
        self,
        model: Module,
        optim: Optimizer,
        scheduler: LRScheduler | None,
        file_name: str,
        file_dir: List[str] | None = None
    ):
        _file_dir = [] if file_dir is None else file_dir
        _file_dir += [f"{file_name}.h5"]
        _state_dict = load(Path.Join(_file_dir, self.result_root))

        model.load_state_dict(_state_dict["model"])
        optim.load_state_dict(_state_dict["optim"])
        if scheduler is not None:
            scheduler.load_state_dict(_state_dict["scheduler"])

        return model, optim, scheduler

    def Decision_for_learning(
        self,
        epoch: int,
        model: Module,
        optim: Optimizer,
        scheduler: LRScheduler | None
    ):
        raise NotImplementedError

    def Main_work(self, gpu_num: int, config: Dict[str, Any]):
        _device = device(
            f"cuda:{self.gpus[gpu_num]}" if len(self.gpus) else "cpu"
        )

        # set holder for debug and decision
        self.Set_holder(config["debug"])

        # dataset and dataloader
        _dataloader_config: Dict[Mode, Dict[str, Any]] = config["dataloader"]
        _loaders = dict((
            _mode, self.Set_dataloader(
                self.Set_dataset(_mode, config["dataset"]), _config
            )) for _mode, _config in _dataloader_config.items()
        )
        # model and optimizer (with scheduler)
        _model = self.Set_model(_device, config["model"])
        _optim, _scheduler = self.Set_optimizer(_model, config["optimizer"])

        # load pretrained weight
        if self.last_epoch:
            _model, _optim, _scheduler = self.Load_weight(
                _model, _optim, _scheduler, "last"
            )

        # set loss
        self.Set_loss(config["loss"])

        for _epoch in range(self.last_epoch, self.max_epoch):
            for _mode in self.apply_mode:
                _dataloader = _loaders[_mode]
                _data_ct = 0

                for _data in _dataloader:
                    _in, _target, _ = self.Set_intput_n_target(_device, _data)
                    _out = self.Get_output(_mode, _model, _in)

                    _data_ct = self.Core(
                        _epoch, _mode, _model, _out, _target, _optim, _data_ct
                    )

            if _scheduler is not None:
                _scheduler.step()

            # make decision for learning in each epoch
            self.Decision_for_learning(_epoch, _model, _optim, _scheduler)

    def Run(self, config: Dict[str, Any]):
        self.Main_work(0, config)  # not use distribut0
