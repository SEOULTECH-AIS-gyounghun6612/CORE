from enum import Enum
from typing import Tuple, Union, Dict, List, Any
from dataclasses import dataclass

from torch import Tensor, load, save
from torch.autograd.grad_mode import no_grad
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from python_ex._System import Path
from python_ex._Project import Template, Config


class Mode(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


@dataclass
class TrainerConfig(Config):
    # trainer config
    porject_name: str

    gpus: List[int]

    max_epoch: int
    last_epoch: int

    apply_mode: List[str]

    # loss

    # dataset
    dataset: str

    # dataloader
    batch_size: int
    drop_last: Dict[str, bool]

    def Get_trainer_config(self) -> Dict[str, Any]:
        return {
            "project_name": self.porject_name,
            "description": self.dataset,
            "max_epoch": self.max_epoch,
            "last_epoch": self.last_epoch,
            "apply_mode": [Mode(_name) for _name in self.max_epoch],
            "gpus": self.gpus
        }

    def Get_dataloader_config(self, train_mode: Mode) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "shuffle": train_mode == Mode.TRAIN,
            "drop_last": self.drop_last
        }

    def Get_model_config(self) -> Dict[str, Any]:
        raise NotImplementedError

    def Get_optimmizer(self) -> Dict[str, Any]:
        raise NotImplementedError

    def Config_to_parameter(self, ) -> Dict[str, Any]:
        return {
            "trainer": self.Get_trainer_config(),
            "dataloader": dict((_mode, self.Get_dataloader_config(_mode)) for _mode in self.apply_mode),
            "model": self.Get_model_config(),
            "optimizer": self.Get_optimmizer()
        }


class LearningProcess(Template):
    def __init__(
        self, max_epoch: int, apply_mode: List[Mode], gpus: List[int],
        project_name: str, description: str | None = None, result_root: str | None = None, last_epoch: int = 0
    ):
        super().__init__(project_name, description, result_root)

        self.max_epoch = max_epoch
        self.last_epoch = last_epoch
        self.apply_mode = apply_mode

        self.gpus = gpus

    def Set_dataloader(self, dataloader_config: Dict[str, Any]) -> DataLoader:
        raise NotImplementedError

    def Set_model(self, gpu_id: int, model_config: Dict[str, Any]) -> Module:
        raise NotImplementedError

    def Set_optimizer(self, model: Module, optimizer_config: Dict[str, Any]) -> Tuple[Optimizer, _LRScheduler | None]:
        raise NotImplementedError

    def Set_loss(self, loss_name: List[str]):
        raise NotImplementedError

    def Set_gpu(self, gpu_id: int, data: Union[Tensor | List[Tensor]]) -> Tuple[Tensor | List[Tensor], Tensor | List[Tensor]]:
        raise NotImplementedError

    def Get_loss(
        self,
        epoch: int,
        mode: Mode,
        output: Union[Tensor | List[Tensor]],
        target: Union[Tensor | List[Tensor]],
        optim: Optimizer,
        **kwarg
    ):
        raise NotImplementedError

    def Get_output(mode: Mode, model: Module, intput_data: Union[Tensor | List[Tensor]], **kwarg) -> Tensor | List[Tensor]:
        if mode is Mode.TRAIN:
            return model(intput_data)

        else:
            with no_grad():
                return model(intput_data)

    def Save_weight(self, model: Module, optim: Optimizer, scheduler: _LRScheduler | None, file_name: str, file_dir: List[str] | None):
        save(
            {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scheduler": scheduler.state_dict()
            },
            Path.Join(f"{file_name}.h5" if file_dir is None else file_dir + [f"{file_name}.h5"], self.result_root)
        )

    def Load_weight(self, model: Module, optim: Optimizer, scheduler: _LRScheduler | None, file_name: str, file_dir: List[str] | None):
        _state_dict = load(Path.Join(f"{file_name}.h5" if file_dir is None else file_dir + [f"{file_name}.h5"], self.result_root))

        model.load_state_dict(_state_dict["model"])
        optim.load_state_dict(_state_dict["optim"])
        scheduler.load_state_dict(_state_dict["model"]) if scheduler is not None else ...

        return model, optim, scheduler

    def Core(self, gpu_num: int, config: Dict[str, Any]):
        _this_gpu_id = self.gpus[gpu_num]

        _model = self.Set_model(_this_gpu_id, config["model"])
        _optim, _scheduler = self.Set_optimizer(_model, config["optimizer"])

        if self.last_epoch:
            _model, _optim, _scheduler = self.Load_weight(_model, _optim, _scheduler, "last")

        _dataloaders = self.Set_dataloader(config["dataloader"])

        for _epoch in range(self.last_epoch, self.max_epoch):
            for _mode in self.apply_mode:
                _dataloader = _dataloaders[_mode]

                for _data in _dataloader:
                    _input, _target = self.Set_gpu(_this_gpu_id, _data)
                    _output: Tensor | List[Tensor] = self.Get_output(_mode, _model, _input)

                    self.Get_loss(_epoch, _mode, _output, _target, _optim)

            _scheduler.step()

    def Run(self, config: Dict[str, Any]):
        self.Core(0, config)  # not use distribute
