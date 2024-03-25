from typing import Dict, List, Any
from dataclasses import dataclass

from python_ex._System import Path
from python_ex._Project import Config

from .Trainer import Mode


@dataclass
class TrainerConfig(Config):
    # trainer config
    porject_name: str

    gpus: List[int]

    max_epoch: int
    last_epoch: int

    apply_mode: List[str]

    # dataset
    dataset: str
    category: str

    # dataloader
    batch_size: int
    drop_last: Dict[str, bool]

    # optimizer and scheduler
    learning_rate: float

    # model

    # loss

    # debug
    eval_parameters: List[str]

    def Get_trainer_config(self) -> Dict[str, Any]:
        return {
            "project_name": self.porject_name,
            "description": Path.Join(self.category, self.dataset),
            "max_epoch": self.max_epoch,
            "last_epoch": self.last_epoch,
            "apply_mode": [Mode(_name) for _name in self.apply_mode],
            "gpus": self.gpus
        }

    def Get_dataset_config(self):
        raise NotImplementedError

    def Get_dataloader_config(self, learning_mode: str) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "shuffle": learning_mode == Mode.TRAIN.value,
            "drop_last": self.drop_last[learning_mode]
        }

    def Get_model_config(self) -> Dict[str, Any]:
        raise NotImplementedError

    def Get_optimmizer_config(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
        }

    def Get_loss_config(self) -> Dict[str, Any]:
        raise NotImplementedError

    def Get_debug_config(self) -> Dict[str, Any]:
        return {
            "holder_name": self.eval_parameters
        }

    def Config_to_parameter(self, ) -> Dict[str, Any]:
        return {
            "trainer": self.Get_trainer_config(),
            "dataset": self.Get_dataset_config(),
            "dataloader": dict((Mode(_name), self.Get_dataloader_config(_name)) for _name in self.apply_mode),
            "optimizer": self.Get_optimmizer_config(),
            "model": self.Get_model_config(),
            "loss": self.Get_loss_config(),
            "debug": self.Get_debug_config(),
        }
