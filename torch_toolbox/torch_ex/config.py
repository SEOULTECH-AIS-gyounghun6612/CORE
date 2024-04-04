from typing import Dict, List, Any
from dataclasses import dataclass

from python_ex.system import Path
from python_ex.project import Config

from .learning import Mode


@dataclass
class LearningConfig(Config):
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

    def Get_learning_parameter(self) -> Dict[str, Any]:
        return {
            "project_name": self.porject_name,
            "description": Path.Join(self.category, self.dataset),
            "max_epoch": self.max_epoch,
            "last_epoch": self.last_epoch,
            "apply_mode": [Mode(_name) for _name in self.apply_mode],
            "gpus": self.gpus
        }

    def Get_dataset_parameter(self):
        raise NotImplementedError

    def Get_dataloader_parameter(self, learning_mode: str) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "shuffle": learning_mode == Mode.TRAIN.value,
            "drop_last": self.drop_last[learning_mode]
        }

    def Get_model_parameter(self) -> Dict[str, Any]:
        return {
            "model_parmeter": {
            }
        }

    def Get_optimmizer_parameter(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
        }

    def Get_loss_parameter(self) -> Dict[str, Any]:
        raise NotImplementedError

    def Get_debug_parameter(self) -> Dict[str, Any]:
        return {
            "holder_name": self.eval_parameters
        }

    def Config_to_parameter(self, ) -> Dict[str, Any]:
        return {
            "trainer": self.Get_learning_parameter(),
            "dataset": self.Get_dataset_parameter(),
            "dataloader": dict((
                Mode(_name), self.Get_dataloader_parameter(_name)
            ) for _name in self.apply_mode),
            "optimizer": self.Get_optimmizer_parameter(),
            "model": self.Get_model_parameter(),
            "loss": self.Get_loss_parameter(),
            "debug": self.Get_debug_parameter(),
        }
