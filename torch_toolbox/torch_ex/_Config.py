from typing import List, Dict, Any, Tuple, Type
from types import ModuleType
from dataclasses import dataclass, field

from python_ex._Base import Directory, File
from python_ex._Project import Debuging, Config


# if this file in package folder
from ._Base import Process_Name
from ._Learning import Learning_Process, Multi_Method
from ._Model_n_Optim import Optim


# -- Main code -- #
class Learning_Config():
    class Base(Config):
        def __init__(self, file_name: str, file_dir: str = "./config/"):
            _, self._file_name = File._Extension_check(file_name, ["json"], True)
            self.file_dir = Directory._Divider_check(file_dir)

            self._options = self._Load()

        # Freeze function
        def _Set_default_option(self):
            return {
                # --- basement option --- #
                "project_name": Directory._Relative_root(True)[:-len(Directory._Divider)],
                "description": "",
                "result_root": "./runs/",
                "learning_plan": {
                    "train" : {
                        "amplification": 1,
                        "augmentations": {
                            "apply_mathod": "Albumentations",
                            "output_size": [256, 256],
                            # "rotate_limit": 0,
                            # "hflip_rate": 0.0,
                            # "vflip_rate": 0.0,
                            # "is_norm": True,
                            # "norm_mean": [0.485, 0.456, 0.406],
                            # "norm_std": [0.229, 0.224, 0.225],
                            # "apply_to_tensor": True,
                            # "group_parmaeter": None
                        }
                    },
                    "val" : {
                        "amplification": 1,
                        "augmentations": {
                            "apply_mathod": "Albumentations",
                            "output_size": [256, 256],
                            # "rotate_limit": 0,
                            # "hflip_rate": 0.0,
                            # "vflip_rate": 0.0,
                            # "is_norm": True,
                            # "norm_mean": [0.485, 0.456, 0.406],
                            # "norm_std": [0.229, 0.224, 0.225],
                            # "apply_to_tensor": True,
                            # "group_parmaeter": None
                        }
                    }
                },
                "max_epoch": 100,
                "last_epoch": -1,

                # --- dataloader option --- #
                # dataset
                "data_root": "./data/",
                "dataset_name": "",

                # dataloader
                "batch_size_per_node": 512,
                "num_worker_per_node":8,
                "display_term": 0.01,

                # --- scheduler option --- #
                "optim_name": "Adam",
                "scheduler_name": "Cosin_Annealing",
                # "term": float(_max_epoch / 10),
                "term_amp": 1,
                "maximum": 0.0001,
                "minimum": 0.00005,
                "decay": 1,  # -> ?

                # --- multi process option --- #
                "multi_method": "Auto",
                "world_size": 2,
                "device_rank": 0,
                "max_gpu_count": 2,
                "multi_protocal": "tcp://127.0.0.1:10001"  # local
            }

        def _Save(self, config_file: Dict[str, Any]):
            File._Json(self.file_dir, self._file_name, True, config_file)

        def _Load(self):
            if File._Exist_check(self.file_dir, self._file_name):
                # load config
                return File._Json(self.file_dir, self._file_name)
            else:  # load template
                print(f"config file {self.file_dir}{self._file_name} not exsit. \nyou must config data initialize before use it")
                return self._Set_default_option()

        def _Make_learning_process(self, Learning_process: Type[Learning_Process.Basement]):
            # Make learning process
            _learning_opt = dict((_key, self._options[_key]) for _key in ["project_name", "description", "result_root", "learning_plan", "max_epoch", "last_epoch"])
            _learning_opt.update({"mode_list": [Process_Name(_mode) for _mode in self._options["learning_plan"].keys()]})
            _learning_process = Learning_process(**_learning_opt)

            # set multi process 
            _processer_opt = dict((_key, self._options[_key]) for _key in ["world_size", "device_rank", "max_gpu_count", "multi_protocal"])
            _processer_opt.update({"multi_method": Multi_Method(self._options["multi_method"])})
            _learning_process._Set_processer_option(**_processer_opt)
            _learning_process._Set_model_n_optim(**self._Make_model_n_optim_option())
            _learning_process._Set_dataloader_option(**self._Make_dataloader_option())

            return _learning_process

        def _Make_model_n_optim_option(self) -> Dict[str, Any]:
            _scheduler_option = dict((_key, self._options[_key]) for _key in ["term_amp", "maximum", "minimum", "decay"])
            _scheduler_option.update({"term": float(self._options["max_epoch"] / 10)})

            return {
                "optim_name": Optim.Supported(self._options["optim_name"]),
                "initial_lr": self._options["maximum"],
                "scheduler_name": Optim.Scheduler.Supported.Cosin_Annealing,
                "scheduler_option": _scheduler_option
            }

        def _Make_dataloader_option(self) -> Dict[str, Any]:
            _dataloader_option = dict((_key, self._options[_key]) for _key in ["batch_size_per_node", "num_worker_per_node", "display_term"])
            return _dataloader_option