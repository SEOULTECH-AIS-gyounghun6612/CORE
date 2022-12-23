from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from torch import cuda
from python_ex._base import Directory, Utils, JSON_WRITEABLE


if __package__ == "":
    # if this file in local project
    from torch_ex._torch_base import Learning_Mode, Process_Type
    from torch_ex._trainer import Multi_Method
else:
    # if this file in package folder
    from ._torch_base import Learning_Mode, Process_Type
    from ._trainer import Multi_Method
    ...


# -- DEFINE CONFIG -- #


# -- Mation Function -- #
class Learning_Config:
    @dataclass
    class E2E(Utils.Config):
        # Infomation about learning
        _Project_Name: str = "End_to_End_learning"
        _Detail: str = "Empty"
        _Date: str = Utils.Time._apply_text_form(Utils.Time._stemp(), True, "%Y-%m-%d")
        _Save_root: str = Directory._relative_root()

        # About Learning type and style
        _Batch_size_in_node: int = 4
        _Max_num_workers: int = 2

        _Max_epochs: int = 100
        _Last_epoch: int = -1
        _Learning_list: List[Learning_Mode] = field(default_factory=lambda: [Learning_Mode.TRAIN, Learning_Mode.VALIDATION])


@dataclass
class Hardware_Config(Utils.Config):
    # about use GPU
    _GPU_list: List[int] = field(default_factory=lambda: list(range(cuda.device_count())))

    # about multi process
    _Multi_protocol: Multi_Method = Multi_Method.NONE

    # about distribute
    _Num_of_global_node: int = 1
    _This_computer_rank: int = 0
    _Host_address: str = "tcp://127.0.0.1:10001"

    def _get_parameter(self) -> Dict[str, Any]:
        return {

        }

    def _convert_to_dict(self) -> Dict[str, JSON_WRITEABLE]:
        return super()._convert_to_dict()


@dataclass
class Tracker_Config(Utils.Config):
    # Logging parameter in each mode;
    _Tracking_pram: Dict[Learning_Mode, Dict[Process_Type, List[str]]]  # str -> logging_loss name

    # Observe parameter in each mode;
    # if None -> all same like logging keys
    _Observing: Dict[Learning_Mode, Dict[Process_Type, Optional[List[str]]]] = field(default_factory=dict)  # str -> logging_loss name

    def _get_parameter(self):
        return {
            "tracking_param": self._Tracking_pram,
            "observing_param": self._Observing}

    def _convert_to_dict(self):
        return {
            "_Tracking": dict((
                _mode_key.value,
                dict((
                    _process_key.value,
                    dict((_name, None) for _name in _name_info)
                ) for _process_key, _name_info in _process_info.items())
            ) for _mode_key, _process_info in self._Tracking_pram.items()),
            "_Observing": dict((
                _mode_key.value,
                dict((
                    _process_key.value,
                    None if _name_info is None else dict((_name, None) for _name in _name_info)
                ) for _process_key, _name_info in _process_info.items())
            ) for _mode_key, _process_info in self._Observing.items())}


@dataclass
class Augment_Config(Utils.Config):
    ...


@dataclass
class Scheduler_Config(Utils.Config):
    _Optim_name: Suport_Optimizer
    _Schedule_name: Suport_Schedule

    _LR_Maximum: float = 0.005
    _LR_Minimum: float = 0.0001
    _LR_Decay: float = 1.0

    _Term: Union[List[int], int] = 50
    _Term_amp: float = 1.0

    def _get_parameter(self, model: Custom_Model) -> Dict[str, Any]:
        return {
            "optimizer": optim.__dict__[self._Optim_name.value](model.parameters(), self._LR_Maximum),
            "schedule_name": self._Schedule_name,
            "term": self._Term,
            "term_amp": self._Term_amp,
            "maximum": self._LR_Maximum,
            "minimum": self._LR_Minimum,
            "decay": self._LR_Decay}

    def _convert_to_dict(self) -> Dict[str, Union[Dict, str, int, float, bool, None]]:
        return {
            "_Optim_name": self._Optim_name.value,
            "_Schedule_name": self._Schedule_name.value,
            "_LR_Maximum": self._LR_Maximum,
            "_LR_Minimum": self._LR_Minimum,
            "_LR_Decay": self._LR_Decay,
            "_Term": self._Term,
            "_Term_amp": self._Term_amp}


def _file_to_config(file_dir: str):
    ...
