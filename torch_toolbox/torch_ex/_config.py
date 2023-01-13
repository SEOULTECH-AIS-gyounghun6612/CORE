from typing import List, Dict, Any, Tuple, Optional, Type, Union
from dataclasses import dataclass, field

from math import pi, cos, sin, ceil

from torch import cuda
from python_ex._base import Directory, File, Utils, JSON_WRITEABLE


if __package__ == "":
    # if this file in local project
    from torch_ex._torch_base import Learning_Mode, Parameter_Type
    from torch_ex._process import Multi_Method
    from torch_ex._label import Support_Label, Label_Style, File_Style, Label
    from torch_ex._dataset import Supported_Transform, Supported_Augment, Augment
    from torch_ex._optimizer import Suport_Optimizer, Suport_Schedule
else:
    # if this file in package folder
    from ._torch_base import Learning_Mode, Parameter_Type
    from ._process import Multi_Method
    from ._label import Support_Label, Label_Style, File_Style, Label
    from ._dataset import Supported_Transform, Supported_Augment, Augment
    from ._optimizer import Suport_Optimizer, Suport_Schedule


# -- Mation Function -- #
class Learning_Config:
    @dataclass
    class E2E(Utils.Config):
        # Infomation about learning
        project_name: str = "End_to_End_learning"
        description: str = "Empty"

        # _date: str = Utils.Time._apply_text_form(Utils.Time._stemp(), True, "%Y-%m-%d")
        save_root: str = Directory._relative_root()

        # About Learning type and style
        max_epoch: int = 100
        last_epoch: int = -1
        learning_mode: List[str] = field(default_factory=lambda: [Learning_Mode.TRAIN.value, Learning_Mode.VALIDATION.value])

        batch_size_per_node: int = 4
        num_worker_per_node: int = 2

        gpu_ids: List[int] = field(default_factory=lambda: list(range(cuda.device_count())))

        world_size: int = 1
        this_rank: int = 1
        multi_method: str = Multi_Method.DDP.value
        multi_protocal: Optional[str] = "tcp://127.0.0.1:10001"

        def _get_parameter(self) -> Dict[str, Any]:
            _param = super()._get_parameter()

            _param["learning_mode"] = [Learning_Mode(_value) for _value in self.learning_mode]

            _param["multi_method"] = Multi_Method(_param["multi_method"])

            if _param["multi_method"] is Multi_Method.NONE:
                _param["world_size"] = 1
            else:
                if self.this_rank * cuda.device_count() > self.world_size:
                    _param["world_size"] = self.this_rank * cuda.device_count()

            return _param

    @staticmethod
    def _Build(meta_data: Dict[str, Any]) -> Utils.Config:
        _learing_type = meta_data["type"]
        _learing_parameter = meta_data["parameter"]

        return Learning_Config.__dict__[_learing_type](**_learing_parameter)


@dataclass
class Tracker_Config(Utils.Config):
    # Logging parameter in each mode;
    # Learning_Mode -> Parameter_Type -> obj_name
    tracking: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    # Observe parameter in each mode;
    # Learning_Mode -> Parameter_Type -> obj_name
    observing: Dict[str, Dict[str, Optional[List[str]]]] = field(default_factory=dict)

    def _get_parameter(self) -> Dict[str, Any]:
        return {
            "tracking_param": dict((
                Learning_Mode(_mode),
                dict((Parameter_Type(_process), _obj) for _process, _obj in _tracking_info.items())) for _mode, _tracking_info in self.tracking.items()),
            "observing_param": dict((
                Learning_Mode(_mode),
                dict((Parameter_Type(_process), _obj) for _process, _obj in _observing_info.items())) for _mode, _observing_info in self.observing.items())
        }


@dataclass
class Augment_Config(Utils.Config):
    data_size: List[int]
    rotate_limit: int = 0

    is_norm: bool = True
    norm_mean: Optional[List[float]] = None
    norm_std: Optional[List[float]] = None

    horizontal_flip_rate: float = 0.0
    vertical_flip_rate: float = 0.0

    # transform_list: Dict[str, Dict[str, Dict[str, JSON_WRITEABLE]]]  # Dict[Learning_Mode, Dict[Tr_name, Parameter dict]]

    agment_method: str = Supported_Augment.ALBUIMIENTATIONS.value
    agment_option: Optional[Dict[str, JSON_WRITEABLE]] = None

    def _get_parameter(self):
        _tr_dict = {}
        if self.rotate_limit:
            _rad = pi * self.rotate_limit / 180
            _h = ceil(self.data_size[1] * sin(_rad) + self.data_size[0] * cos(_rad))
            _w = ceil(self.data_size[0] * sin(_rad) + self.data_size[1] * cos(_rad))
            _tr_dict.update({"Resize": {"size": [_h, _w] + self.data_size[2:]}})
            _tr_dict.update({"Rotate": {"angle_limit": self.rotate_limit}})
            _tr_dict.update({"Random_Crop": {"size": self.data_size}})
        else:
            _tr_dict.update({"Resize": {"size": self.data_size}})

        if self.horizontal_flip_rate or self.vertical_flip_rate:
            _tr_dict.update({"Random_Flip": {"horizontal_flip_rate": self.horizontal_flip_rate, "vertical_flip_rate": self.vertical_flip_rate}})

        if self.is_norm:
            _param = {}
            _param.update({"mean": self.norm_mean}) if self.norm_mean is not None else ...
            _param.update({"std": self.norm_std}) if self.norm_std is not None else ...
            _tr_dict.update({"Normalization": _param})
        _tr_dict.update({"To_Tenor": {}})

        _transform_list = [Supported_Transform.__dict__[_tr_name](**_tr_param) for _tr_name, _tr_param in _tr_dict.items()]

        return Augment.__dict__[self.agment_method](_transform_list, **self.agment_option)


@dataclass
class Dataset_Config(Utils.Config):
    label_name: str  # Support_Label
    label_style: List[str]  # List[Label_Style]

    file_info: Dict[str, List[Tuple[str, str, Optional[Any]]]]  # Dict[Learning_Mode, ]

    amplification: Dict[str, int]  # Dict[Learning_Mode, int]

    augment: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]

    meta_file: Optional[str] = None
    data_root: str = "./data/"

    def _get_parameter(self) -> Dict[str, Any]:
        return {
            "label_process": Label.Process.__dict__[self.label_name](
                Support_Label(self.label_name),
                [Label_Style(_style_name) for _style_name in self.label_style],
                self.meta_file),
            "file_profiles": Label.File_IO.__dict__[self.label_name](
                dict((
                    Learning_Mode(_mode_name),
                    [(Label_Style(_label_style), File_Style(_file_style), _optional) for _label_style, _file_style, _optional in _data_list]
                ) for _mode_name, _data_list in self.file_info.items()),
                self.data_root
            ),
            "amplification": dict((
                Learning_Mode(_mode_name),
                _value
            )for _mode_name, _value in self.amplification.items()),
            "augmentation": dict((
                Learning_Mode(_mode_name),
                [Augment_Config(**_aug)._get_parameter() for _aug in _aug_param] if isinstance(_aug_param, list) else Augment_Config(**_aug_param)._get_parameter()
            ) for _mode_name, _aug_param in self.augment.items())
        }


@dataclass
class Optim_n_Schedule_Config(Utils.Config):
    optim_name: str  # Suport_Optimizer
    schedule_name: str  # Suport_Schedule

    initial_lr: float = 0.0005
    maximum_lr: float = 0.0001
    minimum_lr: float = 0.00001
    lr_decay: float = 0.99

    schedule_term: int = 10
    schedule_term_amp: int = 1

    def _get_parameter(self) -> Dict[str, Any]:
        return {
            "optim_name": Suport_Optimizer(self.optim_name),
            "initial_lr": self.initial_lr,

            "schedule_name": Suport_Schedule(self.schedule_name),
            "term": self.schedule_term,
            "term_amp": self.schedule_term_amp,
            "maximum": self.maximum_lr,
            "minimum": self.minimum_lr,
            "decay": self.lr_decay
        }


class Config():
    def __init__(self, file_name: str, file_dir: str = "./config/"):
        self._Load(file_name, file_dir)

    # Freeze function
    def _Save(self, file_name: str, file_dir: str = "./config/"):
        _, _file_name = File._extension_check(file_name, ["json"], True)
        _file_dir = Directory._divider_check(file_dir)

        File._json(_file_dir, _file_name, True, self._config)

    def _Load(self, file_name: str, file_dir: str = "./config/"):
        _, _file_name = File._extension_check(file_name, ["json"], True)
        _file_dir = Directory._divider_check(file_dir)

        _config_file = f"{_file_dir}{_file_name}"

        if File._exist_check(_config_file):
            # load config
            self._config: Dict[str, Any] = File._json(_file_dir, _file_name)
        else:
            self._config = {}
            print(f"config file {_config_file} not exsit. \nyou must config data initialize before use it")

    def _Get_learning_config(self):
        return Learning_Config._Build(self._config["learning"])._get_parameter()

    def _Set_learning_config(self, learning_config: Learning_Config.E2E):
        self._config.update({"learning": {
            "type": learning_config.__class__.__name__,
            "parameter": learning_config._convert_to_dict()
        }})

    def _Get_tracker_config(self):
        return Tracker_Config(**self._config["tracker"])._get_parameter()

    def _Set_tracker_config(self, tracker_config: Tracker_Config):
        self._config.update({"tracker": tracker_config._convert_to_dict()})

    def _Get_dataset_config(self):
        return Dataset_Config(**self._config["dataset"])._get_parameter()

    def _Set_dataset_config(self, dataset_config: Dataset_Config):
        self._config.update({"dataset": dataset_config._convert_to_dict()})

    def _Get_optim_n_shedule_config(self):
        return Optim_n_Schedule_Config(**self._config["optim_n_shedule"])._get_parameter()

    def _Set_optim_n_shedule_config(self, optim_n_shedule_config: Optim_n_Schedule_Config):
        self._config.update({"optim_n_shedule": optim_n_shedule_config._convert_to_dict()})

    def _Set_model_structure_config(self, model_config: Utils.Config):
        self._config.update({"model_config": {
            "type": model_config.__class__.__name__,
            "parameter": model_config._convert_to_dict()
        }})

    def _Get_model_structure_config(self, model_config: Type[Utils.Config]):
        _model_type = self._config["model_config"]["type"]

        if model_config.__name__ == _model_type:
            return model_config(**self._config["model_config"]["parameter"])._get_parameter()
        else:
            raise ValueError(
                f"The information in the config file does not match the data in the config structure.\n\
                  config file: {_model_type} != input config: {model_config.__name__}\n\
                  Please check it again.")
