from typing import List, Dict, Any, Tuple, Type
from types import ModuleType
from dataclasses import dataclass, field

from python_ex._Base import Directory, File
from python_ex._Project import Debuging, Config


if __package__ == "":
    # if this file in local project
    from torch_ex._Base import Process_Name
    from torch_ex._Layer import Custom_Model
    from torch_ex._Learning import Multi_Method
    from torch_ex._Dataset import Supported_Augment, Custom_Dataset_Process, Data_Organization, Augment, Label_Style, Data_Format
    from torch_ex._optimizer import Suport_Optimizer, Suport_Schedule
else:
    # if this file in package folder
    from ._Base import Process_Name
    from ._Layer import Custom_Model
    from ._Learning import Multi_Method
    from ._Dataset import Supported_Augment, Custom_Dataset_Process, Data_Organization, Augment, Label_Style, Data_Format
    from ._optimizer import Suport_Optimizer, Suport_Schedule


# -- Main code -- #
class Learning_Config:
    @dataclass
    class E2E(Config):
        # Infomation about learning
        project_name: str
        description: str

        # _date: str = Utils.Time._apply_text_form(Utils.Time._stemp(), True, "%Y-%m-%d")
        save_root: str

        # About Learning type and style
        mode_list: List[str]
        max_epoch: int
        last_epoch: int

        multi_method: str
        world_size: int
        this_rank: int
        usable_gpus: List[Tuple[int, str]]
        multi_protocal: str | None

        def _Get_learning_parameter(self) -> Dict[str, Any]:
            return {
                "project_name": self.project_name,
                "description": self.description,
                "save_root": self.save_root,
                "mode_list": [Process_Name(_value) for _value in self.mode_list],
                "max_epoch": self.max_epoch,
                "last_epoch": self.last_epoch
            }

        def _Get_processer_parameter(self) -> Dict[str, Any]:
            world_size = self.world_size if self.world_size >= len(self.usable_gpus) else len(self.usable_gpus)

            return {
                "multi_method": Multi_Method(self.multi_method),
                "world_size": world_size,
                "this_rank": self.this_rank,
                "usable_gpus": self.usable_gpus,
                "multi_protocal": self.multi_protocal
            }

    @dataclass
    class Reinforcement(E2E):
        max_step: int
        exploration_rate: float
        exploration_discont: float
        exploration_minimum: float

        reward_value_list: List[float]

        reward_milestone: List[float]
        reward_model: Dict[str, Any] | None
        reward_model_save_dir: str | None

        learning_range: int
        reward_discount: float
        memory_size: int
        memory_threshold: int

        def _Get_reinforcement_parameter(self):
            return {
                "max_step": self.max_step,
                "exploration_rate": self.exploration_rate,
                "exploration_discont": self.exploration_discont,
                "exploration_minimum": self.exploration_minimum,
                "learning_range": self.learning_range,
                "reward_discount": self.reward_discount,
                "memory_size": self.memory_size,
                "memory_threshold": self.memory_threshold
            }

        def _Get_reward_model_parameter(self):
            return {

            }

    @staticmethod
    def _Build(type: str, config_data: Dict[str, Any]) -> Config:
        return Learning_Config.__dict__[type](**config_data)


@dataclass
class Augment_Config(Config):
    """
    ### 데이터 증폭을 위한 변형 설정

    -------------------------------------------------------------------------------------------
    ## Parameters
        output_size (List[int])
            :
        rotate_limit: (int)
            :
        is_norm: (bool)
            :
        norm_mean: (List[float] | None)
            :
        norm_std: (List[float] | None)
            :

        horizontal_flip_rate (defalt: 0.0)
            :
        vertical_flip_rate (defalt: 0.0)
            :

        agment_method
            :
        agment_option
            :
    -------------------------------------------------------------------------------------------
    """
    output_size: List[int]
    rotate_limit: int = 0

    horizontal_flip_rate: float = 0.0
    vertical_flip_rate: float = 0.0

    is_norm: bool = True
    norm_mean: List[float] = [0.485, 0.456, 0.406]
    norm_std: List[float] = [0.229, 0.224, 0.225]

    apply_to_tensor: bool = False

    # transform_list: Dict[str, Dict[str, Dict[str, JSON_WRITEABLE]]]  # Dict[Learning_Mode, Dict[Tr_name, Parameter dict]]

    agment_method: str = Supported_Augment.ALBUIMIENTATIONS.value
    agment_option: Dict[str, JSON_WRITEABLE] | None = field(default_factory=lambda: {"bbox_parameter": None, "keypoints_parameter": None, "group_parmaeter": None})

    def _Get_parameter(self) -> Dict[str, Any]:
        _parameter = super()._Get_parameter()
        _parameter["agment_method"] = Supported_Augment(self.agment_method)

        return _parameter


@dataclass
class Dataset_Config(Utils.Config):
    mode_list: List[str]
    data_organization: str | None
    data_info: List[Tuple[str, str]]  # List[(Label.Style, Data.Format)]

    amplification: Dict[str, int]  # Dict[Learning_Mode, int]
    augment: Dict[str, List[Dict[str, Any]]]

    data_root: str = "./data/"
    label_meta_file: str | None = None
    batch_size_per_node: int = 4
    num_worker_per_node: int = 4

    def _Get_parameter(self) -> Dict[str, Any]:
        _data_parameters = {
            "root_dir": self.data_root,
            "activate_mode": self.mode_list,
            "data_info": [(Label_Style(_label_style), Data_Format(data_format)) for _label_style, data_format in self.data_info],
            "label_meta_file": self.label_meta_file,
        }
        return {
            "data_organization": Data_Organization._Build(self.data_organization, **_data_parameters) if self.data_organization is not None else _data_parameters,
            "amplification": dict((Process_Name(_mode_name), _value)for _mode_name, _value in self.amplification.items()),
            "augmentation": dict(
                (Process_Name(_mode_name), [Augment._Build(**Augment_Config(**_aug)._Get_parameter()) for _aug in _aug_param]) for _mode_name, _aug_param in self.augment.items()),
            "batch_size_per_node": self.batch_size_per_node,
            "num_worker_per_node": self.num_worker_per_node
        }


@dataclass
class Tracker_Config(Utils.Config):
    # Logging parameter in each mode;
    # Learning_Mode -> Parameter_Type -> obj_name
    tracking: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    # Observe parameter in each mode;
    # Learning_Mode -> Parameter_Type -> obj_name
    observing: Dict[str, Dict[str, List[str] | None]] = field(default_factory=dict)

    def _Get_parameter(self) -> Dict[str, Any]:
        return {
            "tracking_param": dict((
                Process_Name(_mode),
                dict((Scale_Type(_process), _obj) for _process, _obj in _tracking_info.items())) for _mode, _tracking_info in self.tracking.items()),
            "observing_param": dict((
                Process_Name(_mode),
                dict((Scale_Type(_process), _obj) for _process, _obj in _observing_info.items())) for _mode, _observing_info in self.observing.items())
        }


@dataclass
class Optim_n_Schedule_Config(Utils.Config):
    optim_name: str  # Suport_Optimizer
    schedule_name: str  # Suport_Schedule

    initial_lr: float = 0.0001
    maximum_lr: float = 0.0001
    minimum_lr: float = 0.00001
    lr_decay: float = 0.99

    schedule_term: int = 10
    schedule_term_amp: int = 1

    def _Get_parameter(self) -> Dict[str, Any]:
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


@dataclass
class Custom_Model_Config(Utils.Config):
    model_name: str


class Config():
    def __init__(self, file_name: str, file_dir: str = "./config/"):
        self._Load(file_name, file_dir)

    # Freeze function
    def _Save(self, file_name: str, file_dir: str = "./config/"):
        _, _file_name = File._Extension_check(file_name, ["json"], True)
        _file_dir = Directory._Divider_check(file_dir)

        File._Json(_file_dir, _file_name, True, self._config)

    def _Load(self, file_name: str, file_dir: str = "./config/"):
        _, _file_name = File._Extension_check(file_name, ["json"], True)
        _file_dir = Directory._Divider_check(file_dir)

        _config_file = f"{_file_dir}{_file_name}"

        if File._Exist_check(_config_file):
            # load config
            self._config: Dict[str, Any] = File._Json(_file_dir, _file_name)
        else:
            self._config = {}
            print(f"config file {_config_file} not exsit. \nyou must config data initialize before use it")

    def _Set_learning_option(self, learning_config: Learning_Config.E2E):
        self._config.update({"learning_type": learning_config.__class__.__name__})
        self._config.update(learning_config._Convert_to_dict())

    def _Set_data_option(self, dataset_config: Dataset_Config):
        self._config.update(dataset_config._Convert_to_dict())

    def _Get_learning_option(self):
        _learning_type = self._config["learning_type"]
        _learning_config = Learning_Config.__dict__[_learning_type]

        assert isinstance(_learning_config, (Learning_Config.E2E, Learning_Config.Reinforcement))

        # _meta_parameter = dict((_key, _data) for _key, _data in self._config.items() if _key in _learning_config.__dict__.keys())
        # return Learning_Config._Build(_learning_type, _meta_parameter)._Get_parameter()
        ...

    def _Get_data_option(self, data_organization: Type[Data_Organization.Basement] | None = None, custom_dataset: Type[Custom_Dataset_Process] | None = None):
        _opt_in_config = dict((_key, _data) for _key, _data in self._config.items() if _key in Dataset_Config.__dict__.keys())
        _parameter = Dataset_Config(**_opt_in_config)._Get_parameter()

        _organization: Data_Organization.Basement | Dict[str, Any] = _parameter["data_organization"]
        _augmentation_opt: Dict[Process_Name, List[Dict[str, Any]]] = _parameter["augmentation"]

        if isinstance(_organization, dict):
            if data_organization is not None:
                _organization = data_organization(**_organization)
            else:
                assert False  # incollect parameter for data organization when use set dataset config

        _dataset_opt = {
            "organization": _organization,
            "amplification": _parameter["amplification"],
            "augmentations": dict((
                _learning_process, [Augment._Build(group_parmaeter=_organization._Get_data_clusturing(), **_each_param) for _each_param in _params]
            ) for _learning_process, _params in _augmentation_opt.items())
        }

        return {
            "dataset_process": Custom_Dataset_Process(**_dataset_opt) if custom_dataset is None else custom_dataset(**_dataset_opt),
            "batch_size_per_node": _parameter["batch_size_per_node"],
            "num_worker_per_node": _parameter["num_worker_per_node"]
        }

    def _Get_tracker_config(self):
        return Tracker_Config(**self._config["tracker"])._Get_parameter()

    def _Set_tracker_config(self, tracker_config: Tracker_Config):
        self._config.update({"tracker": tracker_config._Convert_to_dict()})

    def _Get_optim_n_shedule_config(self):
        return Optim_n_Schedule_Config(**self._config["optim_n_shedule"])._Get_parameter()

    def _Set_optim_n_shedule_config(self, optim_n_shedule_config: Optim_n_Schedule_Config):
        self._config.update({"optim_n_shedule": optim_n_shedule_config._Convert_to_dict()})

    def _Get_model_structure_config(self, source: ModuleType) -> Tuple[Type[Custom_Model], Dict]:
        _model_config_name = self._config["model_config"]["name"]
        _model_config_parma = self._config["model_config"]["parameter"]
        if _model_config_name in source.__dict__.keys():
            _model_config: Custom_Model_Config = source.__dict__[_model_config_name](**_model_config_parma)
            if _model_config.model_name in source.__dict__.keys():
                return source.__dict__[_model_config.model_name], _model_config._Get_parameter()
            else:
                raise ValueError(
                    f"The information in the config file have fatal error.\n\
                    -> Model {self._config['model_config']['parameter']['model_name']} is not exist in {source.__name__}.\n\
                    Please check it again.")
        else:
            raise ValueError(
                f"The information in the config file have fatal error.\n\
                -> Config {self._config['model_config']['name']} is not exist in {source.__name__}.\n\
                Please check it again.")

    def _Set_model_structure_config(self, model_config: Custom_Model_Config):
        self._config.update({"model_config": {
            "name": model_config.__class__.__name__,
            "parameter": model_config._Convert_to_dict()
        }})
