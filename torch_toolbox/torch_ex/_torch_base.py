from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union
from math import log10, floor

from torch import Tensor, cuda, tensor, stack, clip, distributions, cat
from torch.nn import Module
from torch.optim import Optimizer, Adam

from python_ex._numpy import np_base, np_dtype, ndarray, evaluation
from python_ex._result import log, logging_option
from python_ex._base import directory, utils
from python_ex._label import Label_process, Label_style, File_style

# from python_ex import _error as _e
from enum import Enum


class learing_mode(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class opt():
    @dataclass
    class _dataloader():
        # dataloader setting
        Batch_size: int
        Num_workers: int

        # dataset setting
        Data_process: Label_process.label_basement
        Data_label_style: Label_style
        Data_file_style: File_style

    class _learning():
        @dataclass
        class E2E():
            # Infomation about learning
            Learing_name: str
            Learning_date: str
            Learning_detail: str

            # About Learning type and style
            Learning_mode: List[learing_mode]
            Max_epochs: int
            This_epoch: int

            # Debugging parameter for learning
            Save_root: str

            # About Optimizer
            LR_initail: List[float] = field(default_factory=lambda: [0.005, ])
            LR_discount: List[float] = field(default_factory=lambda: [0.1, ])

            # About logging
            Log_file: str = "learning_log.json"
            Logging_parameters: Dict[learing_mode, List[str]] = field(default_factory=dict)     # Dict[Learning_mode: str, Logging_parameter: List[str]]
            Display_paramerters: Dict[learing_mode, List[str]] = field(default_factory=dict)  # Dict[Learning_mode: str, Debugging_paramerter: List[str]]

            # About GPU system
            Use_cuda: bool = cuda.is_available()

            def make_save_directorty(self):
                return debug.make_result_directory(self.Learing_name, self.Save_root)

            def opt_to_file_data(self):
                ...

        @dataclass
        class reinforcement(E2E):
            # reinforcement train option
            Max_step: int = 100

            Q_discount: float = 0.99

            Reward_threshold: List[float] = field(default_factory=list)
            Reward_value: List[float] = field(default_factory=list)
            Reward_fail: float = -10
            # Reward_relation_range: int = 80

            # action option
            Action_size: List[Union[int, List[int]]] = field(default_factory=list)
            Action_range: List[List[int]] = field(default_factory=list)  # [[Max, Min]]

            # replay option
            Memory_size: int = 1000
            Minimum_memroy_size: int = 100
            Exploration_threshold: int = 1.0
            Exploration_discount: float = 0.99
            Exploration_Minimum: int = 1.0

    class _layer_opt():
        @dataclass
        class backbone():
            type: int

            is_pretrained: bool = True
            is_trainable: bool = False

            use_flat: bool = False
            use_avg_pooling: bool = False

        @dataclass
        class fc():
            in_features: int
            out_features: int
            bias: bool = True

        @dataclass
        class conv2d():
            in_channels: int
            out_channels: int
            kernel_size: int
            stride: int = 1
            padding: int = 0
            dilation: int = 1
            groups: int = 1
            bias: bool = True

        @dataclass
        class attention():
            attention_type: str
            input_dim: int
            output_dim: int
            head_count: int

            def __init__(self, attention_type: str, input_dim: int, output_dim: int, num_head: int):
                self.attention_type = attention_type
                self.input_dim = input_dim
                self.head_count = num_head
                self.output_dim = output_dim + (output_dim % self.head_count) if (output_dim % num_head) else output_dim

            def get_head_dim(self):
                return self.output_dim // self.head_count

        @dataclass
        class postion_encoding():
            encoding_type: str
            max_len: int

            def __init__(self, attention_type: str, input_dim: int, output_dim: int, num_head: int):
                ...

        @dataclass
        class norm2d():
            norm_type: str

            num_features: int
            eps: float = 1e-5
            momentum: float = 0.1
            affine: bool = True
            track_running_stats: bool = True

            def to_parameters(self):
                if self.norm_type == "BatchNorm":
                    return {"num_features": self.num_features, "eps": self.eps, "momentum": self.momentum, "affine": self.affine, "track_running_stats": self.track_running_stats}
                else:
                    return {}

        @dataclass
        class active_function():
            active_type: str

            # ReLU - basement
            inplace: bool = True

            # LeakyReLU
            negative_slope: float = 0.01

            # Tanh, Sigmoid
            # empty

            def to_parameters(self):
                if self.active_type == "ReLU":
                    return {"inplace": self.inplace}
                elif self.active_type == "LeakyReLU":
                    return {"inplace": self.inplace, "negative_slope": self.negative_slope}
                else:  # Tanh, Sigmoid
                    return {}

    @dataclass
    class _optim_opt():
        optimize_type: str

        def make(self, model: Module, Learning_rate: float) -> Optimizer:
            if self.optimize_type == "Adam":
                return Adam(model.parameters(), Learning_rate)


class torch_utils():
    class _directory():
        @staticmethod
        def make_result_diretory(root: str = None, object_dir: str = None):
            _obj_dir = f"result{directory.SLASH}"
            _obj_dir += f"{utils.time_stemp(is_text=True)}{directory.SLASH}" if object_dir is None else directory._slash_check(object_dir)

            return directory._make(_obj_dir, root)

    class _tensor():
        @staticmethod
        def holder(sample, is_shape: bool = False, value: int = 0, dtype: type = np_dtype.np_float32):
            _array = np_base.get_array_from(sample, is_shape, value, dtype)
            return torch_utils._tensor.from_numpy(_array, dtype)

        @classmethod
        def from_numpy(self, np_array: ndarray, dtype: type = np_dtype.np_float32):
            return tensor(np_base.type_converter(np_array, dtype))

        @staticmethod
        def to_numpy(tensor: Tensor, dtype: type = np_dtype.np_float32) -> ndarray:
            try:
                _array = tensor.numpy()
            except RuntimeError:
                _array = tensor.detach().numpy()

            return np_base.type_converter(_array, dtype)

        @staticmethod
        def make_tensor(size, shape_sample=None, norm_option=None, dtype="uint8", value=[0, 1]):
            # in later
            pass

        @staticmethod
        def _range_cut(tensor: Tensor, range_min, rage_max):
            return clip(tensor, range_min, rage_max)

        @staticmethod
        def _norm(mu, std):
            return distributions.Normal(mu, std)

        @staticmethod
        def _stack(tensor_list: Tuple[Tensor], dim: int):
            return stack(tensor_list, dim=dim)

        @staticmethod
        def make_partition(tensor: Tensor, shape: List[int]):
            # _t_shpae = tensor.shape
            pass

    class _layer():
        @staticmethod
        def position_encoding():
            ...

        @staticmethod
        def get_conv_pad(kernel_size, input_size, interval=1, stride=1):
            if type(kernel_size) != list:
                kernel_size = [kernel_size, kernel_size]

            if stride != 1:
                size_h = input_size[0]
                size_w = input_size[1]

                pad_hs = (stride - 1) * (size_h - 1) + interval * (kernel_size[0] - 1)
                pad_ws = (stride - 1) * (size_w - 1) + interval * (kernel_size[1] - 1)
            else:
                pad_hs = interval * (kernel_size[0] - 1)
                pad_ws = interval * (kernel_size[1] - 1)

            pad_l = pad_hs // 2
            pad_t = pad_ws // 2

            return [pad_t, pad_ws - pad_t, pad_l, pad_hs - pad_l]

        @staticmethod
        def concatenate(layers, dim=1):
            tmp_layer = layers[0]
            for _layer in layers[1:]:
                tmp_layer = cat([tmp_layer, _layer], dim=dim)

            return tmp_layer

    class _evaluation():
        @staticmethod
        def iou(result: Tensor, label: Tensor, class_num: int) -> ndarray:
            np_result = result.cpu().detach().numpy()
            np_label = label.cpu().detach().numpy()

            iou = evaluation.iou(np_result, np_label, class_num)
            return iou

        @staticmethod
        def miou(result: Tensor, label: Tensor, class_num: int) -> ndarray:
            iou = torch_utils._evaluation.iou(result, label, class_num)
            return iou, iou.mean()


class debug():
    @staticmethod
    def make_result_directory(try_name: str = None, root_dir: str = None):
        _root = directory._relative_root() if root_dir is None else root_dir
        _date = utils.time_stemp(True) if try_name is None else try_name

        return directory._make(f"result/{_date}/", _root)

    class process_log(log):
        def __init__(self, logging_parameter: Dict[str, List[str]], save_dir: str, file_name: str, is_restore: bool = False):
            holder = {}

            if not is_restore:
                for _Learning_mode in logging_parameter.keys():
                    holder[_Learning_mode] = {}
                    for parameter in logging_parameter[_Learning_mode]:
                        holder[_Learning_mode][parameter] = []

            super().__init__(data=holder, save_dir=save_dir, file_name=file_name, is_resotre=is_restore)

        def set_logging_mode(self, learing_mode: learing_mode):
            self.active_mode = learing_mode

        def annotation_update(self, name: str, data: Any, is_overwrite: bool = False):
            _flag = name in self.annotation.keys()

            if is_overwrite and _flag:
                self.add({name: data}, logging_option.OVERWRITE)

            else:
                self.add({name: data}, logging_option.ADD)

        def update(self, data: Dict):
            self.add(data, save_point=self.data[self.active_mode])

        def get_debugging_string(self, epoch: int, data_count: int, max_data_count: int) -> str:
            _data_st = max_data_count * epoch
            _data_ed = max_data_count * epoch + data_count

            _data = self.get_data(self.annotation["Debugging_paramerters"][self.active_mode], self.data[self.active_mode], _data_st, _data_ed)

            _debugging_string = ""

            for _param in self.annotation["Debugging_paramerters"][self.active_mode]:
                if _param in _data.keys():
                    _debugging_string += f"{_param} {sum(_data[_param]) / data_count},"
            return _debugging_string

        def file_data_to_opts(self) -> Dict[str, Any]:
            for _opt_key in self.annotation.keys():
                ...
            ...

        def progress_bar(self, epoch: int, data_count: int, decimals: int = 1, length: int = 25, fill: str = '█'):
            def make_count_string(this_count, max_value):  # [3/25] -> [03/25]
                _string_ct = floor(log10(max_value)) + 1
                _this = f"{this_count}".rjust(_string_ct, " ")
                _max = f"{max_value}".rjust(_string_ct, " ")

                return f"{_this}/{_max}"

            _max_epoch = self.info["Max_epochs"]
            _max_in_dataloader = self.info["Dataloader"][f"{self.active_mode}_max_length"]

            _prefix = f"{self.active_mode:<10} {make_count_string(epoch, _max_epoch)} {make_count_string(data_count, _max_in_dataloader)}"
            _suffix = self.get_debugging_string(epoch, data_count, _max_in_dataloader)

            length = length if len(_prefix) + len(_suffix) + 20 <= length else len(_prefix) + len(_suffix) + 20

            utils.Progress_Bar(data_count, _max_in_dataloader, _prefix, _suffix, decimals, length, fill)
