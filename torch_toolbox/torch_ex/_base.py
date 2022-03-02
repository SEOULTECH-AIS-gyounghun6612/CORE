from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union
from math import log10, floor

from torch import Tensor, cuda, tensor, clip, stack, distributions, cat
from torch.nn import Module, ModuleList

from python_ex._numpy import np_base, ndarray, evaluation, labeled_holder
from python_ex._base import utils
from python_ex._label import process, style

# from python_ex import _error as _e


class opt():
    @dataclass
    class _log():
        Label: Dict[str, Union[Dict, type]]
        Debug: Dict[str, Union[Dict, type]]
        Display: Dict[str, Union[Dict, str, list]]

    @dataclass
    class _data():
        # info
        Batch_size: int
        Num_workers: int
        File_process: process.basement
        Data_style: style.basement
        Data_size: List[int]

    class _learning():
        @dataclass
        class base():
            Max_epochs: int
            Learnig_style: List[str]

            # DEFAULTED VALUE
            Learning_rate: Union[float, List[float]] = 0.001
            is_cuda: bool = cuda.is_available()

        @dataclass
        class reinforcement(base):
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


class torch_utils():
    class _tensor():
        @staticmethod
        def holder(sample, is_shape: bool = False, value: int = 0, dtype: type = np_base.np_dtype.np_float32):
            _array = np_base.get_array_from(sample, is_shape, value, dtype)
            return torch_utils._tensor.from_numpy(_array, dtype)

        @classmethod
        def from_numpy(self, np_array: ndarray, dtype: type = np_base.np_dtype.np_float32):
            return tensor(np_base.type_converter(np_array, dtype))

        @staticmethod
        def to_numpy(tensor: Tensor, dtype: type = np_base.np_dtype.np_float32) -> ndarray:
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

    class _layer():
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

        @staticmethod
        def make_block_list(layers: List[Module]):
            return ModuleList(layers)

    class _evaluation():
        @staticmethod
        def iou(result: Tensor, label: Tensor, class_num) -> Tensor:
            np_result = result.cpu().detach().numpy()
            np_label = label.cpu().detach().numpy()

            iou = evaluation.iou(np_result, np_label, class_num)
            return iou

        @staticmethod
        def miou(result, label, class_num):
            iou = evaluation.iou(result, label, class_num)
            return iou.mean()


class log(labeled_holder):
    def __init__(self, opt: opt._log, info: Dict[str, Any]) -> None:
        label = opt.Label
        super().__init__(label, info)
        self.debug_parameter = opt.Debug
        self.display_parameter = opt.Display

    def save(self, save_dir, file_name="log.json"):
        return super().save(save_dir, file_name=file_name)

    def get_log(self, learning_mode, target, key) -> ndarray:
        return self.data[learning_mode][target][key]

    def get_display_string(self, learning_mode, call_num):
        _display = self.display_parameter[learning_mode]
        _string = ""
        for _target in _display.keys():
            for _key in _display[_target]:
                save_log = self.get_log(learning_mode, _target, _key)[call_num].item()
                _string += f"{_key}: "
                _string += f"{save_log:.3f} " if isinstance(save_log, float) else f"{save_log:3d} "

        return _string

    def get_debug_string(self, learning_mode, last_data_count):
        _debug = self.debug_parameter[learning_mode]
        _string = ""

        # make debug string
        for _target in _debug.keys():
            for _key in _debug[_target]:
                _string += _key + f": {self.get_log(learning_mode, _target, _key)[-last_data_count:].mean(): .3f} "

        return _string

    def progress_bar(self, epoch, learning_mode, data_num, this_data_num, decimals=1, length=25, fill='█'):
        _max_epoch = self.info["Train_opt"]["Max_epochs"]
        _e_num_ct = floor(log10(_max_epoch)) + 1
        _epoch = f"{epoch}".rjust(_e_num_ct, " ")
        _epochs = f"{_max_epoch}".rjust(_e_num_ct, " ")

        _max_data_length = self.info["Dataloader"][learning_mode]["data_length"]
        _d_num_ct = floor(log10(_max_data_length)) + 1
        _data_num = f"{data_num}".rjust(_d_num_ct, " ")
        _data_size = f"{_max_data_length}".rjust(_d_num_ct, " ")

        _prefix = f"{learning_mode:<10}{_epoch}/{_epochs} {_data_num}/{_data_size}"

        utils.Progress_Bar(data_num, _max_data_length, _prefix, self.get_debug_string(learning_mode, this_data_num), decimals, length, fill)

        return data_num + this_data_num
