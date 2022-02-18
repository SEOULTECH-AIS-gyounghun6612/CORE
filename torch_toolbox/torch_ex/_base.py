from dataclasses import dataclass, field
from typing import Dict, List
from math import log10, floor

from torch import Tensor, cuda, tensor, clip

from python_ex import _numpy
from python_ex._base import utils
from python_ex._label import process, style

# from python_ex import _error as _e


class opt():
    @dataclass
    class _log():
        Logging_list: Dict[str, List[str]]
        Display_list: Dict[str, List[str]]
        Save_root: str

        def get_log_tree(self):
            train_log = {key: {} for key in self.Logging_list["train"]} if self.Logging_list["train"] is not None else {}
            validation_log = {key: {} for key in self.Logging_list["validation"]} if self.Logging_list["validation"] is not None else {}
            test_log = {key: {} for key in self.Logging_list["test"]} if self.Logging_list["test"] is not None else {}
            return {"train": train_log, "validation": validation_log, "test": test_log}

    @dataclass
    class _data():
        # info
        Batch_size: int
        Num_workers: int
        Data_read_process: process.basement
        Data_style: style.basement
        Data_size: List[str] = field(default_factory=list)

    class _learning():
        @dataclass
        class base():
            Max_epochs: int
            Train_style: List[str] = field(default_factory=list)

            # DEFAULTED VALUE
            Learning_rate: float = 0.001
            is_cuda: bool = cuda.is_available()

        @dataclass
        class reinforcement(base):
            Action_size: List[int] = field(default_factory=list)
            Memory_size: int = 1000
            Discount: float = 0.99
            Max_step: int = 100

            Reward_th: List[float] = field(default_factory=list)
            Reward_value: List[float] = field(default_factory=list)
            Reward_relation_range: int = 80


class torch_utils():
    class _tensor():
        @staticmethod
        def holder(sample, is_shape=False, value=0, dtype="float32"):
            _array = _numpy.base.get_array_from(sample, is_shape, value, dtype)
            return torch_utils._tensor.from_numpy(_array, dtype)

        @classmethod
        def from_numpy(self, np_array, dtype="float32"):
            return tensor(_numpy.base.type_converter(np_array, dtype))

        @staticmethod
        def to_numpy(tensor: Tensor, dtype="float32") -> _numpy.np.ndarray:
            try:
                _array = tensor.numpy()
            except RuntimeError:
                _array = tensor.detach().numpy()

            return _numpy.base.type_converter(_array, dtype)

        @staticmethod
        def make_tensor(size, shape_sample=None, norm_option=None, dtype="uint8", value=[0, 1]):
            # in later
            pass

        @staticmethod
        def range_cut(tensor: Tensor, range_min, rage_max):
            return clip(tensor, range_min, rage_max)

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


class evaluation():
    @staticmethod
    def iou(result: Tensor, label: Tensor, class_num) -> Tensor:
        np_result = result.cpu().detach().numpy()
        np_label = label.cpu().detach().numpy()

        iou = _numpy.evaluation.iou(np_result, np_label, class_num)
        return iou

    @staticmethod
    def miou(result, label, class_num):
        iou = evaluation.iou(result, label, class_num)
        return iou.mean()


class log(_numpy.log):
    log_data: Dict[str, Dict[str, _numpy.np.ndarray]] = {}
    log_info: Dict[str, Dict] = {}

    def __init__(self, opt: opt._log) -> None:
        self.opt = opt
        super(log, self).__init__(opt.get_log_tree())

        # for def "progress_bar" and def "get_log_display"
        self.this_mode: str = "train"
        self.this_epoch: int = 0

    def save(self, file_name="log.json"):
        return super().save(self.opt.Save_root, file_name=file_name)

    def get_log(self, learning_mode, parameter, block="epoch"):
        _log_data = self.log_data[learning_mode]

        if parameter in _log_data.keys():
            if isinstance(block, str):  # set mode
                if block == "epoch":
                    _block_ct = len(_log_data[parameter]) % self.log_info["dataloader"][self.this_mode]["data_length"]
            elif isinstance(block, int):  # set length
                _block_ct = 1

            _value = _log_data[parameter][-_block_ct:].mean()
        else:
            _value = 0

        return _value

    def get_log_display(self, learning_mode=None, block="epoch"):
        learning_mode = self.this_mode if learning_mode is None else learning_mode
        log_display = ""
        for _param in self.opt.Display_list[learning_mode]:
            _value = self.get_log(learning_mode, _param, block)
            log_display += _param + (f": +{_value:>6.4f} " if _value >= 0 else f": {_value:>6.4f} ")

        return log_display

    def progress_bar(self, data_num, decimals=1, length=25, fill='█'):
        _max_epoch = self.log_info["learning"]["Max_epochs"]
        _e_num_ct = floor(log10(_max_epoch)) + 1
        _epoch = f"{self.this_epoch}".rjust(_e_num_ct, " ")
        _epochs = f"{_max_epoch}".rjust(_e_num_ct, " ")

        _max_data_length = self.log_info["dataloader"][self.this_mode]["data_length"]
        _d_num_ct = floor(log10(_max_data_length)) + 1
        _data_num = f"{data_num}".rjust(_d_num_ct, " ")
        _data_size = f"{_max_data_length}".rjust(_d_num_ct, " ")

        _prefix = f"{self.this_mode:<10}{_epoch}/{_epochs} {_data_num}/{_data_size}"

        utils.Progress_Bar(data_num, _max_data_length, _prefix, self.get_log_display(), decimals, length, fill)
