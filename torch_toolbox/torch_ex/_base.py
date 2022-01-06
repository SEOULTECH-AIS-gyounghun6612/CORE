from dataclasses import dataclass
from typing import Dict, List
from numpy import ndarray

import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import Dataset

from python_ex import _numpy
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
        Batch_size: int
        Data_root: str
        Data_size: List[int]
        Dataset: Dataset
        Num_worker: int

    class _learning():
        @dataclass
        class base():
            Modes: List[str]
            Max_epochs: int

            # DEFAULTED VALUE
            Learning_rate: float = 0.001
            is_cuda: bool = torch.cuda.is_available()

        @dataclass
        class reinforcement():
            Modes: List[str]
            Max_epochs: int

            Action_size: List[int]
            Memory_size: int
            Discount: float
            Max_step: int

            # DEFAULTED VALUE
            Learning_rate: float = 0.001
            is_cuda: bool = torch.cuda.is_available()


class torch_utils():
    class _tensor():
        torch_type = {
            "uint8": torch.uint8, "int32": torch.int32, "bool": torch.bool, "float32": torch.float32}

        @staticmethod
        def holder(sample, is_shape=False, value=0, dtype="float32"):
            _array = _numpy.base.get_array_from(sample, is_shape, value, dtype)
            return torch_utils._tensor.from_numpy(_array, dtype)

        @classmethod
        def from_numpy(self, np_array, dtype="float32"):
            return torch.tensor(np_array, dtype=self.torch_type[dtype])

        @staticmethod
        def to_numpy(tensor: torch.Tensor, type=None) -> _numpy.np.ndarray:
            try:
                return tensor.numpy()
            except RuntimeError:
                return tensor.detach().numpy()

        @staticmethod
        def make_tensor(size, shape_sample=None, norm_option=None, dtype="uint8", value=[0, 1]):
            # in later fix it
            _np_array = _numpy.base.get_array(size, shape_sample, norm_option, dtype, value)
            return torch_utils._tensor.from_numpy(_np_array, dtype)

        @staticmethod
        def range_cut(tensor: torch.Tensor, range_min, rage_max):
            return torch.clip(tensor, range_min, rage_max)


class toolbox():
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


class loss():
    @staticmethod
    def mse(output, target) -> torch.Tensor:
        """
        Args:
            output: [batch, c, h, w]
            target: [batch, c, h, w]
        Return:
            loss
        """
        return MSELoss()(output, target)

    @staticmethod
    def cross_entropy(output, target, ignore_index=-100):
        """
        Args:
            output: [batch, class_num, h, w]
            target: [batch, h, w]
        Return:
            loss
        """
        return CrossEntropyLoss(ignore_index=ignore_index)(output, target)

    @staticmethod
    def mean(output, target):
        return torch.mean(output * target)


class evaluation():
    @staticmethod
    def iou(result: torch.Tensor, label: torch.Tensor, class_num) -> torch.Tensor:
        np_result = result.cpu().detach().numpy()
        np_label = label.cpu().detach().numpy()

        iou = _numpy.evaluation.iou(np_result, np_label, class_num)
        return iou

    @staticmethod
    def miou(result, label, class_num):
        iou = evaluation.iou(result, label, class_num)
        return iou.mean()


class log(_numpy.log):
    log_data: Dict[str, Dict[str, ndarray]] = {}

    def __init__(self, opt: opt._log) -> None:
        self.opt = opt
        super(log, self).__init__(opt.get_log_tree())

    def save(self, file_name="log.json"):
        return super().save(self.opt.Save_root, file_name=file_name)

    def get_log_display(self, learning_mode):
        _log_data = self.log_data[learning_mode]
        log_display = ""
        for _param in self.opt.Display_list[learning_mode]:
            if _param in _log_data.keys():
                log_display += _param + f":{_log_data[_param][-1]:>6.4f} "

        return log_display
