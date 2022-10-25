from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple, Union
from math import log10, floor

from torch import Tensor, tensor, stack, clip, distributions, cat

from python_ex._base import Directory, Utils
from python_ex._label import Learning_Mode
from python_ex._result import Log
from python_ex._numpy import np_base, np_dtype, ndarray, evaluation


# -- DEFINE CONSTNAT -- #
class Logging_Factor(Enum):
    LOSS = 0
    ACC = 1
    IoU = 2
    mIoU = 3


# -- DEFINE CONFIG -- #
@dataclass
class Log_Config(Utils.Config):
    _Save_root: str = Directory._relative_root()
    _File: str = "learning_log.json"
    _Tracking: Dict[Learning_Mode, List[Logging_Factor]] = field(default_factory=dict)     # Dict[Learning_mode: str, Logging_parameter: List[str]]
    _Debugging: Dict[Learning_Mode, List[Logging_Factor]] = field(default_factory=dict)  # Dict[Learning_mode: str, Debugging_paramerter: List[str]]

    def _convert_to_dict(self) -> Dict[str, Any]:
        return super()._convert_to_dict()

    def _restore_from_dict(self, data: Dict[str, Any]):
        return super()._restore_from_dict(data)

    def _make_data_holder(self):
        __holder = {}
        for __Learning_mode in self._Tracking.keys():
            __holder[__Learning_mode.value] = {}
            for parameter in self._Tracking[__Learning_mode]:
                __holder[__Learning_mode.value][parameter.value] = []
        return __holder


# -- Mation Function -- #
class Torch_Utils():
    class Directory():
        @staticmethod
        def _make_result_diretory(root: str = None, object_dir: str = None):
            _obj_dir = f"result{Directory.SLASH}"
            _obj_dir += f"{Utils.Time_stemp(is_text=True)}{Directory.SLASH}" if object_dir is None else Directory._slash_check(object_dir)

            return Directory._make(_obj_dir, root)

    class Tensor():
        @staticmethod
        def _holder(sample, is_shape: bool = False, value: int = 0, dtype: type = np_dtype.np_float32):
            _array = np_base.get_array_from(sample, is_shape, value, dtype)
            return Torch_Utils.Tensor._from_numpy(_array, dtype)

        @classmethod
        def _from_numpy(self, np_array: ndarray, dtype: type = np_dtype.np_float32):
            return tensor(np_array)

        @staticmethod
        def _to_numpy(tensor: Tensor, dtype: type = np_dtype.np_float32) -> ndarray:
            try:
                _array = tensor.numpy()
            except RuntimeError:
                _array = tensor.detach().numpy()

            return np_base.type_converter(_array, dtype)

        @staticmethod
        def _make_tensor(size, shape_sample=None, norm_option=None, dtype="uint8", value=[0, 1]):
            _value = np_base.get_random_array(size, value, norm_option, dtype)
            return Torch_Utils.Tensor._from_numpy(_value)

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

    class Layer():
        @staticmethod
        def _position_encoding():
            ...

        @staticmethod
        def _get_conv_pad(kernel_size, input_size, interval=1, stride=1):
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
        def _concatenate(layers, dim=1):
            tmp_layer = layers[0]
            for _layer in layers[1:]:
                tmp_layer = cat([tmp_layer, _layer], dim=dim)

            return tmp_layer

    class _evaluation():
        @staticmethod
        def iou(result: Tensor, label: Tensor, ingnore_class: List[int]) -> ndarray:
            _batch_size, _class_num = result.shape[0: 2]
            np_result = Torch_Utils.Tensor._to_numpy(result.cpu().detach()).argmax(axis=1)  # [batch_size, h, w]
            np_label = Torch_Utils.Tensor._to_numpy(label.cpu().detach()).argmax(axis=1)  # [batch_size, h, w]

            iou = np_base.get_array_from([_batch_size, _class_num], True, dtype=np_dtype.np_float32)

            for _b in range(_batch_size):
                iou[_b] = evaluation.iou(np_result[_b], np_label[_b], _class_num, ingnore_class)

            return iou

        @staticmethod
        def miou(result: Tensor, label: Tensor, ingnore_class: List[int]) -> Tuple[ndarray]:
            _batch_size, _class_num = result.shape[0: 2]
            np_result = Torch_Utils.Tensor._to_numpy(result.cpu().detach()).argmax(axis=1)  # [batch_size, h, w]
            np_label = Torch_Utils.Tensor._to_numpy(label.cpu().detach()).argmax(axis=1)  # [batch_size, h, w]

            iou = np_base.get_array_from([_batch_size, _class_num], True, dtype=np_dtype.np_float32)
            miou = np_base.get_array_from([_batch_size, ], True, dtype=np_dtype.np_float32)

            for _b in range(_batch_size):
                iou[_b], miou[_b] = evaluation.miou(np_result[_b], np_label[_b], _class_num, ingnore_class)

            return iou, miou


class Debug():
    @staticmethod
    def _make_result_directory(project_name: str = None, root_dir: str = None):
        __root = Directory._relative_root() if root_dir is None else root_dir
        __project_folder = Utils.Time_stemp(True) if project_name is None else project_name

        return Directory._make(f"{__project_folder}/", __root)

    class Learning_Log(Log):
        def __init__(self, config: Log_Config):
            self._Debugging_factor = config._Debugging

            __file_name = config._File
            __save_dir = config._Save_root

            super().__init__(data=config._make_data_holder(), save_dir=__save_dir, file_name=__file_name)

        # Freeze function
        def _set_activate_mode(self, learing_mode: Learning_Mode):
            self._Active_mode = learing_mode

        def _learning_tracking(self, tracking_data: Dict[Logging_Factor, Union[int, float, List]]):
            __mode = self._Active_mode.value
            __tracking = {}

            for __factor in tracking_data.keys():
                __tracking[__factor.value] = tracking_data[__factor]

            self._insert({__mode: __tracking}, access_point=self._Data, is_overwrite=False)

        def _learning_debug(self, batch_size: int, data_count: int):
            __data_st = data_count // batch_size + int(data_count % batch_size)
            __debugging_string = ""

            for _debugging in self._Data[self._Active_mode.value].keys():
                if Logging_Factor(_debugging) in self._Debugging:
                    __value = sum(self._Data[self._Active_mode.value][_debugging][-__data_st:])
                    __value = __value / data_count
                    __debugging_string += "{_debugging}: {__value:>7.3f}, "

            return __debugging_string[:-1]

        def _progress_bar(self, epoch_info: List[int], data_info: List[int], decimals: int = 1, length: int = 25, fill: str = '█'):
            def _make_count_string(this_count, max_value):  # [3/25] -> [03/25]
                _string_ct = floor(log10(max_value)) + 1
                _this = f"{this_count}".rjust(_string_ct, " ")
                _max = f"{max_value}".rjust(_string_ct, " ")

                return f"{_this}/{_max}"

            __epoch = epoch_info[0]
            __max_epoch = epoch_info[1]

            __data_count = data_info[0]
            __max_data_count = data_info[1]
            __batch_size = data_info[2]

            _prefix = f"{self._Active_mode.value:>10} {_make_count_string(__epoch, __max_epoch)} {_make_count_string(__data_count, __max_data_count)}"
            _suffix = self._learning_debug(__batch_size, __data_count)

            length = length if len(_prefix) + len(_suffix) + 20 <= length else len(_prefix) + len(_suffix) + 20

            Utils.Progress_Bar(__data_count, __max_data_count, _prefix, _suffix, decimals, length, fill)
