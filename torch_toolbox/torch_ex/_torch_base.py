from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

from torch import Tensor, tensor, stack, clip, distributions, cat

from python_ex._base import Directory, Utils
from python_ex._label import Learning_Mode
from python_ex._result import Log
from python_ex._numpy import Array_Process, Dtype, ndarray, Evaluation_Process


# -- DEFINE CONSTNAT -- #
MAIN_RANK: int = 0


# -- DEFINE CONFIG -- #
@dataclass
class Log_Config(Utils.Config):
    # Logging parameter in each mode;
    _Loss_logging: Dict[Learning_Mode, List[str]] = field(default_factory=dict)  # str -> logging_loss name
    _Acc_logging: Dict[Learning_Mode, List[str]] = field(default_factory=dict)  # str -> logging_acc name

    # Tracking parameter in each mode;
    # if None -> all same like logging keys
    _Loss_tracking: Dict[Learning_Mode, List[str]] = field(default_factory=dict)  # str -> logging_loss name
    _Acc_tracking: Dict[Learning_Mode, List[str]] = field(default_factory=dict)  # str -> logging_acc name

    def _convert_to_dict(self) -> Dict[str, Any]:
        _dict = {
            "_Loss_logging": dict((_key.value, _value) for (_key, _value) in self._Loss_logging.items()),
            "_Acc_logging": dict((_key.value, _value) for (_key, _value) in self._Acc_logging.items()),
            "_Loss_tracking": dict((_key.value, _value) for (_key, _value) in self._Loss_tracking.items()),
            "_Acc_tracking": dict((_key.value, _value) for (_key, _value) in self._Acc_tracking.items()),
        }

        return _dict

    def _restore_from_dict(self, data: Dict[str, Any]):
        self._Loss_logging = dict((Learning_Mode(_key), _value) for (_key, _value) in data["_Loss_logging"].items())
        self._Acc_logging = dict((Learning_Mode(_key), _value) for (_key, _value) in data["_Acc_logging"].items())

        self._Loss_tracking = dict((Learning_Mode(_key), _value) for (_key, _value) in data["_Loss_tracking"].items())
        self._Acc_tracking = dict((Learning_Mode(_key), _value) for (_key, _value) in data["_Acc_tracking"].items())


# -- Mation Function -- #
class Torch_Utils():
    class Directory():
        @staticmethod
        def _make_diretory(directory: str, root: str = None, this_rank: int = 0):
            _maked = Directory._make(directory, root) if this_rank is MAIN_RANK else f"{root}{Directory._Divider}{directory}"
            return _maked

    class Tensor():
        @staticmethod
        def _holder(sample, is_shape: bool = False, value: int = 0, dtype: type = Dtype.NP_FLOAT32):
            _array = Array_Process._get_array_from(sample, is_shape, value, dtype)
            return Torch_Utils.Tensor._from_numpy(_array, dtype)

        @classmethod
        def _from_numpy(self, np_array: ndarray, dtype: type = Dtype.NP_FLOAT32):
            return tensor(np_array)

        @staticmethod
        def _to_numpy(tensor: Tensor, dtype: type = Dtype.NP_FLOAT32) -> ndarray:
            try:
                _array = tensor.numpy()
            except RuntimeError:
                _array = tensor.detach().numpy()

            return Array_Process.type_converter(_array, dtype)

        @staticmethod
        def _make_tensor(size, shape_sample=None, norm_option=None, dtype="uint8", value=[0, 1]):
            _value = Array_Process.get_random_array(size, value, norm_option, dtype)
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
            np_label = Torch_Utils.Tensor._to_numpy(label.cpu().detach())  # [batch_size, h, w]

            iou = Array_Process._get_array_from([_batch_size, _class_num], True, dtype=Dtype.NP_FLOAT32)

            for _b in range(_batch_size):
                iou[_b] = Evaluation_Process._iou(np_result[_b], np_label[_b], _class_num, ingnore_class)

            return iou.tolist()

        @staticmethod
        def miou(result: Tensor, label: Tensor, ingnore_class: List[int]) -> Tuple[ndarray]:
            _batch_size, _class_num = result.shape[0: 2]
            np_result = Torch_Utils.Tensor._to_numpy(result.cpu().detach()).argmax(axis=1)  # [batch_size, h, w]
            np_label = Torch_Utils.Tensor._to_numpy(label.cpu().detach())  # [batch_size, h, w]

            iou = Array_Process._get_array_from([_batch_size, _class_num], True, dtype=Dtype.NP_FLOAT32)
            miou = Array_Process._get_array_from([_batch_size, ], True, dtype=Dtype.NP_FLOAT32)

            for _b in range(_batch_size):
                iou[_b], miou[_b] = Evaluation_Process._miou(np_result[_b], np_label[_b], _class_num, ingnore_class)

            return iou.tolist(), miou.tolist()


class Debug():
    class Learning_Log(Log):
        def _log_init(self, config: Log_Config):
            self._Loss_tracking = config._Loss_tracking if len(config._Loss_tracking.keys()) else config._Loss_logging
            self._Acc_tracking = config._Acc_tracking if len(config._Acc_tracking.keys()) else config._Acc_logging

            self._insert(data=self._make_data_holder(config._Loss_logging, config._Acc_logging), access_point=self._Data)

        # Freeze function
        def _make_data_holder(self, loss_logging: Dict[Learning_Mode, List[str]], acc_logging: Dict[Learning_Mode, List[str]]):
            _holder = {}

            # loss
            for _mode in loss_logging.keys():
                if _mode.value not in _holder:
                    _holder[_mode.value] = {}

                _holder[_mode.value]["loss"] = {}
                for _loss_name in loss_logging[_mode]:
                    _holder[_mode.value]["loss"][_loss_name] = {}

            # acc
            for _mode in acc_logging.keys():
                if _mode.value not in _holder:
                    _holder[_mode.value] = {}

                _holder[_mode.value]["acc"] = {}
                for _acc_name in acc_logging[_mode]:
                    _holder[_mode.value]["acc"][_acc_name] = {}

            # time
            for _mode in _holder.keys():
                _holder[_mode]["process_time"] = {}
            return _holder

        def _set_activate_mode(self, learing_mode: Learning_Mode):
            self._Active_mode = learing_mode

        def _learning_logging(self, epoch: int, loss: Dict[str, Union[int, float, List]] = None, acc: Dict[str, Union[int, float, List]] = None, process_time: float = None):
            _tracking = {}

            if loss is not None:
                _tracking["loss"] = {}
                for _factor in loss.keys():
                    _tracking["loss"][_factor] = {epoch: loss[_factor]}

            if acc is not None:
                _tracking["acc"] = {}
                for _factor in acc.keys():
                    _tracking["acc"][_factor] = {epoch: acc[_factor]}

            if process_time is not None:
                _tracking["process_time"] = {epoch: process_time}

            self._insert({self._Active_mode.value: _tracking}, access_point=self._Data, is_overwrite=False)

        def _learning_tracking(self, epoch: int):
            def _make_string(data: Union[int, float, List[Union[List, int, float]]]):
                if isinstance(data, int):
                    return f"{data:>4d}, "

                elif isinstance(data, float):
                    return f"{data:>7.3f}, "

                # data => List[Union[List, int, float]]
                _data = Array_Process._get_array_from(data, dtype=Dtype.NP_FLOAT32)
                if isinstance(data[0], (int, float)):
                    return f"{_data.mean():>7.3f}, "

                else:
                    _data: ndarray = _data.mean(0)
                    _list_string = "["
                    for _ct in range(_data.shape[0]):
                        _list_string += f"{_data[_ct]:>7.3f}, "
                    return f"{_list_string[:-2]}], "

            _debugging_string = ""

            _logging_mode = self._Active_mode
            _loss_tracking = self._Loss_tracking[_logging_mode]
            _acc_tracking = self._Acc_tracking[_logging_mode]

            _picked_data = self._get_data(
                place={_logging_mode.value: {
                    "loss": {_data_name: epoch for _data_name in _loss_tracking},
                    "acc": {_data_name: epoch for _data_name in _acc_tracking}
                }},
                access_point=self._Data)

            # loss
            for _loss_keys in _loss_tracking:
                _data_string = _make_string(_picked_data[_logging_mode.value]["loss"][_loss_keys])
                _debugging_string += f"{_loss_keys}: {_data_string}"

            # acc
            for _acc_keys in _acc_tracking:
                _data_string = _make_string(_picked_data[_logging_mode.value]["acc"][_acc_keys])
                _debugging_string += f"{_acc_keys}: {_data_string}"

            return _debugging_string[:-2] if len(_debugging_string) else _debugging_string

        def _progress_length(self, epoch: int):
            _logging_mode = self._Active_mode
            _loss_tracking = self._Loss_tracking[_logging_mode]
            _picked_data = self._get_data_length(
                place={
                    _logging_mode.value: {
                        "loss": {_loss_tracking[0]: epoch}
                    }},
                access_point=self._Data)[_logging_mode.value]["loss"][_loss_tracking[0]]

            return len(_picked_data)

        def _get_using_time(self, epoch: int, is_average: bool = False):
            _time_list = self._Data[self._Active_mode.value]["process_time"][epoch]
            if isinstance(_time_list, list):
                return sum(_time_list) / len(_time_list) if is_average else sum(_time_list)
            else:
                return _time_list

        def _get_learning_time(self, epoch: int, max_batch_count: int):
            _sum_time = self._get_using_time(epoch)
            _maximun_time = self._get_using_time(epoch, True) * max_batch_count
            return _sum_time, _maximun_time
