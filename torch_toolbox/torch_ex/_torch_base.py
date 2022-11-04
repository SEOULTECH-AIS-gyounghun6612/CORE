from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

from torch import Tensor, tensor, stack, clip, distributions, cat

from python_ex._base import Directory, Utils
from python_ex._label import Learning_Mode
from python_ex._result import Log
from python_ex._numpy import np_base, np_dtype, ndarray, evaluation


# -- DEFINE CONSTNAT -- #


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
        def _make_result_diretory(root: str = None, object_dir: str = None):
            _obj_dir = f"result{Directory.SLASH}"
            _obj_dir += f"{Utils._time_stemp(is_text=True)}{Directory.SLASH}" if object_dir is None else Directory._slash_check(object_dir)

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

            return iou.sum(0).tolist()

        @staticmethod
        def miou(result: Tensor, label: Tensor, ingnore_class: List[int]) -> Tuple[ndarray]:
            _batch_size, _class_num = result.shape[0: 2]
            np_result = Torch_Utils.Tensor._to_numpy(result.cpu().detach()).argmax(axis=1)  # [batch_size, h, w]
            np_label = Torch_Utils.Tensor._to_numpy(label.cpu().detach()).argmax(axis=1)  # [batch_size, h, w]

            iou = np_base.get_array_from([_batch_size, _class_num], True, dtype=np_dtype.np_float32)
            miou = np_base.get_array_from([_batch_size, ], True, dtype=np_dtype.np_float32)

            for _b in range(_batch_size):
                iou[_b], miou[_b] = evaluation.miou(np_result[_b], np_label[_b], _class_num, ingnore_class)

            return iou.sum(0).tolist(), miou.sum(0).tolist()


class Debug():
    @staticmethod
    def _make_result_directory(project_name: str, root_dir: str = None):
        _root = Directory._relative_root() if root_dir is None else root_dir
        return Directory._make(f"{project_name}/", _root)

    class Learning_Log(Log):
        def __init__(self, config: Log_Config):
            loss_logging = config._Loss_logging
            acc_logging = config._Acc_logging

            self._Loss_tracking = config._Loss_tracking if len(config._Loss_tracking.keys()) else config._Loss_logging
            self._Acc_tracking = config._Acc_tracking if len(config._Acc_tracking.keys()) else config._Acc_logging

            super().__init__(data=self._make_data_holder(loss_logging, acc_logging))

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

        def _learning_tracking(self, epoch: int, data_count: int):
            def _make_string(data: Union[int, float, List], count: int):
                _data_nd: ndarray = np_base.get_array_from(data, dtype=np_dtype.np_float32).sum(0) / count
                if _data_nd.ndim:
                    _list_string = "["
                    for _ct in range(_data_nd.shape[0]):
                        _list_string += f"{_data_nd[_ct]:>7.3f}, "
                    return f"{_list_string[:-2]}]"
                else:
                    return f"{_data_nd:>7.3f}, "

            _debugging_string = ""

            # loss
            for _loss_keys in self._Loss_tracking[self._Active_mode]:
                _data_list = self._Data[self._Active_mode.value]["loss"][_loss_keys][epoch]
                _data_string = _make_string(_data_list, data_count)
                _debugging_string += f"{_loss_keys}: {_data_string}"

            # acc
            for _acc_keys in self._Acc_tracking[self._Active_mode]:
                _data_list = self._Data[self._Active_mode.value]["acc"][_acc_keys][epoch]
                _data_string = _make_string(_data_list, data_count)
                _debugging_string += f"{_acc_keys}: {_data_string}"

            return _debugging_string[:-2] if len(_debugging_string) else _debugging_string

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
