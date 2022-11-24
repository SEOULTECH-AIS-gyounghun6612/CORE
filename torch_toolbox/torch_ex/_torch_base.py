from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
from enum import Enum

from torch import Tensor, tensor, stack, clip, distributions, cat

from python_ex._base import Utils, JSON_WRITEABLE
from python_ex._label import Learning_Mode
from python_ex._result import Log
from python_ex._numpy import Array_Process, Np_Dtype, ndarray, Evaluation_Process


# -- DEFINE CONSTNAT -- #
MAIN_RANK: int = 0


class Log_Category(Enum):
    LOSS = "loss"
    ACC = "acc"


# -- DEFINE CONFIG -- #
@dataclass
class Log_Config(Utils.Config):
    # Logging parameter in each mode;
    _Logging: Dict[Learning_Mode, Dict[Log_Category, List[str]]]  # str -> logging_loss name

    # Observe parameter in each mode;
    # if None -> all same like logging keys
    _Observing: Dict[Learning_Mode, Dict[Log_Category, List[str]]] = field(default_factory=dict)  # str -> logging_loss name

    def _get_parameter(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        return {
            "_Logging": dict(
                (_l_key.value, dict((_t_key.value, dict((_p, None) for _p in _l)) for _t_key, _l in _i.items())) for _l_key, _i in self._Logging.items()),
            "_Observing": dict(
                (_l_key.value, dict((_t_key.value, dict((_p, None) for _p in _l)) for _t_key, _l in _i.items())) for _l_key, _i in self._Observing.items())}

    def _convert_to_dict(self) -> Dict[str, Any]:
        return {
            "_Logging": dict(
                (_l_key.value, dict((_t_key.value, dict((_p, None) for _p in _l)) for _t_key, _l in _i.items())) for _l_key, _i in self._Logging.items()),
            "_Observing": dict(
                (_l_key.value, dict((_t_key.value, dict((_p, None) for _p in _l)) for _t_key, _l in _i.items())) for _l_key, _i in self._Observing.items())}


# -- Mation Function -- #
class Tensor_Process():
    @staticmethod
    def _holder(sample, is_shape: bool = False, value: int = 0, dtype: Np_Dtype = Np_Dtype.FLOAT):
        _array = Array_Process._converter(sample, is_shape, value, dtype)
        return tensor(_array)

    @staticmethod
    def _to_numpy(tensor: Tensor, dtype: Np_Dtype = Np_Dtype.FLOAT) -> ndarray:
        try:
            _array = tensor.numpy()
        except RuntimeError:
            _array = tensor.detach().numpy()

        return Array_Process._converter(_array, dtype=dtype)

    # In later fix it
    @staticmethod
    def _make_tensor(size: List[int], shape_sample=None, norm_option=None, dtype: Np_Dtype = Np_Dtype.UINT, value=[0, 1]):
        _value = Array_Process._get_random_array(size, dtype=dtype)
        return tensor(_value)

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

    class Evaluation():
        @staticmethod
        def iou(result: Tensor, label: Tensor, ingnore_class: List[int]) -> ndarray:
            _batch_size, _class_num = result.shape[0: 2]
            np_result = Tensor_Process._to_numpy(result.cpu().detach()).argmax(axis=1)  # [batch_size, h, w]
            np_label = Tensor_Process._to_numpy(label.cpu().detach())  # [batch_size, h, w]

            iou = Array_Process._converter([_batch_size, _class_num], True, dtype=Np_Dtype.FLOAT)

            for _b in range(_batch_size):
                iou[_b] = Evaluation_Process._iou(np_result[_b], np_label[_b], _class_num, ingnore_class)

            return iou.tolist()

        @staticmethod
        def miou(result: Tensor, label: Tensor, ingnore_class: List[int]) -> Tuple[ndarray, ndarray]:
            _batch_size, _class_num = result.shape[0: 2]
            np_result = Tensor_Process._to_numpy(result.cpu().detach()).argmax(axis=1)  # [batch_size, h, w]
            np_label = Tensor_Process._to_numpy(label.cpu().detach())  # [batch_size, h, w]

            iou = Array_Process._converter([_batch_size, _class_num], True, dtype=Np_Dtype.FLOAT)
            miou = Array_Process._converter([_batch_size, ], True, dtype=Np_Dtype.FLOAT)

            for _b in range(_batch_size):
                iou[_b], miou[_b] = Evaluation_Process._miou(np_result[_b], np_label[_b], _class_num, ingnore_class)

            return iou.tolist(), miou.tolist()


class Layer_Process():
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


class Debug():
    class Learning_Log(Log):
        _Data: Dict[str, Dict[str, JSON_WRITEABLE]]  # {Learning_mode:  {Logging_mode: {Logging_Parameter: {Epoch: []}}

        def __init__(self, logging_param: Dict[str, Dict[str, Dict[str, None]]], observing_param: Dict[str, Dict[str, Dict[str, None]]]):
            self._Observing = dict((
                _learning_key, dict((
                    _target_key, observing_param[_learning_key][_target_key] if _target_key in observing_param[_learning_key].keys() else _name_info
                ) for _target_key, _name_info in _target_info.items()) if _learning_key in observing_param.keys() else _target_info
            ) for _learning_key, _target_info in logging_param.items())

            super().__init__(data=self._make_data_holder(logging_param))

        # Freeze function
        def _make_data_holder(self, logging: Dict[str, Dict[str, Dict[str, None]]]):
            _holder = dict((
                _learning_key, dict((
                    _target_key, dict((
                        _name, {}) for _name in _name_info)
                ) for _target_key, _name_info in _target_info.items())
            ) for _learning_key, _target_info in logging.items())

            # time
            for _mode in _holder.keys():
                _holder[_mode]["process_time"] = {}
            return _holder

        def _set_activate_mode(self, learing_mode: Learning_Mode):
            self._Active_mode = learing_mode

        def _learning_logging(
                self,
                epoch: int,
                loss: Optional[Dict[str, JSON_WRITEABLE]] = None,
                acc: Optional[Dict[str, JSON_WRITEABLE]] = None,
                process_time: Optional[float] = None):
            _holder = {
                "loss": {},
                "acc": {},
                "process_time": {},
            }

            if loss is not None:
                _holder["loss"].update(dict((_name_key, {f"{epoch}": value}) for _name_key, value in loss.items()))

            if acc is not None:
                _holder["acc"].update(dict((_name_key, {f"{epoch}": value}) for _name_key, value in acc.items()))

            if process_time is not None:
                _holder["process_time"] = {f"{epoch}": process_time}

            self._insert(_holder, access_point=self._Data[self._Active_mode.value], is_overwrite=False)

        def _learning_observing(self, epoch: int):
            def _make_string(data: JSON_WRITEABLE):
                if isinstance(data, int):
                    return f"{data:>4d}, "

                elif isinstance(data, float):
                    return f"{data:>7.3f}, "

                elif isinstance(data, (tuple, list)):
                    _data = Array_Process._converter(data, dtype=Np_Dtype.FLOAT)
                    if len(_data.shape) >= 2:
                        _list_string = "["
                        for _value in _data.mean(0):
                            _list_string += f"{_data[_value]:>7.3f}, "
                        return f"{_list_string[:-2]}], "
                    else:
                        return f"{_data.mean():>7.3f}, "

                else:
                    return "Dict data"

            _learning_mode = self._Active_mode.value
            _tracking = self._Observing[_learning_mode]

            _debugging_string = ""

            for _target_key, _data in self._Data[_learning_mode].items():
                if isinstance(_data, dict):
                    _picked_data = self._get_data(
                        data_info=dict((_log_param, f"{epoch}") for _log_param in _tracking[_target_key].keys()),
                        access_point=_data)

                    _debugging_string += " ".join([f"{_key}: {_make_string(_value)}" for _key, _value in _picked_data.items()])

            return _debugging_string

        def _get_progress_time(self, epoch: int, is_average: bool = False):
            _learning_mode = self._Active_mode
            _time_list = self._get_data(data_info={"process_time": f"{epoch}"}, access_point=self._Data[_learning_mode.value])
            # in later fix it
            _time_list = _time_list[f"process_time_{epoch}"]
            if isinstance(_time_list, (float, list)):
                return _time_list if isinstance(_time_list, float) else sum(_time_list) / len(_time_list) if is_average else sum(_time_list)
            else:
                return False

        def _get_learning_time(self, epoch: int, max_batch_count: int):
            _sum_time = self._get_progress_time(epoch)
            if _sum_time:
                _maximun_time = _sum_time * max_batch_count
                return _sum_time, _maximun_time
            else:
                return 0.0, 0.0
