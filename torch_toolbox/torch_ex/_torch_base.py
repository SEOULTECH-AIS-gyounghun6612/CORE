from typing import Dict, List, Tuple, Optional, Union
from enum import Enum

from torch import Tensor, zeros, ones, zeros_like, ones_like, tensor, stack, clip, distributions, cat
from torch import uint8, float32

from python_ex._base import JSON_WRITEABLE
from python_ex._result import Tracker
from python_ex._numpy import Array_Process, Np_Dtype, ndarray, Evaluation_Process, Random_Process


# -- DEFINE CONSTNAT -- #
class Learning_Mode(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class Parameter_Type(Enum):
    LOSS = "loss"
    ACC = "acc"


class Data_Type(Enum):
    UINT = uint8
    FLOAT = float32


# -- Mation Function -- #
class Tensor_Process():
    @staticmethod
    def _Make_tensor(size: List[int], value: Union[int, List[int]], random_option: Optional[Random_Process] = None, dtype: Optional[Data_Type] = None):
        _data_type = dtype if dtype is None else dtype.value
        if isinstance(value, int):
            return ones(size, dtype=_data_type) * value if value else zeros(size, dtype=_data_type)

        else:
            # make random tensor -> not yet
            raise ValueError("function of make random value tensor form size data is not yet")

    @staticmethod
    def _Make_tensor_like(sample: Union[ndarray, Tensor], value: Union[int, List[int]], random_option: Optional[Random_Process] = None, dtype: Optional[Data_Type] = None):
        _data_type = dtype if dtype is None else dtype.value
        if isinstance(value, int):
            _sample = tensor(sample) if isinstance(sample, ndarray) else sample
            return ones_like(_sample, dtype=_data_type) * value if value else zeros_like(_sample, dtype=_data_type)

        else:
            # make random tensor -> not yet
            raise ValueError("function of make random value tensor form sample data is not yet")

    @staticmethod
    def _To_numpy(tensor: Tensor, dtype: Np_Dtype = Np_Dtype.FLOAT) -> ndarray:
        try:
            _array = tensor.numpy()
        except RuntimeError:
            _array = tensor.detach().numpy()

        return Array_Process._converter(_array, dtype=dtype)

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
            np_result = Tensor_Process._To_numpy(result.cpu().detach()).argmax(axis=1)  # [batch_size, h, w]
            np_label = Tensor_Process._To_numpy(label.cpu().detach())  # [batch_size, h, w]

            iou = Array_Process._converter([_batch_size, _class_num], True, dtype=Np_Dtype.FLOAT)

            for _b in range(_batch_size):
                iou[_b] = Evaluation_Process._iou(np_result[_b], np_label[_b], _class_num, ingnore_class)

            return iou.tolist()

        @staticmethod
        def miou(result: Tensor, label: Tensor, ingnore_class: List[int]) -> Tuple[ndarray, ndarray]:
            _batch_size, _class_num = result.shape[0: 2]
            np_result = Tensor_Process._To_numpy(result.cpu().detach()).argmax(axis=1)  # [batch_size, h, w]
            np_label = Tensor_Process._To_numpy(label.cpu().detach())  # [batch_size, h, w]

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


class Tracking():
    class To_Process(Tracker):
        _Data: Dict[str, Dict[str, JSON_WRITEABLE]]  # {Learning_mode:  {Logging_mode: {Logging_Parameter: {Epoch: []}}

        def __init__(self, tracking_param: Dict[Learning_Mode, Dict[Parameter_Type, List[str]]], observing_param: Dict[Learning_Mode, Dict[Parameter_Type, Optional[List[str]]]]):
            _tracking_param = dict((
                _mode_key.value,
                dict((
                    _process_key.value,
                    dict((_p, None) for _p in _name_info)
                ) for _process_key, _name_info in _process_info.items())
            ) for _mode_key, _process_info in tracking_param.items())

            self._observing_param = dict((
                _mode_key.value,
                dict((
                    _process_key.value,
                    _tracking_param[_mode_key.value][_process_key.value] if _name_info is None else dict((_p, None) for _p in _name_info)
                ) for _process_key, _name_info in _process_info.items())
            ) for _mode_key, _process_info in observing_param.items())

            super().__init__(data=self._Make_data_holder(_tracking_param))

        # Freeze function
        def _Make_data_holder(self, logging: Dict[str, Dict[str, Dict[str, None]]]):
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

        def _Set_activate_mode(self, learing_mode: Learning_Mode):
            self._Active_mode = learing_mode

        def _Learning_tracking(
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

        def _Learning_observing(self, epoch: int):
            def _Make_string(data: JSON_WRITEABLE):
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
            _tracking = self._observing_param[_learning_mode]

            _tracking_string = ""

            for _target_key in _tracking.keys():
                _access_point = self._Data[_learning_mode][_target_key]

                if isinstance(_access_point, dict):
                    _picked_data = self._get_data(
                        data_info=dict((_log_param, f"{epoch}") for _log_param in _tracking[_target_key].keys()),
                        access_point=_access_point)

                    _tracking_string += " ".join([f"{_key}: {_Make_string(_value)}" for _key, _value in _picked_data.items()])

            return _tracking_string

        def _Get_progress_time(self, epoch: int):
            _learning_mode = self._Active_mode
            _time_list = self._get_data(data_info={"process_time": f"{epoch}"}, access_point=self._Data[_learning_mode.value])["process_time"]

            if isinstance(_time_list, (float, list)):
                return _time_list if isinstance(_time_list, list) else [_time_list, ]
            else:
                return [0.0, ]

        def _Get_observing_length(self, epoch: int):
            _learning_mode = self._Active_mode.value

            _tracking = self._observing_param[_learning_mode]
            _target_key = Parameter_Type.ACC.value if Parameter_Type.ACC.value in _tracking.keys() else Parameter_Type.LOSS.value
            _access_point = self._Data[_learning_mode][_target_key]

            if isinstance(_access_point, dict):
                _access_key = list(_tracking[_target_key].keys())[0]
                _picked_data = self._get_data(
                    data_info={_access_key: f"{epoch}"},
                    access_point=_access_point)[_access_key]

                return len(_picked_data) if isinstance(_picked_data, (list, tuple)) else 1
            else:
                return 0
