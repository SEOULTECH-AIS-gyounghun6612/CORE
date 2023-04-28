from typing import Dict, List, Any, Tuple, Optional, Union
from subprocess import check_output
from enum import Enum

from torch import Tensor, distributions
from torch import uint8, float32
from torch import zeros, ones, zeros_like, ones_like, arange, tensor, stack, clip, cat
from torch import rand, randn, rand_like, randn_like

from python_ex._base import NUMBER, Directory
from python_ex._result import Tracker
from python_ex._numpy import Array_Process, Np_Dtype, ndarray, Evaluation_Process, Random_Process


# -- DEFINE CONSTANT -- #
MAIN_RANK: int = 0


class Process_Name(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class Data_Type(Enum):
    UINT = uint8
    FLOAT = float32


# -- Main code -- #
class System_Utils():
    @staticmethod
    def _Make_directory(obj_dir: str, process_id: int, root_dir: str | None = None) -> str:
        return Directory._Make(f"{obj_dir}", root_dir) if process_id is MAIN_RANK else Directory._Divider_check(f"{root_dir}{Directory._Divider}{obj_dir}{Directory._Divider}")

    class Cuda():
        @staticmethod
        def _Get_useable_gpu_list(threshold_of_rate: float = 0.5, threshold_of_size: Optional[int] = None) -> List[Tuple[int, str]]:
            _command = "nvidia-smi --format=csv --query-gpu=name,index,memory.total,memory.used,memory.free"
            _memory_info_text = check_output(_command.split()).decode('ascii').split('\n')[:-1][1:]

            _useable_gpu_list = []

            for _each_info in _memory_info_text:
                _info_list = _each_info.replace(" MiB", "").split(",")  # gpu name, gpu index, total memory, used memory, free memory

                if threshold_of_size is not None:
                    if int(_info_list[4]) >= threshold_of_size:
                        _useable_gpu_list.append((int(_info_list[1]), _info_list[0]))
                else:
                    if int(_info_list[4]) / int(_info_list[2]) >= threshold_of_rate:
                        _useable_gpu_list.append((int(_info_list[1]), _info_list[0]))

            return _useable_gpu_list

    class Learning_Logger(Tracker):
        """
        """
        def __init__(self, Learning_info: Dict, logging_param: Dict[Process_Name, List[str]]):
            _tracking_param = dict((_mode_key.value, dict((_param, {}) for _param in _logging_params)) for _mode_key, _logging_params in logging_param.items())

            # dict((_param_name, {}) if isinstance(_param_name, dict) else (_param_name, {}) for _param_name in _logging_params)
            #  _logging_params if isinstance(_logging_params, str) else _logging_params

            super().__init__(Learning_info, _tracking_param)

        def _Learning_logging(self, process_name: Process_Name, epoch: int, data: Dict[str, NUMBER]):
            """
            """
            self._Insert(data_block=dict((_logging_param, {str(epoch): _value}) for _logging_param, _value in data.items()), access_point=self._Data[process_name.value])

        def _Make_tracking_text(self, process_name: Process_Name, epoch: int, data_name_list: List[str]):
            """
            """
            _trackin_string = ""

            for _name in data_name_list:
                _observing_data = self._Get_data(data_info={str(epoch): None}, access_point=self._Data[process_name.value][_name])

                if isinstance(_observing_data, int):
                    _string = f"{_observing_data:>4d}, "
                elif isinstance(_observing_data, float):
                    _string = f"{_observing_data:>7.3f}, "
                elif isinstance(_observing_data, (tuple, list)):
                    _data = Array_Process._Convert_from(_observing_data, dtype=Np_Dtype.FLOAT)
                    _string = f"[{', '.join([f'{_value:>7.3f}' for _value in _data.mean(0)])}], " if len(_data.shape) >= 2 else f"{_data.mean():>7.3f}, "
                else:
                    _string = "Dict data"

                _trackin_string = f"{_trackin_string} {_name}: {_string}, "

            return _trackin_string[:-2]

        def _Get_data_length(self, process_name: Process_Name, epoch: int, params: str):
            _observing_data = self._Get_data(data_info={str(epoch): None}, access_point=self._Data[process_name.value][params])
            return len(_observing_data)


class Tensor_Process():
    @staticmethod
    def _Make_tensor(size: Union[int, List[int]], value: Union[NUMBER, List[NUMBER]], rand_opt: Random_Process = Random_Process.NORM, dtype: Optional[Data_Type] = None):
        _data_type = dtype if dtype is None else dtype.value
        if isinstance(value, list):
            _max_value = max(*value)
            _min_value = min(*value)

            if rand_opt is rand_opt.NORM:
                return (randn(size, dtype=_data_type) * (_max_value - _min_value)) + _min_value
            else:  # random process => unifrom
                return (rand(size, dtype=_data_type) * _max_value) - _min_value

        else:
            return ones(size, dtype=_data_type) * value if value else zeros(size, dtype=_data_type)

    @staticmethod
    def _Make_tensor_like(sample: Any, value: Union[NUMBER, List[NUMBER]], rand_opt: Random_Process = Random_Process.NORM, dtype: Optional[Data_Type] = None):
        _data_type = dtype if dtype is None else dtype.value
        _sample = tensor(sample) if not isinstance(sample, Tensor) else sample

        if isinstance(value, list):
            _max_value = max(*value)
            _min_value = min(*value)

            if rand_opt is rand_opt.UNIFORM:
                return (randn_like(_sample, dtype=_data_type) * (_max_value - _min_value)) + _min_value
            else:  # random process => unifrom
                return (rand_like(_sample, dtype=_data_type) * _max_value) - _min_value

        else:
            return ones_like(_sample, dtype=_data_type) * value if value else zeros_like(_sample, dtype=_data_type)

    @staticmethod
    def _Converte_from():
        ...

    @staticmethod
    def _Arange(end: NUMBER, start: NUMBER = 0, step: NUMBER = 1, dtype: Optional[Data_Type] = None):
        _data_type = dtype if dtype is None else dtype.value
        return arange(start, end, step, dtype=_data_type)

    @staticmethod
    def _To_numpy(tensor: Tensor, dtype: Np_Dtype = Np_Dtype.FLOAT) -> ndarray:
        try:
            _array = tensor.numpy()
        except RuntimeError:
            _array = tensor.detach().numpy()

        return Array_Process._Convert_from(_array, dtype=dtype)

    @staticmethod
    def _Flatten(tensor: Tensor):
        _b = tensor.shape[0]
        return tensor.reshape(_b, -1)

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
        def accuracy(result: Tensor, label: Tensor, threshold: Optional[float] = None) -> List[bool]:
            _np_result = Tensor_Process._To_numpy(result.cpu().detach())  # [batch_size, c] or [batch_size]
            _np_label = Tensor_Process._To_numpy(label.cpu().detach())  # [batch_size]

            if result.shape[1] >= 2:
                _np_result = _np_result.argmax(axis=1)  # [batch_size, 1]
            else:
                _threshold = 0.0 if threshold is None else threshold
                _np_result = _np_result > _threshold  # [batch_size, 1]

            _result: ndarray = Array_Process._Convert_from((_np_result == _np_label), dtype=Np_Dtype.FLOAT).reshape(_np_result.shape[0])

            return _result.tolist()

        @staticmethod
        def iou(result: Tensor, label: Tensor, ingnore_class: List[int]) -> ndarray:
            _batch_size, _class_num = result.shape[0: 2]
            np_result = Tensor_Process._To_numpy(result.cpu().detach()).argmax(axis=1)  # [batch_size, h, w]
            np_label = Tensor_Process._To_numpy(label.cpu().detach())  # [batch_size, h, w]

            iou = Array_Process._Make_array_like([_batch_size, _class_num], 0, dtype=Np_Dtype.FLOAT)

            for _b in range(_batch_size):
                iou[_b] = Evaluation_Process._iou(np_result[_b], np_label[_b], _class_num, ingnore_class)

            return iou.tolist()

        @staticmethod
        def miou(result: Tensor, label: Tensor, ingnore_class: List[int]) -> Tuple[ndarray, ndarray]:
            _batch_size, _class_num = result.shape[0: 2]
            np_result = Tensor_Process._To_numpy(result.cpu().detach()).argmax(axis=1)  # [batch_size, h, w]
            np_label = Tensor_Process._To_numpy(label.cpu().detach())  # [batch_size, h, w]

            iou = Array_Process._Make_array([_batch_size, _class_num], 0, dtype=Np_Dtype.FLOAT)
            miou = Array_Process._Make_array([_batch_size, ], 0, dtype=Np_Dtype.FLOAT)

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
