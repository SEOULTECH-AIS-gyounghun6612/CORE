from typing import Type, List, Any, Tuple
from subprocess import check_output
from enum import Enum

from torch import Tensor, distributions
from torch import zeros, ones, zeros_like, ones_like, arange, tensor, stack, clip, cat, float32
from torch import rand, randn, rand_like, randn_like

from python_ex._Base import TYPE_NUMBER, Directory
from python_ex._Numpy import Array_Process, Np_Dtype, ndarray, Evaluation_Process, Random_Process


# -- DEFINE CONSTANT -- #
class Learning_Process(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


# -- Main code -- #
class System_Utils():
    class Base():
        @staticmethod
        def _Print(data: Any, this_rank: int):
            if this_rank: print(data)

        @staticmethod
        def _Make_dir(obj_dir: str, root_dir: str, this_rank: int = 0):
            return Directory._Make(obj_dir, root_dir) if not this_rank else Directory._Divider_check("".join([root_dir, obj_dir]))

    class Cuda():
        @staticmethod
        def _Get_useable_gpu_list(threshold_of_rate: float = 0.5, threshold_of_size: int | None = None) -> List[Tuple[int, str]]:
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


class Tensor_Process():
    @staticmethod
    def _Make_tensor(size: int | List[int], value: TYPE_NUMBER | List[TYPE_NUMBER], rand_opt: Random_Process = None, dtype: Type | None = None) -> Tensor:
        if isinstance(value, list):
            _max_value = max(*value)
            _min_value = min(*value)
            _data_type = type(_max_value) if dtype is None else dtype

            if rand_opt is rand_opt.NORM:
                return (randn(size, dtype=_data_type) * (_max_value - _min_value)) + _min_value
            else:  # random process => unifrom
                return (rand(size, dtype=_data_type) * _max_value) - _min_value

        else:
            _data_type = float32 if dtype is None else dtype
            return ones(size, dtype=_data_type) * value if value else zeros(size, dtype=_data_type)

    @staticmethod
    def _Make_tensor_like(sample: Any, value: TYPE_NUMBER | List[TYPE_NUMBER], rand_opt: Random_Process = None, dtype: Type | None = None):
        _sample = tensor(sample) if not isinstance(sample, Tensor) else sample

        if isinstance(value, list):
            _max_value = max(*value)
            _min_value = min(*value)
            _data_type = sample.dtype if dtype is None else dtype

            if rand_opt is rand_opt.UNIFORM:
                return (randn_like(_sample, dtype=_data_type) * (_max_value - _min_value)) + _min_value
            else:  # random process => unifrom
                return (rand_like(_sample, dtype=_data_type) * _max_value) - _min_value

        else:
            _data_type = sample.dtype if dtype is None else dtype
            return ones_like(_sample, dtype=_data_type) * value if value else zeros_like(_sample, dtype=_data_type)

    @staticmethod
    def _Convert_from(source: Tensor | ndarray | List | Tuple, dtype: Type | None = None):
        if isinstance(source, Tensor):
            return source if dtype is None else source.to(dtype=dtype)
        else:
            raise NotImplementedError

    @staticmethod
    def _Arange(end: TYPE_NUMBER, start: TYPE_NUMBER = 0, step: TYPE_NUMBER = 1, dtype: Type | None = None) -> Tensor:
        _data_type = float32 if dtype is None else dtype
        return arange(start, end, step, dtype=_data_type)

    @staticmethod
    def _To_numpy(tensor: Tensor, dtype: Np_Dtype = Np_Dtype.FLOAT) -> ndarray:
        try:
            _array = tensor.numpy()
        except RuntimeError:
            _array = tensor.detach().numpy()

        return Array_Process._Convert_from(_array, dtype=dtype)

    @staticmethod
    def _Flatten(tensor: Tensor, is_minibatch=True):
        _shape = tensor.shape
        return tensor.reshape(_shape[0], -1) if is_minibatch else tensor.reshape(-1)

    @staticmethod
    def _range_cut(tensor: Tensor, range_min, rage_max):
        return clip(tensor, range_min, rage_max)

    @staticmethod
    def _norm(mu, std):
        return distributions.Normal(mu, std)

    @staticmethod
    def _stack(tensor_list: List[Tensor], dim: int = 0):
        return stack(tensor_list, dim=dim)

    @staticmethod
    def make_partition(tensor: Tensor, shape: List[int]):
        # _t_shpae = tensor.shape
        pass

    class Evaluation():
        @staticmethod
        def accuracy(result: Tensor, label: Tensor, threshold: float | None = None) -> List[bool]:
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
