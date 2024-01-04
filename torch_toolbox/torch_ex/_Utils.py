from typing import List, Any, Tuple
from subprocess import check_output

from python_ex._System import Path


class System_Utils():
    class Base():
        @staticmethod
        def _Print(data: Any, this_rank: int):
            if not this_rank: print(data)

        @staticmethod
        def _Make_dir(obj_dir: str, root_dir: str, this_rank: int = 0):
            if not this_rank:
                return Path._Make_directory(obj_dir, root_dir)
            else:
                return Path._Join(obj_dir, root_dir)

    class Cuda():
        @staticmethod
        def _Get_gpu_list(threshold: int | float | None = None) -> List[Tuple[str, int, int, int, int]]:
            _command = "nvidia-smi --format=csv --query-gpu=name,index,memory.total,memory.used,memory.free"
            _memory_info_text = check_output(_command.split()).decode('ascii').split('\n')[:-1][1:]

            _gpu_list: List[Tuple[str, int, int, int, int]] = []

            for _each_info in _memory_info_text:
                _gpu_info = _each_info.replace(" MiB", "").split(",")

                # gpu name, gpu index, total memory, used memory, free memory
                _gpu_info = (_gpu_info[0], int(_gpu_info[1]), int(_gpu_info[2]), int(_gpu_info[3]), int(_gpu_info[4]))

                if isinstance(threshold, float):  # about rate
                    if _gpu_info[4] / _gpu_info[2] < threshold: continue
                elif isinstance(threshold, int):  # absolute size
                    if _gpu_info[4] < threshold: continue
                else:
                    ...
                _gpu_list.append(_gpu_info)

            return _gpu_list


# class Evaluation_Utils():
#     @staticmethod
#     def accuracy(result: Tensor, label: Tensor, threshold: float | None = None) -> List[bool]:
#         _np_result = Tensor_Process._To_numpy(result.cpu().detach())  # [batch_size, c] or [batch_size]
#         _np_label = Tensor_Process._To_numpy(label.cpu().detach())  # [batch_size]

#         if result.shape[1] >= 2:
#             _np_result = _np_result.argmax(axis=1)  # [batch_size, 1]
#         else:
#             _threshold = 0.0 if threshold is None else threshold
#             _np_result = _np_result > _threshold  # [batch_size, 1]

#         _result: ndarray = Array_Process._Convert_from((_np_result == _np_label), dtype=Np_Dtype.FLOAT).reshape(_np_result.shape[0])

#         return _result.tolist()

#     @staticmethod
#     def iou(result: Tensor, label: Tensor, ingnore_class: List[int]) -> ndarray:
#         _batch_size, _class_num = result.shape[0: 2]
#         np_result = Tensor_Process._To_numpy(result.cpu().detach()).argmax(axis=1)  # [batch_size, h, w]
#         np_label = Tensor_Process._To_numpy(label.cpu().detach())  # [batch_size, h, w]

#         iou = Array_Process._Make_array_like([_batch_size, _class_num], 0, dtype=Np_Dtype.FLOAT)

#         for _b in range(_batch_size):
#             iou[_b] = Evaluation_Process._iou(np_result[_b], np_label[_b], _class_num, ingnore_class)

#         return iou.tolist()

#     @staticmethod
#     def miou(result: Tensor, label: Tensor, ingnore_class: List[int]) -> Tuple[ndarray, ndarray]:
#         _batch_size, _class_num = result.shape[0: 2]
#         np_result = Tensor_Process._To_numpy(result.cpu().detach()).argmax(axis=1)  # [batch_size, h, w]
#         np_label = Tensor_Process._To_numpy(label.cpu().detach())  # [batch_size, h, w]

#         iou = Array_Process._Make_array([_batch_size, _class_num], 0, dtype=Np_Dtype.FLOAT)
#         miou = Array_Process._Make_array([_batch_size, ], 0, dtype=Np_Dtype.FLOAT)

#         for _b in range(_batch_size):
#             iou[_b], miou[_b] = Evaluation_Process._miou(np_result[_b], np_label[_b], _class_num, ingnore_class)

#         return iou.tolist(), miou.tolist()


# class Layer_Process():
#     @staticmethod
#     def _get_conv_pad(kernel_size, input_size, interval=1, stride=1):
#         if type(kernel_size) != list:
#             kernel_size = [kernel_size, kernel_size]

#         if stride != 1:
#             size_h = input_size[0]
#             size_w = input_size[1]

#             pad_hs = (stride - 1) * (size_h - 1) + interval * (kernel_size[0] - 1)
#             pad_ws = (stride - 1) * (size_w - 1) + interval * (kernel_size[1] - 1)
#         else:
#             pad_hs = interval * (kernel_size[0] - 1)
#             pad_ws = interval * (kernel_size[1] - 1)

#         pad_l = pad_hs // 2
#         pad_t = pad_ws // 2

#         return [pad_t, pad_ws - pad_t, pad_l, pad_hs - pad_l]

#     @staticmethod
#     def _concatenate(layers, dim=1):
#         tmp_layer = layers[0]
#         for _layer in layers[1:]:
#             tmp_layer = cat([tmp_layer, _layer], dim=dim)

#         return tmp_layer
