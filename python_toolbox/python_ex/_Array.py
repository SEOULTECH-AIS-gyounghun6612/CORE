from __future__ import annotations
from typing import List, Dict, Union, Optional, Tuple, Literal
from enum import Enum
from io import BufferedReader

import numpy as np

from numpy import ndarray

from ._System import Path, File


class Binary_Order(Enum):
    DIRECTION_ROW = 'C'
    DIRECTION_COL = 'F'
    DIRECTION_ANY = 'A'  # ‘A’ (short for Any) means ‘F’ if array is Fortran contiguous, ‘C’ otherwise <- in later check the Fortran


# -- DEFINE FUNCTION -- #
class Array_IO(File.Basement):
    @staticmethod
    def _Write_by_binary(file_path: str, data: ndarray, data_format: str):
        with open(file_path, "wb") as f:
            data.tofile(f, format=data_format)

    @staticmethod
    def _Read_from_binary(file_stream: BufferedReader, data_format: type):
        _b_data = file_stream.read()

        return np.frombuffer(_b_data, dtype=data_format)

    @staticmethod
    def _Save(save_dir: str, file_name: str, array: Union[ndarray, List[ndarray], Dict[str, ndarray]], is_compress: bool = True):
        """Auto save array.

        If save ONE array, using "numpy.save". The others, using "numpy.savez" or "numpy.savez_compressed"

        Parameters
        --------------------
        save_dir
            Directory to save data.
        file_name
            File name to save data. This function automatically fixes if you use the wrong ext or don't know what to use ext.
        array
            Saved data.
        is_compress
            When saving multi-array and this value is True, switch to using compressed.
        """
        # if save directory not exist, raise error
        assert Path._Exist_check(save_dir, Path.Type.DIRECTORY), f"Data save Directrory : {save_dir} is NOT EXIST.\n check it"

        _file_ext = "npy" if isinstance(array, ndarray) else "npz"

        if file_name.find(".") == -1:
            _file_name = f"{file_name}.{_file_ext}"
        elif file_name.split(".")[-1] is _file_ext:
            _file_name = file_name
        else:
            _file_name = file_name.replace(file_name.split(".")[-1], _file_ext)

        _save_file = f"{Path._Seperater_check(save_dir)}{_file_name}"

        if isinstance(array, ndarray):
            np.save(_save_file, array)
        elif isinstance(array, list):
            np.savez_compressed(_save_file, *array) if is_compress else np.savez(_save_file, *array)
        else:
            np.savez_compressed(_save_file, **array) if is_compress else np.savez(_save_file, **array)

    @staticmethod
    def _Load(file_name: str, directory: str):
        return np.load(Path._Join(file_name, directory))

    # in later fix it
    class RLE():
        @staticmethod
        def _from_nparray(data: ndarray, order: Literal['A', 'C', 'F'] = 'F'):
            if data is not None:
                _size = data.shape
                _size = (int(_size[0]), int(_size[1]))

                return_RLE = []
                # zeros
                if (data == np.zeros_like(data)).all():
                    return_RLE.append(_size[0] * _size[1])
                # ones
                elif (data == np.ones_like(data)).all():
                    return_RLE.append(0)
                    return_RLE.append(_size[0] * _size[1])
                # else
                else:
                    _line = data.reshape(_size[0] * _size[1], order=order)
                    _count_list = []

                    # in later add annotation
                    for _type in range(2):
                        _points = np.where(_line == _type)[0]
                        _filter = _points[1:] - _points[:-1]
                        _filter = _filter[np.where(_filter != 1)[0]]
                        _count = _filter[np.where(_filter != 1)[0]] - 1

                        if _points[0]:
                            _count = np.append((_points[0], ), _count)
                        _count_list.append(_count)

                    _one_count, _zero_count = _count_list

                    if _line[0]:
                        _zero_count = np.append((0, ), _zero_count)

                    for _ct in range(len(_one_count)):
                        return_RLE.append(int(_zero_count[_ct]))
                        return_RLE.append(int(_one_count[_ct]))

                    _last_count = int(len(_line) - sum(return_RLE))
                    return_RLE.append(_last_count)

                return {"size": _size, "counts": return_RLE}

            else:
                return None

        @staticmethod
        def _to_nparray(data, order: Literal['A', 'C', 'F'] = "F"):
            if data is not None:
                _rle_data = data["counts"]
                _list = []
                for _type, _count in enumerate(_rle_data):
                    [_list.append(_type % 2) for _ct in range(_count)]
                _list = np.reshape(_list, data["size"], order)
                return _list
            else:
                return None


class Array_Toolbox():
    @staticmethod
    def _Convert_to_square_matrix(obj_array: np.ndarray, m: int | None = None):
        assert len(obj_array.shape) == 2

        _c, _w = obj_array.shape

        _max = max(_c, _w) if m is None else max(_c, _w, m)

        if (_c == _w) and _c >= _max:
            return obj_array
        else:
            _holder = np.eye(_max)
            _holder[:_c, :_w] = obj_array
            return _holder

    @staticmethod
    def _Inv(obj_array: np.ndarray):
        _convert_array = Array_Toolbox._Convert_to_square_matrix(obj_array)
        return np.linalg.inv(_convert_array)

    @staticmethod
    def _Norm(array: ndarray):
        _m = np.mean(array)
        _std = np.std(array)
        return (array - _m) / _std

    @staticmethod
    def _Range_masking(obj_array: ndarray, cut_range: Tuple[int, int] = (-1, 1), direction="outside"):
        """
        #### 설정된 범위에 따른 입력된 배열의 마스킹 결과 출력
        ---------------------------------------------------------------------------------------
        #### Parameter

        """
        _max, _min = max(cut_range), min(cut_range)

        _max_mask = obj_array >= _max
        _min_mask = obj_array <= _min

        if direction == "outside":
            return obj_array * (_max_mask + _min_mask)
        else:  # inside
            return obj_array * (1 - _max_mask) * (1 - _min_mask)

    @staticmethod
    def range_converter(obj_array: ndarray, from_range: Tuple[int, int], to_range: Tuple[int, int]) -> ndarray:
        _fr_max, _fr_min = max(from_range), min(from_range)
        _tr_max, _tr_min = max(to_range), min(to_range)

        _obj_array = np.clip(obj_array, _fr_min, _fr_max)
        _obj_array = (_obj_array - _fr_min) / (_fr_max - _fr_min)

        return (_obj_array * (_tr_max - _tr_min)) + _tr_min

    # what is it?
    @staticmethod
    def bincount(array_1D, max_value):
        array_1D = np.round(array_1D) * (array_1D >= 0)
        holder = np.bincount(array_1D.reshape(-1))

        if len(holder) <= max_value:
            count_list = np.zeros(max_value + 1)
            count_list[:len(holder)] = holder
        else:
            count_list = holder[:max_value]

        return count_list


class Evaluation_Process():
    # @staticmethod
    # # in later fix it
    # def _iou(result: ndarray, label: ndarray, class_num: int, ignore_class: List[int] = []):
    #     """

    #     """
    #     # result -> [h, w]
    #     # label -> [h, w]

    #     iou = []
    #     _G_interest = Custom_Array_Process._Make_array(result.shape, 0)
    #     for _ig_class in ignore_class:
    #         _G_interest = np.logical_or(_G_interest, result != _ig_class)

    #     for _class in range(class_num):
    #         _G_result: ndarray = (result == _class)
    #         _G_label: ndarray = (label == _class)

    #         _intersection = np.logical_and(_G_result, _G_label)
    #         _union = np.logical_and(np.logical_or(_G_result, _G_label), _G_interest)

    #         iou.append(_intersection.sum() / _union.sum() if _union.sum() else 0.00)

    #     return iou

    # @classmethod
    # def _miou(cls, result: ndarray, label: ndarray, class_num: int, ignore_class: List[int] = []):
    #     iou = cls._iou(result, label, class_num, ignore_class)
    #     iou_np = Custom_Array_Process._Convert_from(iou, dtype=Np_Dtype.FLOAT)
    #     _used_class = list(range(class_num))
    #     for _ig_class in ignore_class:
    #         _used_class.remove(_ig_class)

    #     return [iou, np.mean(iou_np[_used_class])]

    class Baddeley():
        def __init__(self, p: int = 2) -> None:
            self.p = p

        def __call__(self, image: ndarray, target: ndarray) -> float:
            return self._get_value(image, target)

        def _get_value(self, image: ndarray, target: ndarray):
            c = np.sqrt(np.sum(np.array(target.shape) * np.array(target.shape))).item()
            N = target.shape[0] * target.shape[1]
            _h_map = np.array([np.ones(target.shape[1]) * _h_map for _h_map in range(target.shape[0])])
            _w_map = np.array([list(range(target.shape[1])) for _ct in range(target.shape[0])])

            tager_edge_points = np.where(target >= 1)
            if len(tager_edge_points[0]):
                _h_target_map = np.abs(_h_map[:, :, np.newaxis] - (np.ones((list(target.shape) + [1, ])) * tager_edge_points[0]))
                _w_target_map = np.abs(_w_map[:, :, np.newaxis] - (np.ones((list(target.shape) + [1, ])) * tager_edge_points[1]))

                _target_map = np.min(np.sqrt((_h_target_map * _h_target_map) + (_w_target_map * _w_target_map)), -1) / c
                _target_map = np.where(_target_map >= 1, 1, _target_map)

            else:
                _target_map = np.zeros(target.shape)

            image_edge_points = np.where(image >= 1)
            if len(image_edge_points[0]):
                _h_image_map = np.abs(_h_map[:, :, np.newaxis] - (np.ones((list(image.shape) + [1, ])) * image_edge_points[0]))
                _w_image_map = np.abs(_w_map[:, :, np.newaxis] - (np.ones((list(image.shape) + [1, ])) * image_edge_points[1]))

                _image_map = np.min(np.sqrt((_h_image_map * _h_image_map) + (_w_image_map * _w_image_map)), -1) / c
                _image_map = np.where(_image_map >= 1, 1, _image_map)

            else:
                _image_map = np.zeros(image.shape)

            # cal |w(d(x, A))-w(d(x, B))|
            _edge_compare = np.abs(_target_map - _image_map)
            return (np.sum(np.power(_edge_compare, self.p)) / N) ** (1.0 / float(self.p))

    class Confustion_Matrix():
        @staticmethod
        def _Calculate_Confusion_Matrix(array: ndarray, target: ndarray, interest: Optional[ndarray] = None):
            """
            Args:
                array :
                target : np.uint8 ndarray
                interest :
            Returns:
                Confusion Matrix (list)
            """
            if interest is None:
                interest = np.ones_like(array, np.uint8)
            _compare = (array == target).astype(np.uint8)

            _compare_255 = (_compare * 254)  # collect is 254, not 0 -> Tx
            _inv_compare_255 = ((1 - _compare) * 254)  # wrong is 254, not 0 -> Fx

            _tp = np.logical_and(_compare_255, target.astype(bool))       # collect_FG
            _tn = np.logical_and(_compare_255, ~(target.astype(bool)))    # collect_BG
            _fn = np.logical_and(_inv_compare_255, target.astype(bool))     # wrong_BG
            _fp = np.logical_and(_inv_compare_255, ~(target.astype(bool)))  # wrong_FG

            return (_tp * interest, _tn * interest, _fn * interest, _fp * interest)

        @staticmethod
        def _Confusion_Matrix_to_value(TP, TN, FN, FP):
            _pre = TP / (TP + FP) if TP + FP else TP / (TP + FP + 0.00001)
            _re = TP / (TP + FN) if TP + FN else TP / (TP + FN + 0.00001)
            _fm = (2 * _pre * _re) / (_pre + _re) if _pre + _re else (2 * _pre * _re) / (_pre + _re + 0.00001)

            return _pre, _re, _fm
