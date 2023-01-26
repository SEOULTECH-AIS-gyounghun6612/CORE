from enum import Enum
from typing import List, Union, Optional, Tuple, Literal
import numpy as np
from numpy import ndarray

if __package__ == "":
    import _error as _e
else:
    from . import _error as _e

_error_message = _e.Custom_error("AIS_utils", "_numpy")


# -- DEFINE CONSTNAT -- #
class Np_Dtype(Enum):
    INT = np.int32
    UINT = np.uint8
    BOOL = np.bool8
    FLOAT = np.float32
    STRING = np.string_


class Random_Process(Enum):
    RANDOM = "random"
    NORM = "Norm"


# -- DEFINE FUNCTION -- #
class Numpy_IO():
    @staticmethod
    def _Save(save_dir: str, array: ndarray):
        """
        Args:
            save_dir :
            data :
        Returns:
            Empty
        """
        if save_dir.split("/")[-1].split(".")[-1] == "npz":
            np.savez_compressed(save_dir, data=array)
        else:
            np.savez(save_dir, data=array)

    @staticmethod
    def _Load(file_name: str, directory: str):
        ...

    class RLE():
        size_key = "size"
        count_key = "counts"

        @classmethod
        def _from_nparray(cls, data, order='F'):
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

                return {cls.size_key: _size, cls.count_key: return_RLE}

            else:
                return None

        @classmethod
        def _to_nparray(cls, data, order: Literal['A', 'C', 'F'] = "F"):
            if data is not None:
                rle_data = data[cls.count_key]
                array = np.array([], dtype=np.uint8)
                for _type, _count in enumerate(rle_data):
                    value = _type % 2
                    if value:
                        _tmp = np.ones(_count, dtype=np.uint8)
                    else:
                        _tmp = np.zeros(_count, dtype=np.uint8)

                    array = np.concatenate((array, _tmp), axis=None, dtype=np.uint8)

                array = np.reshape(array, data[cls.size_key], order)
                return array
            else:
                return None


class Array_Process():
    @staticmethod
    def _Make_array(size: List[int], value: Union[int, List[int]], random_option: Optional[Random_Process] = None, dtype: Optional[Np_Dtype] = None):
        ...

    @staticmethod
    def _Make_array_like(size: List[int], value: Union[int, List[int]], random_option: Optional[Random_Process] = None, dtype: Optional[Np_Dtype] = None):
        ...

    @staticmethod
    def _converter(source: Union[float, List, Tuple, ndarray], is_shape: bool = False, value: int = 0, dtype: Optional[Np_Dtype] = None):
        if isinstance(source, ndarray):
            _source = source
        elif is_shape:
            if isinstance(source, (list, tuple)):
                _shape = source
            else:
                _shape = int(source)
            _source = np.array(value * np.ones(_shape))
        else:
            _source = np.array(source)
        return _source if dtype is None else _source.astype(dtype.value)

    @classmethod
    def _get_random_array(cls, shape: List, scale: int = 1, minnimum: int = 0, norm_factor: Tuple[float, float] = (1, 0), dtype: Optional[Np_Dtype] = None):
        _array = cls._converter(
            source=norm_factor[1] * np.random.randn(*shape) + norm_factor[0] if norm_factor is not None else np.random.rand(*shape), dtype=dtype)
        return _array * scale + minnimum

    @classmethod
    def _normalization(cls, array: ndarray):
        _m = np.mean(array)
        _std = np.std(array)

        return (cls._converter(array, dtype=Np_Dtype.FLOAT) - _m) / _std

    @staticmethod
    def _clip(array: ndarray, value_min: int, value_max: int):
        return np.clip(array, value_min, value_max)

    # in later fix it

    @staticmethod
    def _make_mask():
        ...

    @staticmethod
    def range_cut(array: ndarray, cut_range: Union[int, Tuple[int, int]] = (-1, 1), direction="outside", ouput_type="active_map"):
        under_thread = cut_range if isinstance(cut_range, int) else min(cut_range)
        over_thread = cut_range if isinstance(cut_range, int) else max(cut_range)

        if direction == "outside":
            # --here-- under_thread --nope!-- over_thread --here--
            under_cuted = (array <= under_thread)
            over_cuted = (array >= over_thread)

            output = np.logical_or(under_cuted, over_cuted)
        elif direction == "upper":
            # --nope!-- under_thread --nope!-- over_thread --here--
            over_cuted = (array >= over_thread)

            output = over_cuted
        elif direction == "under":
            # --here-- under_thread --nope!-- over_thread --nope!--
            under_cuted = (array <= under_thread)

            output = under_cuted
        else:
            # --nope!-- under_thread --here-- over_thread --nope!--
            under_cuted = (array >= under_thread)
            over_cuted = (array <= over_thread)

            output = np.logical_and(under_cuted, over_cuted)

        return output if ouput_type == "active_map" else array * Array_Process._converter(output, dtype=Np_Dtype.INT)

    @staticmethod
    def range_converter(array: ndarray, from_range: List[Union[int, float]], to_range: List[Union[int, float]], dtype: Np_Dtype) -> ndarray:
        _from_range_max = from_range[0]
        _from_range_min = from_range[1]
        _from_term = _from_range_max - _from_range_min

        _to_range_max = to_range[0]
        _to_range_min = to_range[1]
        _to_term = _to_range_max - _to_range_min

        _convert = (array - _from_range_min) / _from_term
        _convert = (_convert * _to_term) + _to_range_min

        if dtype in [Np_Dtype.UINT, Np_Dtype.INT]:
            _convert = np.round(_convert)

        return Array_Process._converter(_convert, dtype=dtype)

    @staticmethod
    def stack(data_list: list, channel=-1):
        return np.stack(data_list, axis=channel)

    # @staticmethod
    # def shift(data: ndarray, axis: int, diretion: int):
    #     pass

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


class Image_Process():
    @staticmethod
    def _image_normalization(image: ndarray, mean: List[float] = [0.485, 0.456, 0.40], std: List[float] = [0.229, 0.224, 0.225]):
        _, _, _c = image.shape
        _norm_image = Array_Process._converter(image, dtype=Np_Dtype.FLOAT)

        for _ct_c in range(_c):
            _norm_image[:, :, _ct_c] = ((image[:, :, _ct_c] / 255) - mean[_ct_c]) / std[_ct_c]

        return _norm_image

    @staticmethod
    def _image_denormalization(norm_image: ndarray, mean: List[float] = [0.485, 0.456, 0.40], std: List[float] = [0.229, 0.224, 0.225]):
        _, _, _c = norm_image.shape
        _denorm_image = Array_Process._converter(norm_image, dtype=Np_Dtype.UINT)

        for _ct_c in range(_c):
            _denorm_image[:, :, _ct_c] = ((norm_image[:, :, _ct_c] * std[_ct_c]) + mean[_ct_c]) * 255

        return _denorm_image

    @staticmethod
    def _distance(delta_x: ndarray, delta_y: ndarray, is_euclid: bool):
        if is_euclid:
            return np.sqrt(np.square(delta_x, dtype=np.float32) + np.square(delta_y, dtype=np.float32), dtype=np.float32)
        else:
            return np.abs(delta_x, dtype=np.float32) + np.abs(delta_y, dtype=np.float32)

    @staticmethod
    def _gredient_direction(delta_x: ndarray, delta_y: ndarray, sector: int = 8, is_half_cut: bool = True):
        """
        if sector count is even (2n)
        +0 ~ +pi -> n sector, n block (size pi / n)
        -0 ~ -pi -> n sector, n block (size pi / n)

        if sector count is odd (2n + 1) => 2 * (2n + 1)
        +0 ~ +pi -> 2n + 1 sector, n block (size 2 * pi / (2n + 1)) -> left pi / (2n + 1)
        -0 ~ -pi -> 2m + 1 sector, n block (size 2 * pi / (2nm + 1)) -> left pi / (2n + 1)
        last one is left sector (2 * pi / (2n + 1))
        """
        _block = sector if sector % 2 else sector / 2
        _block_to_sector = 2 if sector % 2 else 1
        _term = _block_to_sector * np.pi / _block
        _bais = _term / 2

        # get theta
        theta = np.arctan2(delta_y, delta_x)

        if is_half_cut and not (sector % 2):
            theta = theta + np.pi * (theta < 0)
            sector = int(sector / 2)

        # theta convert to direction -> start in 180 dgree (clockwize)
        _tmp_holder = []
        _theta = theta.copy()
        for _ct in range(sector):
            _th = np.pi - (_bais + _term * _ct)
            _filterd = (_theta >= _th)
            _tmp_holder.append(_filterd)
            _theta = _theta - _theta * _filterd

        _tmp_holder[0] = np.logical_or(_tmp_holder[0], theta == _theta)
        _tmp_holder = Array_Process.stack(_tmp_holder)
        _direction = np.argmax(_tmp_holder, -1)

        return _direction

    @staticmethod
    def _image_shift(image: ndarray, direction: int, step_size: int = 1):
        holder = np.zeros_like(image)
        if direction == 0:
            holder[:, :-step_size] = image[:, step_size:]  # left
        elif direction == 1:
            holder[:-step_size, :-step_size] = image[step_size:, step_size:]  # left top
        elif direction == 2:
            holder[:-step_size, :] = image[step_size:, :]  # top
        elif direction == 3:
            holder[:-step_size, step_size:] = image[step_size:, :-step_size]  # right top
        elif direction == 4:
            holder[:, step_size:] = image[:, :-step_size]  # right
        elif direction == 5:
            holder[step_size:, step_size:] = image[:-step_size, :-step_size]  # right down
        elif direction == 6:
            holder[step_size:, :] = image[:-step_size, :]  # down
        elif direction == 7:
            holder[step_size:, :-step_size] = image[:-step_size, step_size:]  # left down

        return holder

    @staticmethod
    def _conver_to_last_channel(image: ndarray):
        img_shape = image.shape
        if len(img_shape) == 2:
            # gray iamge
            return image[:, :, np.newaxis]
        else:
            # else image
            divide_data = [image[ct] for ct in range(img_shape[0])]
            return Array_Process.stack(divide_data)

    @staticmethod
    def _conver_to_first_channel(image: ndarray):
        img_shape = image.shape
        if len(img_shape) == 2:
            # gray iamge
            return image[np.newaxis, :, :]
        else:
            # else image
            divide_data = [image[:, :, ct] for ct in range(img_shape[-1])]
            return Array_Process.stack(divide_data, 0)

    @staticmethod
    def _direction_check(object_data: ndarray, direction_array: ndarray, check_list: List[int], is_bidirectional: bool = True):
        filtered = np.zeros_like(object_data)
        for _direction in check_list:
            _tmp_direction_check = np.ones_like(object_data)
            _label = Image_Process._image_shift(object_data, _direction, 1)
            _tmp_direction_check *= (object_data * (direction_array == _direction)) > _label
            if is_bidirectional:
                _label = Image_Process._image_shift(object_data, _direction + 4, 1)
                _tmp_direction_check *= (object_data * (direction_array == _direction)) > _label

            filtered += _tmp_direction_check

        return filtered

    @staticmethod
    def _color_finder(image: ndarray, color_list: List[Union[int, Tuple[int, int, int]]]):
        _holder: List[ndarray] = []
        for _color in color_list:
            _finded = np.all((image == _color), 2) if isinstance(_color, list) else (image == _color)
            _holder.append(_finded)

        return _holder[0].astype(int) if len(_holder) == 1 else np.logical_or(*[data for data in _holder])

    @staticmethod
    def _classfication_resize(original: ndarray, size: List[int]):
        _new = Array_Process._converter(size + [original.shape[-1], ], True)

        _pos = np.where(original == 1)
        _new_pos = []
        for _ct in range(len(_pos) - 1):
            _new_pos.append(np.round((size[_ct] - 1) * _pos[_ct] / original.shape[_ct]).astype(int))
        _new_pos.append(_pos[-1])

        _new[tuple(_new_pos)] = 1

        return _new

    # @staticmethod
    # def _string_to_img(string: str, shape: Union[Tuple[int, int], Tuple[int, int, int]]):
    #     return np.fromstring(string, dtype=np.uint8).reshape(shape)


class Evaluation_Process():
    @staticmethod
    # in later fix it
    def _iou(result: ndarray, label: ndarray, class_num: int, ignore_class: List[int] = []):
        """

        """
        # result -> [h, w]
        # label -> [h, w]

        iou = []
        _G_interest = Array_Process._converter(result.shape, True)
        for _ig_class in ignore_class:
            _G_interest = np.logical_or(_G_interest, result != _ig_class)

        for _class in range(class_num):
            _G_result: ndarray = (result == _class)
            _G_label: ndarray = (label == _class)

            _intersection = np.logical_and(_G_result, _G_label)
            _union = np.logical_and(np.logical_or(_G_result, _G_label), _G_interest)

            iou.append(_intersection.sum() / _union.sum() if _union.sum() else 0.00)

        return iou

    @classmethod
    def _miou(cls, result: ndarray, label: ndarray, class_num: int, ignore_class: List[int] = []):
        iou = cls._iou(result, label, class_num, ignore_class)
        iou_np = Array_Process._converter(iou, dtype=Np_Dtype.FLOAT)
        _used_class = list(range(class_num))
        for _ig_class in ignore_class:
            _used_class.remove(_ig_class)

        return [iou, np.mean(iou_np[_used_class])]

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


# in later fix it
# def Neighbor_Confusion_Matrix(
#         img: np.ndarray, target: np.ndarray, interest: np.ndarray) -> list:
#     """
#     Args:
#         img :
#         target : np.uint8 ndarray
#         ext : if you want image ext.
#               if you set Default, and save_dir has not ext, autometically set .avi
#         frame :
#         shape :
#     Returns:
#         Confusion Matrix (list)
#     """
#     _nb_tp = np.zeros_like(img[1: -1, 1: -1])
#     _nb_tn = np.zeros_like(img[1: -1, 1: -1])
#     _nb_fp = np.zeros_like(img[1: -1, 1: -1])
#     _nb_fn = np.zeros_like(img[1: -1, 1: -1])

#     for _h_c in range(3):
#         image_h_s = NEIGHBOR_SPACE[_h_c][0]
#         image_h_e = NEIGHBOR_SPACE[_h_c][1]
#         target_h_s = NEIGHBOR_SPACE[2 - _h_c][0]
#         target_h_e = NEIGHBOR_SPACE[2 - _h_c][1]
#         roi_h_s = NEIGHBOR_SPACE2[_h_c][0]
#         roi_h_e = NEIGHBOR_SPACE2[_h_c][1]

#         for _w_c in range(3):
#             image_w_s = NEIGHBOR_SPACE[_w_c][0]
#             image_w_e = NEIGHBOR_SPACE[_w_c][1]
#             target_w_s = NEIGHBOR_SPACE[2 - _w_c][0]
#             target_w_e = NEIGHBOR_SPACE[2 - _w_c][1]
#             roi_w_s = NEIGHBOR_SPACE2[_w_c][0]
#             roi_w_e = NEIGHBOR_SPACE2[_w_c][1]

#             cutted_image = img[image_h_s: image_h_e, image_w_s: image_w_e]
#             cutted_interest = interest[image_h_s: image_h_e, image_w_s: image_w_e]
#             cutted_target = target[target_h_s: target_h_e, target_w_s: target_w_e]

#             _cm = Calculate_Confusion_Matrix(cutted_image, cutted_target, cutted_interest)

#             _nb_tp = np.logical_or(_nb_tp, _cm[0][roi_h_s: roi_h_e, roi_w_s: roi_w_e])
#             _nb_tn = np.logical_or(_nb_tn, _cm[1][roi_h_s: roi_h_e, roi_w_s: roi_w_e])
#             _nb_fp = np.logical_or(_nb_fp, _cm[3][roi_h_s: roi_h_e, roi_w_s: roi_w_e])
#             _nb_fn = np.logical_or(_nb_fn, _cm[2][roi_h_s: roi_h_e, roi_w_s: roi_w_e])

#     _nb_fn = _nb_fn.astype(np.uint8) - np.logical_and(_nb_fn, _nb_tp).astype(np.uint8)
#     _nb_fp = _nb_fp.astype(np.uint8) - np.logical_and(_nb_fp, _nb_tn).astype(np.uint8)

#     output = []
#     output.append(_nb_tp.astype(np.uint8))
#     output.append(_nb_tn.astype(np.uint8))
#     output.append(_nb_fn.astype(np.uint8))
#     output.append(_nb_fp.astype(np.uint8))

#     return output
