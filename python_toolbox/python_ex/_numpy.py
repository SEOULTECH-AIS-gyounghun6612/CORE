from typing import Any, List, Dict
import numpy as np
from numpy import ndarray

if __package__ == "":
    import _error as _e
else:
    from . import _error as _e

_error_message = _e.Custom_error("AIS_utils", "_numpy")


class file():
    @staticmethod
    def save_numpy(save_dir, data):
        """
        Args:
            save_dir :
            data :
        Returns:
            Empty
        """
        _array = np.array(data)
        if save_dir.split("/")[-1].split(".")[-1] == "npz":
            np.savez_compressed(save_dir, data=_array)
        else:
            np.savez(save_dir, data=_array)

    class RLE():
        size_key = "size"
        count_key = "counts"

        @classmethod
        def from_nparray(self, data, order='F'):
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

                return {self.size_key: _size, self.count_key: return_RLE}

            else:
                _error_message.variable("RLE.from_nparray", "None set in parameter 'data'")
                return None

        @classmethod
        def to_nparray(self, data, order='F'):
            if data is not None:
                rle_data = data[self.count_key]
                array = np.array([], dtype=np.uint8)
                for _type, _count in enumerate(rle_data):
                    value = _type % 2
                    if value:
                        _tmp = np.ones(_count, dtype=np.uint8)
                    else:
                        _tmp = np.zeros(_count, dtype=np.uint8)

                    array = np.concatenate((array, _tmp), axis=None, dtype=np.uint8)

                array = np.reshape(array, data[self.size_key], order)
                return array
            else:
                _error_message.variable("RLE.from_nparray", "None set in parameter 'data'")
                return None


class base():
    NP_type = {"uint8": np.uint8, "bool": np.bool8, "float32": np.float32}

    @staticmethod
    def get_array_from(sample, is_shape=False, value=0, dtype="uint8"):
        _array = np.ones(sample) * value if is_shape else np.array(sample)
        return base.type_converter(_array, dtype)

    @staticmethod
    def get_random_array(shape, range=[0, 1], norm_option=None, dtype="uint8"):
        if norm_option is not None:  # get normal random
            _array = norm_option[1] * np.random.randn(*shape) + norm_option[0]
        else:
            _array = np.random.rand(*shape)
            _term = max(range) - min(range)
            _min = min(range)

            _array = (_array * _term) + _min

        return base.type_converter(_array, dtype)

    @staticmethod
    def normalization(array):
        _m = np.mean(array)
        _std = np.std(array)

        return (base.type_converter(array, float) - _m) / _std

    @staticmethod
    def type_converter(data, to_type):
        if to_type in ["uint", "uint8", np.uint8]:
            _convert = data.astype(np.uint8)
        elif to_type in ["int32", "int", int]:
            _convert = data.astype(np.int32)
        elif to_type in ["int64", ]:
            _convert = data.astype(np.int64)
        elif to_type in ["float32", "float", float]:
            _convert = data.astype(np.float32)
        elif to_type in ["float64", ]:
            _convert = data.astype(np.float64)
        elif to_type in ["bool", bool]:
            _term = max(data) - min(data)
            _convert = np.round(data / _term)
            _convert = _convert.astype(np.bool8)
        return _convert

    @staticmethod
    def value_cut(array, value_min, value_max):
        return np.clip(array, value_min, value_max)

    @staticmethod
    def range_cut(array, cut_range=[-1, 1], direction="outside", ouput_type="active_map"):
        under_thread = min(cut_range)
        over_thread = max(cut_range)

        if direction == "outside":
            # --here-- under_thread --nope!-- over_thread --here--
            under_cuted = (array <= under_thread)
            over_cuted = (array >= over_thread)

            output = np.logical_or(under_cuted, over_cuted)
        else:
            # --nope!-- under_thread --here-- over_thread --nope!--
            under_cuted = (array >= under_thread)
            over_cuted = (array <= over_thread)

            output = np.logical_and(under_cuted, over_cuted)

        return output if ouput_type == "active_map" else array * base.type_converter(output, int)

    @staticmethod
    def stack(data_list: list, channel=-1):
        return np.stack(data_list, axis=channel)

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


class cal():
    @staticmethod
    def sqrt():
        pass


class image():
    @staticmethod
    def conver_to_last_channel(image):
        img_shape = image.shape
        if len(img_shape) == 2:
            # gray iamge
            return image[:, :, np.newaxis]
        else:
            # else image
            divide_data = [image[ct] for ct in range(img_shape[0])]
            return base.stack(divide_data)

    @staticmethod
    def conver_to_first_channel(image):
        img_shape = image.shape
        if len(img_shape) == 2:
            # gray iamge
            return image[np.newaxis, :, :]
        else:
            # else image
            divide_data = [image[:, :, ct] for ct in range(img_shape[-1])]
            return base.stack(divide_data, 0)

    # classfication -> (h, w, class count)
    # class map     -> (h, w)
    # color map     -> (h, w, 3)
    @staticmethod
    def classfication_to_class_map(classfication):
        # classfication(h, w, class count) -> class map(h, w, 1)
        classfication = image.conver_to_first_channel(classfication)
        return np.argmax(classfication, axis=0)

    @staticmethod
    def classfication_to_color_map(classfication, color_map):
        pass

    @staticmethod
    def class_map_to_classfication(class_map):
        pass

    @staticmethod
    def class_map_to_color_map(class_map, color_list):
        # class map(h, w, 1) -> color map(h, w, 3)
        # if seted "N colors" in "one ID", pick first color in list
        color_list = np.array([x[0] for x in color_list])
        return color_list[class_map].astype(np.uint8)

    @staticmethod
    def color_map_to_classfication(color_map, color_list):
        # color map(h, w, 3) -> classfication(h, w, class count)
        # make empty classfication
        _h, _w, _ = color_map.shape
        _c = len(color_list)  # color list -> [class 0 color, class 1 color, ... ignore color]
        classfication = base.get_array_from([_h, _w, _c], True)

        # color compare
        for _id in range(_c - 1):  # last channel : ignore
            _holder = []  # if seted "N colors" in "one ID", check all color
            for _color in color_list[_id]:
                _holder.append(np.all((color_map == _color), 2))

            classfication[:, :, _id] = _holder[0].astype(int) if len(_holder) == 1\
                else np.logical_or(*[data for data in _holder])

        # make ignore
        classfication[:, :, -1] = 1 - np.sum(classfication, axis=2).astype(bool).astype(int)
        return classfication

    @staticmethod
    def color_map_to_class_map(color_map, color_list):
        pass

    @staticmethod
    def classfication_resize(original, size):
        _new = base.get_array_from(size + [original.shape[-1], ], True)

        _pos = np.where(original == 1)
        _new_pos = []
        for _ct in range(len(_pos) - 1):
            _new_pos.append(np.round((size[_ct] - 1) * _pos[_ct] / original.shape[_ct]).astype(int))
        _new_pos.append(_pos[-1])

        _new[tuple(_new_pos)] = 1

        return _new

    @staticmethod
    def string_to_img(string, h, w, c):
        return np.fromstring(string, dtype=np.uint8).reshape((h, w, c))


class log():
    log_info: Dict[str, str] = {}
    log_holder: Dict[str, Dict] = {}

    def __init__(self, parameters: Dict[str, Dict]) -> None:
        """
        parameters: logging parameters
        """
        self.set_log_holder(parameters)

    def set_log_holder(self, logging_tree: Dict[str, Dict], root: dict = None):
        _root = self.log_holder if root is None else root

        for _node_name in logging_tree.keys():
            _under_nodes_names = logging_tree[_node_name].keys()
            if len(_under_nodes_names):  # _under_nodes exist
                """
                logging_root
                  L ...
                  L _logging_node => logging_tree[_node_name] !!! here !!!
                      L _under_nodes => _logging_node[_under_node_name]
                      L ...
                  L ...
                """
                _root[_node_name] = {}
                self.set_log_holder(logging_tree[_node_name], _root[_node_name])
            else:  # _under_nodes not exist
                """
                logging_root
                  L ...
                  L _node_name  !!! here !!!
                  L ...
                """
                _root[_node_name] = np.array([])

    def info_update(self, key: str, value: Any):
        self.log_info[key] = value

    def update(self, data: Dict, holder: Dict = None):
        _holder = self.log_holder if holder is None else holder

        # data -> dict; dict => Dict[str, dict or ndarray]
        for _node_name in data.keys():
            if _node_name in _holder.keys():
                _node_data = data[_node_name]
                if isinstance(_node_data, dict):  # go to under data node
                    self.update(_node_data, _holder[_node_name])
                else:  # data update
                    _holder[_node_name] = np.append(_holder[_node_name], np.array(_node_data))

    def save(self, save_dir, file_name="log.json"):
        save_pakage = {
            "info": self.log_info,
            "data": self.log_holder}
        file._json(save_dir, file_name, save_pakage, True)

    def load():
        pass

    def plot(self):
        pass


class evaluation():
    class IoU():
        def __init__(self, class_num):
            self.class_num = class_num

        def __call__(self, ouput, label):
            pass

        def get_iou(self):
            pass

        def get_miou(self):
            pass

    @staticmethod
    # in later fix it
    def iou(result, label, class_num):
        line_label = np.reshape(label, -1)
        line_result = np.reshape(result, -1)

        category_array_1d = base.bincount(line_label * class_num + line_result, class_num * class_num)
        category_array_2d = np.reshape(category_array_1d, (class_num, class_num))

        _inter = np.diag(category_array_2d)
        _uni = np.zeros(class_num)

        for _class_ct in range(class_num):
            _w_range = range(_class_ct * class_num, (_class_ct + 1) * class_num)
            _h_range = range(_class_ct, class_num * (class_num - 1) + _class_ct + 1, class_num)
            _uni[_class_ct] = \
                np.sum(category_array_1d[_w_range]) + np.sum(category_array_1d[_h_range]) - _inter[_class_ct]

        return _inter / _uni

    @staticmethod
    def miou(result, label, class_num):
        iou = evaluation.iou(result, label, class_num)
        return np.mean(iou)

    class baddeley():
        def __init__(self, p) -> None:
            self.p = p

        def __call__(self, image, target) -> float:
            self.get_value(image, target)

        def get_value(self, image, target):
            c = np.sqrt(target.shape * target.shape).item()
            N = target.shape[0] * target.shape[1]

            tager_edge_points = self.get_edge_points(target)
            image_edge_points = self.get_edge_points(image)

            # compare target
            compare_target_list = []
            compare_image_list = []
            for _h in range(image.shape[0]):
                for _w in range(image.shape[1]):
                    # get w(d(x, A))
                    compare_target_list.append(min(self.func_d([_h, _w], tager_edge_points), c))
                    # get w(d(x, B))
                    compare_image_list.append(min(self.func_d([_h, _w], image_edge_points), c))

            compare_target_list = np.array(compare_target_list)
            compare_image_list = np.array(compare_image_list)

            # cal |w(d(x, A))-w(d(x, B))|
            tmp_holder = np.abs(compare_target_list - compare_image_list)
            return (np.sum(np.power(tmp_holder, self.p)) / N) ** (1.0 / float(self.p))

        @staticmethod
        def get_edge_points(image):
            _h, _w = image.shape
            edge_point_holder = []
            for _h_ct in range(_h):
                for _w_ct in range(_w):
                    if image[_h_ct, _w_ct]:
                        edge_point_holder.append([_h_ct, _w_ct])
            return np.array(edge_point_holder)

        @staticmethod
        def func_d(position, edge_points):
            interval_list = np.abs(edge_points - position)
            min_interval = interval_list[np.argmin(np.sum(interval_list, 1))]

            return np.sqrt(np.sum(min_interval * min_interval)).item()

    class Confustion_Matrix():
        def Calculate_Confusion_Matrix(array: np.ndarray, target: np.ndarray, interest: np.ndarray = None) -> list:
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
            compare = (array == target).astype(np.uint8)

            compare_255 = (compare * 254)  # collect is 254, not 0 -> Tx
            inv_compare_255 = ((1 - compare) * 254)  # wrong is 254, not 0 -> Fx

            tp = np.logical_and(compare_255, target.astype(bool))       # collect_FG
            tn = np.logical_and(compare_255, ~(target.astype(bool)))    # collect_BG
            fn = np.logical_and(inv_compare_255, target.astype(bool))     # wrong_BG
            fp = np.logical_and(inv_compare_255, ~(target.astype(bool)))  # wrong_FG

            return (tp * interest, tn * interest, fn * interest, fp * interest)

        def Confusion_Matrix_to_value(TP, TN, FN, FP):
            pre = TP / (TP + FP) if TP + FP else TP / (TP + FP + 0.00001)
            re = TP / (TP + FN) if TP + FN else TP / (TP + FN + 0.00001)
            fm = (2 * pre * re) / (pre + re) if pre + re else (2 * pre * re) / (pre + re + 0.00001)

            return pre, re, fm


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