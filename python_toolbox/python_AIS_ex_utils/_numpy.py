import numpy as np
if __package__ == "":
    import _error as _e
else:
    from . import _error as _e

_error_message = _e.Custom_error("AIS_utils", "_numpy")


class file():
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


class base_process():
    @staticmethod
    def type_converter(data, to_type):
        if to_type == "uint8":
            return data.astype(np.uint8)

    @staticmethod
    def stack(data_list: list, channel=-1):
        return np.stack(data_list, axis=channel)

    @staticmethod
    def bincount(array_1D, max_length):
        array_1D = np.round(array_1D)
        holder = np.bincount(array_1D)

        if len(holder) < max_length:
            count_list = np.zeros(max_length)
            count_list[:len(holder)] = holder
        else:
            count_list = holder[:max_length]

        return count_list

    @staticmethod
    def string_to_img(string, h, w, c):
        return np.fromstring(string, dtype=np.uint8).reshape((h, w, c))


class operation():
    @staticmethod
    def sqrt():
        pass

    @staticmethod
    def normal_cut(array, over_cut=1, under_cut=1, direction="outside", output_type="same"):
        _m = np.mean(array)
        _std = np.std(array)

        normal = ((array / 1.) - _m) / _std

        under_standard = -under_cut if under_cut >= 0 else 0
        over_standard = over_cut if over_cut >= 0 else 0

        if direction == "outside":
            under_cuted = (normal <= under_standard)
            over_cuted = (normal >= over_standard)

            output = under_cuted + over_cuted
        else:
            under_cuted = (normal >= under_standard)
            over_cuted = (normal <= over_standard)

            output = under_cuted * over_cuted

        return output.astype(array.dtype) if output_type == "same" else output


class image_extention():
    @staticmethod
    def get_canvus(size, sample=None, background_color=0):
        canvus = np.ones(size) if sample is None else np.ones_like(sample)
        if background_color in ["black", 0, [0, 0, 0]]:
            return (canvus * 0).astype(np.uint8)
        elif background_color in ["white", 255, [255, 255, 255]]:
            return (canvus * 255).astype(np.uint8)
        else:
            return (canvus * background_color).astype(np.uint8)

    @staticmethod
    def conver_to_last_channel(image):
        img_shape = image.shape
        if len(img_shape) == 2:
            # gray iamge
            return image[:, :, np.newaxis]
        else:
            # else image
            divide_data = [image[ct] for ct in range(img_shape[0])]
            return base_process.stack(divide_data)

    @staticmethod
    def conver_to_first_channel(image):
        img_shape = image.shape
        if len(img_shape) == 2:
            # gray iamge
            return image[np.newaxis, :, :]
        else:
            # else image
            divide_data = [image[:, :, ct] for ct in range(img_shape[-1])]
            return base_process.stack(divide_data, 0)

    @staticmethod
    def poly_points(pts):
        return np.round(pts).astype(np.int32)

    # classfication -> (h, w, class count)
    # class map     -> (h, w)
    # color map     -> (h, w, 3)
    @staticmethod
    def classfication_to_class_map(classfication):
        # classfication(h, w, class count) -> class map(h, w, 1)
        classfication = image_extention.conver_to_first_channel(classfication)
        return np.argmax(classfication, axis=0)

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
        classfication = image_extention.get_canvus([_h, _w, _c])

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
    def classfication_resize(original, size):
        _pos = np.where(original == 1)
        _new_pos = [np.round((size[_ct] - 1) * _pos[_ct] / original.shape[_ct]).astype(int) for _ct in range(2)]

        _new = image_extention.get_canvus(size + [original.shape[-1], ])
        _new[_new_pos[0], _new_pos[1], _pos[-1]] = 1

        return _new

    # @staticmethod
    # def classfication_to_class_map(classfication, is_last_ch=False):
    #     if 2 == len(classfication.shape):
    #         _max = classfication.max() + 1
    #         class_map = np.array([[np.eye(_max, _m)[0] for _m in _w] for _w in classfication], dtype=int)
    #         if not is_last_ch:
    #             class_map = image_extention.conver_to_first_channel(class_map)
    #         return class_map

    #     else:
    #         return classfication


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


class evaluation():
    @staticmethod
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

        category_array_1d = base_process.bincount(line_label * class_num + line_result, class_num * class_num)
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

    @staticmethod
    class baddeley():
        def __call__(self, image, target, p) -> float:
            self.get_value(image, target, p)

        def get_value(self, image, target, p):
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
            return (np.sum(np.power(tmp_holder, p)) / N) ** (1.0 / float(p))

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


def load_check():
    print("!!! custom python module ais_utils _numpy load Success !!!")
