import numpy as np


def array2RLE(data, order='F'):
    if data is not None:
        _size = data.shape
        _size = (int(_size[0]), int(_size[1]))

        return_RLE = []
        if (data == np.zeros_like(data)).all():
            return_RLE.append(_size[0] * _size[1])
            return {"size": _size, "counts": return_RLE}
        elif (data == np.ones_like(data)).all():
            return_RLE.append(0)
            return_RLE.append(_size[0] * _size[1])
            return {"size": _size, "counts": return_RLE}
        else:
            _line = data.reshape(_size[0] * _size[1], order=order)
            _count_list = []

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


def RLE2array(data, order='F'):
    if data is not None:
        rle_data = data["counts"]
        array = np.array([], dtype=np.uint8)
        for _type, _count in enumerate(rle_data):
            value = _type % 2
            if value:
                _tmp = np.ones(_count, dtype=np.uint8)
            else:
                _tmp = np.zeros(_count, dtype=np.uint8)

            array = np.concatenate((array, _tmp), axis=None, dtype=np.uint8)

        array = np.reshape(array, data["size"], order)
        return array
    else:
        return None


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


def load_success():
    print("!!! custom python module ais_utils _numpy load Success !!!")
