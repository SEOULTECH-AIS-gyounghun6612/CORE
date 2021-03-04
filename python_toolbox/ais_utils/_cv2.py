"""
File object
=====
    When write down python program, be used cv2 custom function.

Requirement
=====
    cv2     (pip install python-opencv)
    numpy
"""

# Import module
import cv2
import numpy as np

from . import _error as _e
"""
Custom function about np module
=====
"""


def data_stack(data_list: list, is_last_axis: bool = True):
    """ Make merge data array from data_list
    Args:
        data_list :
        is_last_axis :
    Returns:
        return_data (like data_list type) : merge data array
    """
    return_data = np.dstack(data_list) if is_last_axis else np.array(data_list)
    return return_data


"""
Custom function about file W/R
=====
"""
# CONSTANT
COLOR_BGR = 0
COLOR_GRAY = 1
COLOR_BGRA = 2
COLOR_RGB = 3
PIXEL_MAX = 255

IMAGE_EXT = ["jpg", "png", "bmp", ".jpg", ".png", ".bmp"]
VIDEO_EXT = ["mp4", "avi", ".mp4", ".avi"]


# FUNCTION
def read_img(
        file_dir: str,
        color_type: int,
        resize: list = None,
        is_last_channel: bool = True,
        is_debug: bool = False):
    """
    Args:
        file_dir :
        color_type :
        resize :
        is_last_channel :
        is_debug :
    Returns:
        return (np.uint8 array): image data
    """
    # data read
    if color_type == COLOR_BGR:
        _read_img = cv2.imread(file_dir, cv2.IMREAD_COLOR)
    elif color_type == COLOR_GRAY:
        _read_img = cv2.imread(file_dir, cv2.IMREAD_GRAYSCALE)
    elif color_type == COLOR_RGB:
        _read_img = cv2.imread(file_dir, cv2.IMREAD_COLOR)
        _read_img = cv2.cvtColor(_read_img, cv2.COLOR_BGR2RGB)
    elif color_type == COLOR_BGRA:
        # this code dosen't check it. if you wnat use it. check it output
        _read_img = cv2.imread(file_dir, cv2.IMREAD_UNCHANGED)

    # do resize step
    if resize is not None:
        _read_img = img_resize(_read_img, resize)

    # read debug
    if is_debug:
        # if debug option is true, display the image in test window and program will be stop.
        cv2.imshow("reading img", _read_img)
        cv2.waitKey(0)

    if is_last_channel:
        # img => [w, h, c]
        return _read_img[:, :, np.newaxis] if color_type == COLOR_GRAY else _read_img
    else:
        # img => [c, w, h]
        return _read_img[np.newaxis, :, :] if color_type == COLOR_GRAY \
            else data_stack(
                data_list=[_read_img[:, :, 0], _read_img[:, :, 1], _read_img[:, :, 2]],
                is_last_axis=is_last_channel)


def read_video(file_dir):
    """
    Args:
        file_dir :
    Returns:
        return (np.uint8 array): image data
    """
    cap = cv2.VideoCapture(file_dir)
    return cap


def wirte_img_in(save_dir: str, img: np.ndarray, ext: str = None) -> None:
    """
    Args:
        save_dir :
        img : np.uint8 ndarray
        ext : if you want image ext.
              if you set Default, and save_dir has not ext, autometically set .png
    Returns:
        Empty
    """
    if ext is not None:
        assert IMAGE_EXT.index(ext)
        file_dir = save_dir + ext if ext[0] == "." else save_dir + "." + ext
    else:
        if IMAGE_EXT.index(save_dir[-4:]):
            file_dir = save_dir
        else:
            file_dir = save_dir + ".png"

    cv2.imwrite(file_dir, img)


def make_video(
        save_dir: str,
        imgs: np.ndarray,
        ext: str = None,
        frame: float = 30.0,
        shape: [int, int] = None) -> None:
    """
    Args:
        save_dir :
        img : np.uint8 ndarray
        ext : if you want image ext.
              if you set Default, and save_dir has not ext, autometically set .avi
        frame :
        shape :
    Returns:
        Empty
    """
    if shape is not None:
        h, w = shape
    else:
        h, w, _ = np.shape(imgs[0])
    if ext is not None:
        assert VIDEO_EXT.index(ext)
        _ext = ext
        file_dir = save_dir + ext if ext[0] == "." else save_dir + "." + ext
    else:
        if VIDEO_EXT.index(save_dir[-4:]):
            file_dir = save_dir
            _ext = save_dir[-3:]
        else:
            _ext = "avi"
            file_dir = save_dir + ".avi"

    if _ext == "avi":
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    elif _ext == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    video_file = cv2.VideoWriter(file_dir, fourcc, frame, (w, h))

    for _img in imgs:
        video_file.write(_img)
        cv2.imshow('frame', _img)
        cv2.waitKey(1)
    video_file.release()


def save_numpy(save_dir, data):
    """
    Args:
        save_dir :
        img : np.uint8 ndarray
        ext : if you want image ext.
              if you set Default, and save_dir has not ext, autometically set .avi
        frame :
        shape :
    Returns:
        Empty
    """
    if type(data) == list:
        _array = np.array(data)

    elif type(data) == np.ndarray:
        _array = data.copy()

    if save_dir.split("/")[-1].split(".")[-1] == "npz":
        np.savez_compressed(save_dir, data=_array)
    else:
        np.savez(save_dir, data=_array)


"""
Custom function about Debug
=====
"""
# CONSTANT
NEIGHBOR_SPACE = [[1, None], [0, None], [None, -1]]
NEIGHBOR_SPACE2 = [[None, -1], [1, -1], [1, None]]

IMAGE_TEXT_PADDING = 10
SMALL_VALUE = 1E-10

ORIGINAL_WINDOW_NAME = "original"


# FUNCTION
class trackbar():
    def __init__(self, name, value_range):
        """
        Args:
            name :
            value_range : np.uint8 ndarray
        Returns:
            Empty
        """
        self._name = name
        self._range = value_range


class image_process():
    def __init__(self, origianl_image):
        self.original_img = origianl_image
        self.processed_img = None

    def return_original_image(self):
        return {
            "img": self.original_img,
            "window_name": ORIGINAL_WINDOW_NAME,
            "is_zero2one": False,
            "is_First_channel": False,
            "input_text": None,
            "save_dir": None,
            "is_DEBUG": False
        }

    def return_processed_image(self, _name, parameters):
        is_z2o, is_First = self.doing_process(parameters)
        return {
            "img": self.processed_img,
            "window_name": _name,
            "is_zero2one": is_z2o,
            "is_First_channel": is_First,
            "input_text": None,
            "save_dir": None,
            "is_DEBUG": False
        }

    def doing_process(self, parameters):
        print("!!! Process is not set !!!")
        return False, False


class trackbar_window():
    def __init__(self, window_name: str, trackbars: list, image_process, display_origianl: bool = True):
        """
        Args:
            window_name :
            trackbars : trackbar's setting list. this list consist of trackbar class
            image_process : np.uint8 ndarray
            display_origianl :
        Returns:
            Empty
        """
        self.name = window_name
        self.trackbar_list = trackbars
        self.process = image_process

        cv2.namedWindow(self.name)
        self._set_trackbar()

        if display_origianl:
            # display original image
            cv2.namedWindow(ORIGINAL_WINDOW_NAME)
            original_render_dict = self.process.return_original_image()
            img_render(**original_render_dict)

        # display processed image
        parameters = self._get_parameters()
        processed_render_dict = self.process.return_processed_image(self.name, parameters)
        img_render(**processed_render_dict)

    def _set_trackbar(self):
        def nothing(pos):
            pass
        # make window n track bar
        for trackbar in self.trackbar_list:
            _value_range = trackbar._range
            cv2.createTrackbar(
                trackbar._name,
                self.name,
                _value_range[0],
                _value_range[1],
                nothing)
            cv2.setTrackbarPos(
                trackbar._name,
                self.name,
                int(np.mean(_value_range)))

    def _get_parameters(self):
        _parameters = []
        for trackbar in self.trackbar_list:
            _parameters.append(cv2.getTrackbarPos(trackbar._name, self.name))
        return _parameters

    def loop(self, save_dir):
        while(True):
            _event = cv2.waitKeyEx(10)
            parameters = self._get_parameters()
            processed_render_dict = self.process.return_processed_image(self.name, parameters)
            img_render(**processed_render_dict)
            if _event == ord('s'):  # image save
                processed_render_dict["save_dir"] = save_dir
                img_render(**processed_render_dict)
                break
            elif _event == ord('q'):  # loop break
                break

        return _event


def img_render(
        img: np.ndarray,
        window_name: str,
        is_zero2one: bool = True,
        is_First_channel: bool = False,
        input_text: str = None,
        save_dir: str = None,
        is_DEBUG: bool = False) -> None:
    """
    Args:
        img : np.uint8 ndarray
        window_name :
        is_zero2one :
        is_First_channel :
        input_text :
        save_dir :
        is_DEBUG :
    Returns:
        Empty
    """
    _img = img

    # process 1 :
    if is_zero2one:
        _img *= PIXEL_MAX
        _img = np.clip(_img, 0, PIXEL_MAX)
        _img = _img.astype(np.uint8)

    #  process 2 :
    if is_First_channel:
        # in torch, img data style is first channel. so, change to last channel
        _img = np.dstack([_img[0], _img[1], _img[2]]).astype(np.uint8)

    #  process 3 :
    if input_text is not None:
        [_text_w, _text_h], _ = cv2.getTextSize(
            text=input_text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=2)
        _h, _w, _c = np.shape(_img)

        # make text image
        _text_shell =\
            255 * np.ones(
                (_text_h + (2 * IMAGE_TEXT_PADDING), _text_w + (2 * IMAGE_TEXT_PADDING), _c), dtype=np.uint8)
        cv2.putText(img=_text_shell,
                    text=input_text,
                    org=(IMAGE_TEXT_PADDING, _text_h + IMAGE_TEXT_PADDING),
                    fontScale=1,
                    thickness=2,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    color=(0, 0, 0))
        _text_shell = img_resize(_text_shell, [_text_h, _w])

        # combined save image n text image
        _render_image = 255 * np.ones((_h + _text_h, _w, _c), dtype=np.uint8)
        _render_image[:_h, :, :] = _img
        _render_image[_h:, :, :] = _text_shell
        _img = _render_image

    #  process 4 :
    if save_dir is not None:
        wirte_img_in(save_dir, _img)

    #  process 5 :
    if window_name is not None:
        cv2.imshow(window_name, _img)
    cv2.waitKey(10)


def Neighbor_Confusion_Matrix(
        img: np.ndarray, target: np.ndarray, interest: np.ndarray) -> list:
    """
    Args:
        img :
        target : np.uint8 ndarray
        ext : if you want image ext.
              if you set Default, and save_dir has not ext, autometically set .avi
        frame :
        shape :
    Returns:
        Confusion Matrix (list)
    """
    _nb_tp = np.zeros_like(img[1: -1, 1: -1])
    _nb_tn = np.zeros_like(img[1: -1, 1: -1])
    _nb_fp = np.zeros_like(img[1: -1, 1: -1])
    _nb_fn = np.zeros_like(img[1: -1, 1: -1])

    for _h_c in range(3):
        image_h_s = NEIGHBOR_SPACE[_h_c][0]
        image_h_e = NEIGHBOR_SPACE[_h_c][1]
        target_h_s = NEIGHBOR_SPACE[2 - _h_c][0]
        target_h_e = NEIGHBOR_SPACE[2 - _h_c][1]
        roi_h_s = NEIGHBOR_SPACE2[_h_c][0]
        roi_h_e = NEIGHBOR_SPACE2[_h_c][1]

        for _w_c in range(3):
            image_w_s = NEIGHBOR_SPACE[_w_c][0]
            image_w_e = NEIGHBOR_SPACE[_w_c][1]
            target_w_s = NEIGHBOR_SPACE[2 - _w_c][0]
            target_w_e = NEIGHBOR_SPACE[2 - _w_c][1]
            roi_w_s = NEIGHBOR_SPACE2[_w_c][0]
            roi_w_e = NEIGHBOR_SPACE2[_w_c][1]

            cutted_image = img[image_h_s: image_h_e, image_w_s: image_w_e]
            cutted_interest = interest[image_h_s: image_h_e, image_w_s: image_w_e]
            cutted_target = target[target_h_s: target_h_e, target_w_s: target_w_e]

            _cm = Calculate_Confusion_Matrix(cutted_image, cutted_target, cutted_interest)

            _nb_tp = np.logical_or(_nb_tp, _cm[0][roi_h_s: roi_h_e, roi_w_s: roi_w_e])
            _nb_tn = np.logical_or(_nb_tn, _cm[1][roi_h_s: roi_h_e, roi_w_s: roi_w_e])
            _nb_fp = np.logical_or(_nb_fp, _cm[3][roi_h_s: roi_h_e, roi_w_s: roi_w_e])
            _nb_fn = np.logical_or(_nb_fn, _cm[2][roi_h_s: roi_h_e, roi_w_s: roi_w_e])

    _nb_fn = _nb_fn.astype(np.uint8) - np.logical_and(_nb_fn, _nb_tp).astype(np.uint8)
    _nb_fp = _nb_fp.astype(np.uint8) - np.logical_and(_nb_fp, _nb_tn).astype(np.uint8)

    output = []
    output.append(_nb_tp.astype(np.uint8))
    output.append(_nb_tn.astype(np.uint8))
    output.append(_nb_fn.astype(np.uint8))
    output.append(_nb_fp.astype(np.uint8))

    return output


def Calculate_Confusion_Matrix(image: np.ndarray, target: np.ndarray, interest: np.ndarray = None) -> list:
    """
    Args:
        img :
        target : np.uint8 ndarray
        ext : if you want image ext.
              if you set Default, and save_dir has not ext, autometically set .avi
        frame :
        shape :
    Returns:
        Confusion Matrix (list)
    """
    if interest is None:
        interest = np.ones_like(image, np.uint8)
    compare = (image == target).astype(np.uint8)

    compare_255 = (compare * 254)  # collect is 254, not 0 -> Tx
    inv_compare_255 = ((1 - compare) * 254)  # wrong is 254, not 0 -> Fx

    collect_FG = np.logical_and(compare_255, target.astype(bool))       # TP
    collect_BG = np.logical_and(compare_255, ~(target.astype(bool)))    # TN
    wrong_BG = np.logical_and(inv_compare_255, target.astype(bool))     # FN
    wrong_FG = np.logical_and(inv_compare_255, ~(target.astype(bool)))  # FP

    return (collect_FG * interest, collect_BG * interest, wrong_BG * interest, wrong_FG * interest)


def Confusion_Matrix_to_value(TP, TN, FN, FP):
    pre = TP / (TP + FP + SMALL_VALUE)
    recall = TP / (TP + FN + SMALL_VALUE)
    fm = (2 * pre * recall) / (pre + recall + SMALL_VALUE)

    return pre, recall, fm


"""
Custom function about image process
=====
"""
# CONSTANT
FLIP_OPTION = ["V", "H", "VH", "HV", "v", "h", "hv", "vh", None]
SALT_N_PAPPER = 0
INVERS = 1


# FUNCTION
def img_resize(img: np.ndarray, size: list):
    """
    Args:
        img :
        size : "[int, int] or [float, float]"
    Returns:
        Confusion Matrix (list)
    """
    try:
        h, w, _ = np.shape(img)
    except ValueError:
        h, w = np.shape(img)
    _interpolation = cv2.INTER_AREA
    if type(size[0]) == float and type(size[1]) == float:
        if size[0] >= 1.0 or size[1] >= 1.0:
            _interpolation = cv2.INTER_LINEAR

        return cv2.resize(img, dsize=[0, 0], fx=size[1], fy=size[0], interpolation=_interpolation)

    elif type(size[0]) == int and type(size[1]) == int:
        if size[0] >= w or size[1] >= h:
            _interpolation = cv2.INTER_LINEAR

        return cv2.resize(img, dsize=(size[1], size[0]), interpolation=_interpolation)

    else:
        print("setting error. in this situation, return input image.")
        return img


def make_canny(
        img: np.ndarray, high: int, low: int, is_First_ch=True, is_zero2one: bool = True) -> np.ndarray:
    """
    Args:
        img :
        size : "[int, int] or [float, float]"
    Returns:
        Confusion Matrix (list)
    """
    if is_First_ch:
        if np.shape(img)[0] == 3:
            _img = np.dstack([img[0], img[1], img[2]])
        else:
            _img = np.dstack([img[0], ])

    else:
        _img = img.copy()

    if is_zero2one:
        _img *= PIXEL_MAX
    _img = np.clip(_img, 0, PIXEL_MAX)
    _img = _img.astype(np.uint8)

    if len(np.shape(_img)) == 3:
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)

    cany_img = cv2.Canny(_img, high, low)
    cany_img = np.array([cany_img, cany_img, cany_img], np.uint8) if is_First_ch \
        else np.dstack([cany_img, cany_img, cany_img])
    cany_img = cany_img / 255 if is_zero2one else cany_img
    return cany_img


def make_noise_in_image(
        img: np.ndarray,
        noise_rate: float,
        noise_type: int = SALT_N_PAPPER,
        noise_parameter=None) -> np.ndarray:
    """
    Args:
        img :
        size : "[int, int] or [float, float]"
    Returns:
        Confusion Matrix (list)
    """
    try:
        _h, _w, _c = np.shape(img)
    except ValueError:
        _h, _w = np.shape(img)
        _c = False

    noise_rate = (1 - noise_rate)

    noise_img = np.zeros_like(img)
    if noise_type == SALT_N_PAPPER:
        # noise parameter setting
        if noise_parameter is None:
            _papper_n_salt_rate = 1 / 2
        else:
            _papper_n_salt_rate = (noise_parameter / (1 + noise_parameter))
        _salt_rate = noise_rate
        _papper_rate = noise_rate + (1 - noise_rate) * _papper_n_salt_rate

        # make noise mask
        _rate = np.random.rand(_h, _w)
        _salt_n_papper = (_rate >= _salt_rate).astype(np.uint8)
        _papper = (_rate >= _papper_rate).astype(np.uint8)
        _salt = _salt_n_papper - _papper

        # make noise image
        if _c:
            for _ct_c in range(_c):
                noise_img[:, :, _ct_c] = img[:, :, _ct_c] * (1 - _salt_n_papper) + 255 * _salt
        else:
            noise_img = img * (1 - _salt_n_papper) + 255 * _salt

        return noise_img

    elif noise_type == INVERS:
        # make noise mask
        _rate = np.random.rand(_h, _w)
        _noise_mask = (_rate >= noise_rate).astype(np.uint8)

        # make noise image
        if _c:
            for _ct_c in range(_c):
                _img_max = np.max(img[:, :, _ct_c])
                noise_img[:, :, _ct_c] =\
                    np.where(_noise_mask == 1, img[:, :, _ct_c], _img_max - img[:, :, _ct_c])
        else:
            _img_max = np.max(img)
            noise_img = np.where(_noise_mask, _img_max - img, img)

        return noise_img
    else:
        print("have some problem in noise type setting")


def image_rotate(img, center_rate, angle, scale):
    _h, _w, _c = np.shape(img)
    _matrix = cv2.getRotationMatrix2D((_h * center_rate[0], _w * center_rate[1]), angle, scale)

    return cv2.warpAffine(img, _matrix, (_h, _w))


def image_augmentation(
        file_list: list,
        resize: list = None,
        crop_size: list = None,
        flip: str = None,
        rotate: list = None,
        ext: str = ".png"):
    """
    Args:
        file_list :
        resize : "[int, int] or [float, float]"
        crop_size :
        flip :
        rotate :
        ext :
    Returns:
        Empty
    """
    flip_option_check = flip in FLIP_OPTION

    if not flip_option_check:
        _e.Custom_Variable_Error(
            loacation="ais_utils._cv2.image_augmentation",
            parameters="flip",
            detail="you set the the weird flip option letter")

    _img_list = []
    _file_list = []

    for file in file_list:
        _img = cv2.imread(file, cv2.IMREAD_COLOR)
        _img_list.append(_img)
        _file_list.append(file.split("/")[-1].split(".")[0])

    if resize is not None:
        for _ct in range(len(_img_list)):
            _img_list[_ct] = img_resize(_img_list[_ct], resize)
            _file_list[_ct] += "_{}-{}".format(resize[0], resize[1])

    if crop_size is not None:
        _h, _w, _ = np.shape(_img)
        _start_h = np.random.randint(_h - crop_size[0])
        _start_w = np.random.randint(_w - crop_size[1])
        for _ct in range(len(_img_list)):
            _img_list[_ct] = \
                _img_list[_ct][_start_h: _start_h + crop_size[0], _start_w:_start_w + crop_size[1], ]
            _file_list[_ct] += "_{}-{}-{}-{}".format(_start_h, _start_w, crop_size[0], crop_size[1])

    if flip is not None:
        if "v" == flip or "V" == flip:
            for _ct in range(len(_img_list)):
                _img_list[_ct] = cv2.flip(_img_list[_ct], 1)
                _file_list[_ct] += "_v"
        elif "h" == flip or "H" == flip:
            for _ct in range(len(_img_list)):
                _img_list[_ct] = cv2.flip(_img_list[_ct], 0)
                _file_list[_ct] += "_h"

        elif "hv" == flip or "HV" == flip or "vh" == flip or "VH" == flip:
            for _ct in range(len(_img_list)):
                _img_list[_ct] = cv2.flip(_img_list[_ct], 0)
                _img_list[_ct] = cv2.flip(_img_list[_ct], 1)
                _file_list[_ct] += "_hv"
    else:
        for _ct in range(len(_img_list)):
            _file_list[_ct] += "_o"

    if rotate is not None:
        _angle = np.random.randint(-rotate[1], rotate[1])
        for _ct in range(len(_img_list)):
            _img_list[_ct] = image_rotate(_img_list[_ct], rotate[0], _angle, rotate[2])
            _file_list[_ct] += "_{}-{}-{}-{}".format(rotate[0][0], rotate[0][1], _angle, rotate[2])

    for _ct in range(len(_img_list)):
        _file_list[_ct] += ext

    return [_img_list, _file_list]


# ##################################################### #
# ############ Custom function about trick ############ #
# ##################################################### #
# def float_data_save(file_dir: str, data: float):
#     """ Image file to data
#     Args:
#         data_list :
#         is_last_axis :
#     Returns:
#         return (np.uint8 array): image data
#     """
#     if type(data)
#     int_bits =


def load_success():
    print("!!! custom python module AIS_utils_cv2 load Success !!!")
