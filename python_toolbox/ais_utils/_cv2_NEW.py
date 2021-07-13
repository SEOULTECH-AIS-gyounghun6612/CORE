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
from enum import Enum
from . import _base
from . import _error
from . import _numpy

_error_message = _error.Custom_error("AIS_utils", "_cv2")


class Color_option(Enum):
    BGR = 0
    RGB = 1
    BGRA = 2
    GRAY = 3


class Channel_position(Enum):
    Last = True
    First = False


class Range_option(Enum):
    ZtoM = 0  # [0, 255]
    ZtoO = 1  # [0, 1.0]


# extention about read and write
class cv2_RnW():
    # option
    DEBUG = False

    IMAGE_EXT = ["jpg", "png", "bmp", ".jpg", ".png", ".bmp"]
    VIDEO_EXT = ["mp4", "avi", ".mp4", ".avi"]

    @classmethod
    def image_read(self, filename: str, color_option: Channel_position):
        if not _base.directory.exist_check(filename):
            _error_message.variable_stop(
                "cv2_RnW.image_read",
                "Have some problem in parameter 'filename'. Not exist")
        # data read
        if color_option == Color_option.BGR:
            _read_img = cv2.imread(filename, cv2.IMREAD_COLOR)
        elif color_option == Color_option.GRAY:
            _read_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        elif color_option == Color_option.RGB:
            _read_img = cv2.imread(filename, cv2.IMREAD_COLOR)
            _read_img = cv2.cvtColor(_read_img, cv2.COLOR_BGR2RGB)
        elif color_option == Color_option.BGRA:
            # this code dosen't check it. if you wnat use it. check it output
            _read_img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        # read debug
        if self.DEBUG:
            # if debug option is true, display the image in test window and program will be stop.
            cv2.imshow("image_read", _read_img)
            cv2.waitKey(0)

    @classmethod
    def image_write(self, filename: str, image):
        if not _base.directory.extention_check(filename, self.IMAGE_EXT):
            if self.DEBUG:
                _error_message.variable(
                    "cv2_RnW.image_write",
                    "Have some problem in parameter 'filename'. use default ext")
            filename += "jpg" if filename[-1] == "." else ".jpg"

        cv2.imwrite(filename, image)

    @staticmethod
    def video_capture(location, is_file=False):
        if is_file:
            if not _base.directory.exist_check(location):
                _error_message.variable_stop(
                    "cv2_RnW.image_read",
                    "Have some problem in parameter 'location'. Not exist")
        cap = cv2.VideoCapture(location)
        return cap

    @classmethod
    def video_write(self, filename, video_size, frame=30):
        video_format = filename.split("/")[-1].split(".")[-1]

        if not _base.directory.extention_check(video_format, self.VIDEO_EXT):
            if self.DEBUG:
                _error_message.variable(
                    "cv2_RnW.video_write",
                    "Have some problem in parameter 'filename'. use default ext")
            video_format = "avi"
            filename += "avi" if filename[-1] == "." else ".avi"

        _h, _w = video_size[:2]

        if video_format == "avi":
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        elif video_format == "mp4":
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        return cv2.VideoWriter(filename, fourcc, frame, (_w, _h))


# extention about image process
class base_process():
    M = 255

    @staticmethod
    def channel_converter(image, channel_option: Channel_position):
        if channel_option.value:  # [w, h, c]
            return _numpy.image_extention.conver_to_last_channel(image)
        else:  # [c, w, h]
            return _numpy.image_extention.conver_to_first_channel(image)

    @staticmethod
    def resize(image, size: list):
        _h, _w = image.shape[:2]
        _interpolation = cv2.INTER_AREA

        # ratiol
        if type(size[0]) == float and type(size[1]) == float:
            if size[0] >= 1.0 or size[1] >= 1.0:
                _interpolation = cv2.INTER_LINEAR
            return cv2.resize(image, dsize=[0, 0], fx=size[1], fy=size[0], interpolation=_interpolation)

        # absolute
        elif type(size[0]) == int and type(size[1]) == int:
            if size[0] >= _w or size[1] >= _h:
                _interpolation = cv2.INTER_LINEAR
            return cv2.resize(image, dsize=(size[1], size[0]), interpolation=_interpolation)

        else:
            _error_message.variable(
                "base_process.img_resize",
                "Have some problem in parameter 'size'\n" + "Function return input image")
            return image

    @staticmethod
    def range_converter(image, form_range: Range_option, to_range: Range_option):
        if form_range == Range_option.ZtoO:
            if to_range == Range_option.ZtoM:  # convert to [0.0, 1.0] -> [0, 255]
                
                return image
            else:
                return image
        elif form_range == Range_option.ZtoM:
            if to_range == Range_option.ZtoO:  # convert to [0, 255] -> [0.0, 1.0]

                return image
            else:
                return image


    @classmethod
    def canny(
            self,
            image,
            thresholds,
            kernel_size,
            input_range=Range_option.ZtoM,
            input_channel_option=Channel_position.Last):
        pass


# extention display with control
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
        is_z2o, is_First, input_text = self.doing_process(parameters)
        return {
            "img": self.processed_img,
            "window_name": _name,
            "is_zero2one": is_z2o,
            "is_First_channel": is_First,
            "input_text": input_text,
            "save_dir": None,
            "is_DEBUG": False
        }

    def doing_process(self, parameters):
        print("!!! Process is not set !!!")
        return False, False, None


class trackbar_window():
    def __init__(
            self,
            window_name: str or list,
            trackbars: list,
            image_process: list,):
        """
        Args:
            window_name :
            trackbars : trackbar's setting list. this list consist of trackbar class
            image_process : np.uint8 ndarray
        Returns:
            Empty
        """
        self.name = window_name if type(window_name) == list else [window_name, ]
        self.trackbar_list = trackbars
        self.process = image_process if type(image_process) == list else [image_process, ]

        # display original image
        cv2.namedWindow(ORIGINAL_WINDOW_NAME)
        self._set_trackbar()
        original_render_dict = self.process[0].return_original_image()
        img_render(**original_render_dict)

        # display processed image
        for _prs_ct, _prs in enumerate(self.process):
            # make window
            cv2.namedWindow(self.name[_prs_ct])
            parameters = self._get_parameters()

            processed_render_dict = _prs.return_processed_image(
                self.name[_prs_ct],
                parameters)
            img_render(**processed_render_dict)

    def _set_trackbar(self):
        def nothing(pos):
            pass
        # make window n track bar
        for trackbar in self.trackbar_list:
            _value_range = trackbar._range
            cv2.createTrackbar(
                trackbar._name,
                ORIGINAL_WINDOW_NAME,
                _value_range[0],
                _value_range[1],
                nothing)
            cv2.setTrackbarPos(
                trackbar._name,
                ORIGINAL_WINDOW_NAME,
                int(np.mean(_value_range)))

    def _get_parameters(self):
        _parameters = []
        for trackbar in self.trackbar_list:
            _parameters.append(cv2.getTrackbarPos(trackbar._name, ORIGINAL_WINDOW_NAME))
        return _parameters

    def loop(self):
        output = {
            "original": self.process[0].return_original_image()
        }
        while(True):
            for _prs_ct, _prs in enumerate(self.process):
                parameters = self._get_parameters()
                processed_render_dict = \
                    _prs.return_processed_image(self.name[_prs_ct], parameters)

                img_render(**processed_render_dict)
                output["prs_{}".format(_prs_ct)] = processed_render_dict

            _event = cv2.waitKeyEx(10)
            if _event == ord('q'):  # loop break
                break
            elif _event == ord('s'):
                break

        return _event, output


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

PIXEL_MAX = 255
# FUNCTION
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


"""
Custom function about image process
=====
"""
# CONSTANT
FLIP_OPTION = ["V", "H", "VH", "HV", "v", "h", "hv", "vh", None]
SALT_N_PAPPER = 0
INVERS = 1


def make_canny(
        img: np.ndarray,
        high: int,
        low: int,
        fillter_size: int = None,
        is_First_ch=True,
        is_zero2one: bool = True) -> np.ndarray:
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

    if fillter_size is not None:
        _img = cv2.blur(_img, (fillter_size, fillter_size))
        cany_img = cv2.Canny(_img, high, low, fillter_size)
    else:
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


def write_text_in(img, input_text, option="B", position=None, text_color=(0, 0, 0)):
    [_text_w, _text_h], _ = \
        cv2.getTextSize(text=input_text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=2)

    _h, _w, _c = np.shape(img)

    # make text image
    _text_shell =\
        255 * np.ones(
            (_text_h + (2 * IMAGE_TEXT_PADDING), _text_w + (2 * IMAGE_TEXT_PADDING), _c), dtype=np.uint8)

    if all(text_color, (255, 255, 255)):
        text_color = (128, 128, 128)

    cv2.putText(img=_text_shell,
                text=input_text,
                org=(IMAGE_TEXT_PADDING, _text_h + IMAGE_TEXT_PADDING),
                fontScale=1,
                thickness=2,
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                color=text_color)

    if (option != "I" or option != "i") and position is None:
        _text_shell = img_resize(_text_shell, [_text_h, _w])

        if option != "T" or option != "t":
            # combined save image n text image
            _render_image = 255 * np.ones((_h + _text_h, _w, _c), dtype=np.uint8)
            _render_image[:_text_h, :, :] = _text_shell
            _render_image[_text_h:, :, :] = img
            return _render_image

        else:
            # combined save image n text image
            _render_image = 255 * np.ones((_h + _text_h, _w, _c), dtype=np.uint8)
            _render_image[:_h, :, :] = img
            _render_image[_h:, :, :] = _text_shell
            return _render_image

    else:
        if _text_w >= 80:
            _text_w = 80

        if position[0] - _text_h < 0:
            position[0] = _text_h
        if position[0] > _h:
            position[0] = _h

        if position[1] + _text_w < _w:
            position[1] = _w - _text_w
        if position[1] < 0:
            position[1] = 0

        _text_shell = img_resize(_text_shell, [_text_h, _text_w])
        _mask = _text_shell == np.array((256, 256, 256), np.uint8)

        _base = img[position[0]: position[0] + _text_h, position[1]: position[1] + _text_w, :]

        _base = _base * _mask + _text_shell * (1 - _mask)

        img[position[0]: position[0] + _text_h, position[1]: position[1] + _text_w, :] = _base

        return img


def image_mixing(base, mask, cover_rate):
    _processed_base = ((1 - cover_rate) * base).astype(np.uint8)
    _processed_mask = (cover_rate * mask).astype(np.uint8)

    return _processed_base + _processed_mask


"""
Custom function about data type process
=====
"""


def custom_channel_compress(img: np.ndarray, color_type: int):
    """
    Args:
        img :
        color_type :
    Returns:
        compressed_data
    """
    if color_type == COLOR_GRAY or color_type == COLOR_BGRA:
        _e.Custom_Variable_Error(
            loacation="ais_utils._cv2.custom_channel_compress",
            parameters="color_type",
            detail="this function work for 3 channel data")
    else:
        _w, _h, _c = np.shape(img)
        compressed_data = np.zeros((_w, _h), dtype=np.float32)

        for _ct_c in range(_c):
            compressed_data += img[:, :, _ct_c] / (256 ** _ct_c)

        return compressed_data


def custom_channel_extract(compressed_data: np.ndarray, to_color_type: int):
    """
    Args:
        compressed_data :
        to_color_type :
    Returns:
        extracted_img
    """
    if to_color_type == COLOR_GRAY or to_color_type == COLOR_BGRA:
        _e.Custom_Variable_Error(
            loacation="ais_utils._cv2.custom_channel_extract",
            parameters="to_color_type",
            detail="this function work to 3 channel data")
    else:
        _w, _h = np.shape(compressed_data)
        _restore = np.zeros((_w, _h, 3), dtype=np.uint8)

        for _ct_c in range(3):
            _restore[:, :, _ct_c] = (compressed_data // 1.0).astype(np.uint8)
            compressed_data = (compressed_data - _restore[:, :, _ct_c]) * 256

        return _restore


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
    print("!!! custom python module ais_utils _cv2 load Success !!!")
