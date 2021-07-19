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


class C_position(Enum):
    Last = True
    First = False


class R_option(Enum):
    ZtoM = 0  # [0, 255]
    ZtoO = 1  # [0, 1.0]


# extention about read and write
class cv2_RnW():
    # option
    DEBUG = False

    IMAGE_EXT = ["jpg", "png", "bmp", ".jpg", ".png", ".bmp"]
    VIDEO_EXT = ["mp4", "avi", ".mp4", ".avi"]

    @classmethod
    def image_read(self, filename: str, color_option: Color_option):
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
    def channel_converter(image, channel_option: C_position):
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

    @classmethod
    def range_converter(self, image, form_range: R_option, to_range: R_option):
        if form_range == R_option.ZtoO:
            if to_range == R_option.ZtoM:  # convert to [0.0, 1.0] -> [0, 255]
                return _numpy.base_process.type_converter(image * self.M, "uint8")
            else:
                return image
        elif form_range == R_option.ZtoM:
            if to_range == R_option.ZtoO:  # convert to [0, 255] -> [0.0, 1.0]
                return image / 255
            else:
                return image

    @classmethod
    def canny(self, gray_image, ths, k_size=3, range=R_option.ZtoM, channel=C_position.Last):
        _high = ths[0]
        _low = ths[1]

        canny_image = cv2.Canny(gray_image, _low, _high, k_size)  # [h, w]
        canny_image = _numpy.image_extention.stack_image(
            [canny_image, canny_image, canny_image],
            channel.value)

        return self.range_converter(canny_image, R_option.ZtoM, range)


class gui_process():
    @staticmethod
    def display(image, dispaly_window):
        pass

    def image_cover():
        pass

    @staticmethod
    def write_text(image, text):
        pass

    def draw_rectangle():
        pass

    def draw_circle():
        pass

    def draw_oval():
        pass

    def draw_polygon(image, pts, thick, color):
        if thick == -1:
            return cv2.fillConvexPoly(image, pts, color)


# extention display with control
class trackbar():
    pass


class image_process():
    pass


class trackbar_window():
    pass


class augmentation():
    @staticmethod
    def make_noise():
        pass

    def rotate(img, center_rate, angle, scale):
        pass

    def random_crop():
        pass

    def flip():
        pass

    def image_augmentation():
        pass


def load_success():
    print("!!! custom python module ais_utils _cv2 load Success !!!")
