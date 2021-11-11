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
import random

from enum import Enum

if __package__ == "":
    import _base
    import _numpy
    import _error as _e

else:
    from . import _base
    from . import _numpy
    from . import _error as _e

_error_message = _e.Custom_error("AIS_utils", "_cv2")


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


class CVT_option(Enum):
    GRAY2BGR = 0
    BGR2GRAY = 1
    BGR2RGB = 2


# extention about read and write
class file():
    # option
    DEBUG = False

    IMAGE_EXT = ["jpg", "png", "bmp", ".jpg", ".png", ".bmp"]
    VIDEO_EXT = ["mp4", "avi", ".mp4", ".avi"]

    @classmethod
    def image_read(self, filename: str, color_option: Color_option):
        if not _base.directory._exist_check(filename, is_file=True):
            _error_message.variable_stop(
                "file.image_read",
                ["filename", ],
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

        return _read_img

    @classmethod
    def image_write(self, file_dir: str, image):
        filename = _base.file._name_from_directory(file_dir)
        if not _base.file._extension_check(filename, self.IMAGE_EXT):
            if self.DEBUG:
                _error_message.variable(
                    "file.image_write",
                    "Have some problem in parameter 'filename'. use default ext")
            file_dir += "jpg" if file_dir[-1] == "." else ".jpg"

        cv2.imwrite(file_dir, image)

    @staticmethod
    def video_capture(location, is_file=False):
        if is_file:
            if not _base.directory._exist_check(location):
                _error_message.variable_stop(
                    "file.image_read",
                    "Have some problem in parameter 'location'. Not exist")
        cap = cv2.VideoCapture(location)
        return cap

    @classmethod  # in later fix
    def video_write(self, filename, video_size, frame=30):
        video_format = filename.split("/")[-1].split(".")[-1]

        if not _base.file._extension_check(video_format, self.VIDEO_EXT):
            if self.DEBUG:
                _error_message.variable(
                    "file.video_write",
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

    @staticmethod
    def img_cvt(img, cvt_option: CVT_option):
        if cvt_option == CVT_option.GRAY2BGR:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif cvt_option == CVT_option.BGR2GRAY:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif cvt_option == CVT_option.BGR2RGB:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def filtering(image, array):
        if len(image.shape) > 2:
            # color image
            holder = _numpy.image_extention.get_canvus(size=None, sample=image)
            for _ch_ct in range(image.shape[-1]):
                holder[:, :, _ch_ct] = base_process.filtering(image[:, :, _ch_ct], array)
            return holder
        else:
            return cv2.filter2D(image, -1, array)

    @staticmethod
    class blur():
        default = {
            "gaussian": {},
            "bilateral": {
                "d": -1,
                "sigmaColor": 5,
                "sigmaSpace": 5}}

        def __call__(self, image, style):
            if style == "gaussian":
                pass
            elif style == "bilateral":
                return cv2.bilateralFilter(image, **self.default[style])


class edge_process():
    @staticmethod
    def sobel(image, threshold=1):
        if len(image.shape) > 2:
            # color image
            holder = _numpy.image_extention.get_canvus(size=image.shape[:2])
            for _ch_ct in range(image.shape[-1]):
                holder += edge_process.sobel(image[:, :, _ch_ct])
            return _numpy.base_process.type_converter(holder >= 2, "uint8")
        else:
            dx = cv2.Sobel(image, -1, 1, 0, delta=128)
            dy = cv2.Sobel(image, -1, 0, 1, delta=128)

            return _numpy.operation.normal_cut(cv2.addWeighted(dx, 0.5, dy, 0.5, 0), threshold, threshold)

    @staticmethod
    def canny(gray_image, ths, k_size=3, range=R_option.ZtoM, channel=C_position.Last):
        _high = ths[0]
        _low = ths[1]

        canny_image = cv2.Canny(gray_image, _low, _high, k_size)  # [h, w]
        canny_image = _numpy.image_extention.stack_image(
            [canny_image, canny_image, canny_image],
            channel.value)

        return base_process.range_converter(canny_image, R_option.ZtoM, range)


class gui_process():
    @staticmethod
    def display(image, dispaly_window):
        pass

    def image_cover():
        pass

    # extention display with control
    class trackbar():
        pass

    class image_process():
        pass

    class trackbar_window():
        pass


class draw():
    @staticmethod
    def _padding(image, padding):
        # make holder -> in later add multi padding option
        if isinstance(padding, int):
            _t_pad, _b_pad, _l_pad, _r_pad = [padding, padding, padding, padding]
        elif isinstance(padding, list):
            if len(padding) == 2:
                # [hight_pad, weidth_pad]
                pass
            elif len(padding) == 4:
                # [top, bottom, left, right]
                pass

        if len(image.shape) == 3:
            _h, _w, _c = image.shape
            _holder_shape = [_h + (_t_pad + _b_pad), _w + (_l_pad + _r_pad), _c]
        elif len(image.shape) == 2:
            _h, _w = image.shape
            _holder_shape = [_h + (_t_pad + _b_pad), _w + (_l_pad + _r_pad)]

        _holder = _numpy.image_extention.get_canvus(_holder_shape)
        _holder[_t_pad: -_b_pad, _l_pad: -_r_pad] = image

        return _holder

    @staticmethod
    def _unpadding(image, padding):
        # make holder -> in later add multi padding option
        if isinstance(padding, int):
            _t_pad, _b_pad, _l_pad, _r_pad = [padding, padding, padding, padding]
        elif isinstance(padding, list):
            if len(padding) == 2:
                # [hight_pad, weidth_pad]
                pass
            elif len(padding) == 4:
                # [top, bottom, left, right]
                pass

        return image[_t_pad: -_b_pad, _l_pad: -_r_pad]

    @staticmethod
    def _text(image, text):
        pass

    @staticmethod
    def _rectangle():
        pass

    @staticmethod
    def _circle():
        pass

    @staticmethod
    def _oval():
        pass

    @staticmethod
    def _polygon(image, pts, thick, color, is_close=True):
        pts = _numpy.image_extention.poly_points(pts)
        if thick == -1:
            return cv2.fillConvexPoly(image, pts, color)
        else:
            return cv2.polylines(image, pts, is_close, color, thick)

    @staticmethod
    class canvas():
        figures = None

        draw_object = []
        past_points = []  # [x, y]
        this_point = []  # [x, y]

        draw_pen = {
            "base_color": [0x00, 0x00, 0x00],
            "draw_color": [0x00, 0x00, 0x00],
            "thickness": 3,
            "style": "solid_line"}

        def clear_canvas(self, size, smaple=None, is_color="white"):
            self.figures = _numpy.image_extention.get_canvus(size, smaple, is_color)

        def set_draw_pen(self, base, color, thickness, style):
            self.draw_pen["base_color"] = base
            self.draw_pen["draw_color"] = color
            self.draw_pen["thickness"] = thickness
            self.draw_pen["style"] = style

        def set_object(self, num):
            pass

        def del_object(self, num):
            pass

        def clear_object(self):
            pass

        def draw(self):
            pass


class augmentation():
    def __call__(self, **option):
        pass

    @staticmethod
    def _make_noise():
        pass

    @staticmethod
    def _rotate(img, center_rate, angle, scale):
        pass

    @staticmethod
    def _crop(imgs, size, left_top=[None, None], is_last_ch_imgs=True):
        if is_last_ch_imgs:
            h, w, _ = imgs[0].shape
        else:
            _, h, w = imgs[0].shape

        crop_imgs = []

        LT_h = random.randrange(h - size[0]) if left_top[0] is None else left_top[0]
        LT_W = random.randrange(w - size[1]) if left_top[1] is None else left_top[1]

        for _img in imgs:
            crop_imgs.append(
                _img[LT_h: LT_h + size[0], LT_W: LT_W + size[1]] if is_last_ch_imgs
                else _img[:, LT_h: LT_h + size[0], LT_W: LT_W + size[1]])

        return crop_imgs

    @staticmethod
    def _flip():
        pass

    def image_augmentation():
        pass


def load_check():
    print("!!! custom python module ais_utils _cv2 load Success !!!")
