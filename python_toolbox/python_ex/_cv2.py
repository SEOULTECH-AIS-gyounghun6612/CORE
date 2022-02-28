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
from typing import List, Union
import cv2
import random
from dataclasses import dataclass, field

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
    ZtoFF = 0  # [0, 255]
    ZtoO = 1  # [0, 1.0]


class CVT_option(Enum):
    GRAY2BGR = 0
    BGR2GRAY = 1
    BGR2RGB = 2


class Image_direction(Enum):
    Hight = 0
    width = 1


class file():
    # option
    DEBUG = False

    IMAGE_EXT = ["jpg", "png", "bmp", ".jpg", ".png", ".bmp"]
    VIDEO_EXT = ["mp4", "avi", ".mp4", ".avi"]

    @classmethod
    def image_read(self, filename: str, color_option: Color_option = Color_option.BGR) -> _numpy.ndarray:
        if not _base.file._exist_check(filename):
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
        filename = _base.file._name_from_path(file_dir)
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
    def video_write(self, filename: str, video_size, frame=30):
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


class cv_base():
    @staticmethod
    def image_stack(images, channel_option: C_position):
        if channel_option.value:  # stack to last channel
            _axis = -1
        else:               # stack to first channel
            _axis = 0
        return _numpy.np_base.stack(images, _axis)

    @staticmethod
    def channel_converter(image, channel_option: C_position):
        if channel_option.value:  # [w, h, c]
            return _numpy.image.conver_to_last_channel(image)
        else:  # [c, w, h]
            return _numpy.image.conver_to_first_channel(image)

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
            if to_range == R_option.ZtoFF:  # convert to [0.0, 1.0] -> [0, 255]
                return _numpy.np_base.type_converter(image * 0xff, "uint8")
            else:
                return image
        elif form_range == R_option.ZtoFF:
            if to_range == R_option.ZtoO:  # convert to [0, 255] -> [0.0, 1.0]
                return image / 0xFF
            else:
                return image

    @staticmethod
    def img_cvt(image, cvt_option: CVT_option):
        if cvt_option == CVT_option.GRAY2BGR:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif cvt_option == CVT_option.BGR2GRAY:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif cvt_option == CVT_option.BGR2RGB:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def filtering(image: _numpy.np.ndarray, array):
        if len(image.shape) > 2:
            # color image
            holder = _numpy.np_base.get_array_from(image)
            for _ch_ct in range(image.shape[-1]):
                holder[:, :, _ch_ct] = cv_base.filtering(image[:, :, _ch_ct], array)
            return holder
        else:
            return cv2.filter2D(image, cv2.CV_64F, array)

    @staticmethod
    def padding(image: _numpy.np.ndarray, padding: Union[int, List[int]]):
        """
        padding
            int  A              -> [top : A, bottom: A, left : A, right: A]\n
            list [A, B]         -> [top : A, bottom: A, left : B, right: B]\n
            list [A, B, C, D]   -> [top : A, bottom: B, left : C, right: D]\n
        """
        # make holder -> in later add multi padding option
        if isinstance(padding, int):
            _t_pad, _b_pad, _l_pad, _r_pad = [padding, padding, padding, padding]
        elif isinstance(padding, list):
            if len(padding) == 2:
                # [hight_pad, width_pad]
                _t_pad, _b_pad, _l_pad, _r_pad = [padding[0], padding[0], padding[1], padding[1]]
            elif len(padding) == 4:
                # [top, bottom, left, right]
                _t_pad, _b_pad, _l_pad, _r_pad = padding

        if len(image.shape) == 3:
            _h, _w, _c = image.shape
            _holder_shape = [_h + (_t_pad + _b_pad), _w + (_l_pad + _r_pad), _c]
        elif len(image.shape) == 2:
            _h, _w = image.shape
            _holder_shape = [_h + (_t_pad + _b_pad), _w + (_l_pad + _r_pad)]

        _holder = _numpy.np_base.get_array_from(_holder_shape, True)

        _t_pad = _t_pad if _t_pad else None
        _b_pad = -_b_pad if _b_pad else None
        _l_pad = _l_pad if _l_pad else None
        _r_pad = -_r_pad if _r_pad else None

        _holder[_t_pad: -_b_pad, _l_pad: -_r_pad] = image

        return _holder

    @staticmethod
    def unpadding(image: _numpy.np.ndarray, padding: Union[int, List[int]]):
        # make holder -> in later add multi padding option
        if isinstance(padding, int):
            _t_pad, _b_pad, _l_pad, _r_pad = [padding, padding, padding, padding]
        elif isinstance(padding, list):
            if len(padding) == 2:
                # [hight_pad, width_pad]
                _t_pad, _b_pad, _l_pad, _r_pad = [padding[0], padding[0], padding[1], padding[1]]
            elif len(padding) == 4:
                # [top, bottom, left, right]
                _t_pad, _b_pad, _l_pad, _r_pad = padding

        return image[_t_pad: -_b_pad, _l_pad: -_r_pad]

    class blur():
        default = {
            "gaussian": {
                "ksize": (5, 5),
                "sigmaX": 0
            },
            "bilateral": {
                "d": -1,
                "sigmaColor": 10,
                "sigmaSpace": 5}}

        def __call__(self, image, style):
            if style == "gaussian":
                return cv2.GaussianBlur(image, **self.default[style])
            elif style == "bilateral":
                return cv2.bilateralFilter(image, **self.default[style])


class edge():
    class gradient():
        @staticmethod
        def sobel(image: _numpy.ndarray, is_euclid: bool = True):
            if len(image.shape) > 2:
                # color image
                delta_holder = _numpy.np_base.get_array_from(image.shape[:2], is_shape=True, value=0, dtype="float32")
                direction_holder = _numpy.np_base.get_array_from(image.shape[:2], is_shape=True, value=0, dtype="float32")
                for _ch_ct in range(image.shape[-1]):
                    result = edge.gradient.sobel(image[:, :, _ch_ct], is_euclid)
                    delta_holder += result[0]
                    direction_holder += result[1]

                return delta_holder / 3, (direction_holder / 3).round()
            else:
                dx = cv_base.filtering(image, _numpy.np_base.get_array_from([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float"))
                dy = cv_base.filtering(image, _numpy.np_base.get_array_from([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype="float"))
                dxy = _numpy.cal.distance(dx, dy, is_euclid)
                direction = _numpy.cal.get_direction(dx, dy)
                return dxy, direction

    @staticmethod
    def gradient_to_edge(gradient: List[_numpy.ndarray], threshold: List[int] = 1, is_edge_shrink: bool = True, is_active_map: bool = False):
        # gradient -> [delta, direction]
        _delta = gradient[0]
        _filterd = _numpy.np_base.range_cut(_numpy.np_base.normalization(_delta), [threshold, ], "upper")

        if is_edge_shrink:
            _direction = gradient[1]
            _edge = _numpy.image.direction_check(_delta * _filterd, _direction, [0, 1, 2, 3])
        else:
            _edge = (_filterd != 0)

        return _edge if is_active_map else _numpy.np_base.type_converter(0xFF * _edge, "uint")

    @staticmethod
    def sobel(image: _numpy.np.ndarray, threshold: List[int] = 1, is_euclid: bool = True, is_edge_shrink: bool = True, is_active_map: bool = False):
        if len(image.shape) > 2:
            # color image
            holder = _numpy.np_base.get_array_from(image.shape[:2], is_shape=True, value=0, dtype="float32")
            for _ch_ct in range(image.shape[-1]):
                holder += edge.sobel(image[:, :, _ch_ct], threshold, is_euclid, is_edge_shrink, True)
            holder = (holder >= 2)
            return holder if is_active_map else _numpy.np_base.type_converter(0xFF * holder, "uint8")
        else:
            dx = cv2.Sobel(image, -1, 1, 0, delta=128)
            dy = cv2.Sobel(image, -1, 0, 1, delta=128)

            dxy = _numpy.cal.distance(dx, dy) if is_euclid else cv2.addWeighted(dx, 0.5, dy, 0.5, 0)
            _edge = _numpy.np_base.range_cut(_numpy.np_base.normalization(dxy), [-threshold, threshold], ouput_type="value")

            if is_edge_shrink:
                direction = _numpy.cal.get_direction(dx, dy)
                _edge = _numpy.image.direction_check(_edge, direction, [0, 1, 2, 3])

            else:
                _edge = (_edge != 0)

            return _edge if is_active_map else _numpy.np_base.type_converter(0xFF * _edge, "uint")

    @staticmethod
    def canny(gray_image, ths, k_size=3, range=R_option.ZtoFF, channel=C_position.Last):
        _high = ths[0]
        _low = ths[1]

        canny_image = cv2.Canny(gray_image, _low, _high, k_size)  # [h, w]
        if channel is not None:
            canny_image = cv_base.image_stack([canny_image, canny_image, canny_image], channel)

        return cv_base.range_converter(canny_image, R_option.ZtoFF, range)


class gui_process():
    @staticmethod
    def display(image: _numpy.np.ndarray, dispaly_window: str, ms_delay: int = -1):
        cv2.imshow(dispaly_window, image)
        return cv2.waitKeyEx(ms_delay)

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
    @dataclass
    class pen():
        color: list = field(default_factory=lambda: [0xFF, 0xFF, 0xFF])
        thickness: int = 1

    @dataclass
    class point():
        x: int = 0
        y: int = 0

    class text_position(Enum):
        Inside = 0b10000
        Outside = 0b00000
        Top = 0b00001
        Middle = 0b00010
        Bottom = 0b00000
        Left = 0b00100
        Normal = 0b01000
        Right = 0b00000

    @staticmethod
    def merge(image: List[_numpy.np.ndarray], direction: Image_direction, interval: int = 10):
        _merge_img = image[0]
        if direction == Image_direction.Hight:
            for _img in image[1:]:
                _merge_img = cv_base.padding(_merge_img, [0, interval, 0, 0])
                _merge_img = cv2.vconcat(_merge_img, _img)
        else:  # width
            for _img in image[1:]:
                _merge_img = cv_base.padding(_merge_img, [0, 0, 0, interval])
                _merge_img = cv2.hconcat(_merge_img, _img)

        return _merge_img

    @staticmethod
    def text(image: _numpy.np.ndarray, text: str):
        [_text_w, _text_h], _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=2)
        text_padding = 10
        if len(image.shape) == 3:
            _h, _w, _c = image.shape
            _text_shell_size = [_text_h + (2 * text_padding), _text_w + (2 * text_padding), _c]
            _tpye_image_size = [_h + _text_h, _w, _c]

        elif len(image.shape) == 2:
            _h, _w = image.shape
            _text_shell_size = [_text_h + (2 * text_padding), _text_w + (2 * text_padding)]
            _tpye_image_size = [_h + _text_h, _w]

        _text_shell = _numpy.np_base.get_array_from(_text_shell_size, True, 0xFF)
        cv2.putText(img=_text_shell,
                    text=text,
                    org=(text_padding, _text_h + text_padding),
                    fontScale=1,
                    thickness=2,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    color=0)

        _text_shell = cv_base.resize(_text_shell, [_text_h, _w])
        _tpye_image = _numpy.np_base.get_array_from(_tpye_image_size, True, 0xFF)
        _tpye_image[:_h, :, :] = image
        _tpye_image[_h:, :, :] = _text_shell

        return _tpye_image

    @staticmethod
    def rectangle():
        pass

    @staticmethod
    def circle():
        pass

    @staticmethod
    def oval():
        pass

    @staticmethod
    def _polygon(image, pts, thick, color, is_close=True):
        pts = _numpy.image.poly_points(pts)
        if thick == -1:
            return cv2.fillConvexPoly(image, pts, color)
        else:
            return cv2.polylines(image, pts, is_close, color, thick)

    class canvas():
        background = None

        object_list = []  # draw object list
        points = []  # [point, point, point, point...]

        def __init__(self, size=None, sample=None) -> None:
            self.active_pen = draw.pen()
            if size is not None or sample is not None:
                self.set_canvas(size, sample, is_color=[0xFF, 0xFF, 0xFF])

        def set_pen(self, color, thickness=1):
            self.active_pen.color = color
            self.active_pen.thickness = thickness

        def set_canvas(self, size, sample=None, is_color=0):
            self.background = _numpy.np_base.get_array_from(size, True, is_color) if sample is not None \
                else _numpy.np_base.get_array_from(sample, False, is_color)

        def set_object(self):
            pass

        def del_object(self, obj_num):
            """
            Arg:\n
                target (list) : \n
                obj_num (int, list[int], range) : \n
            """
            if _base.tool_for._list.is_num_over_range(self.object_list, obj_num):
                pass  # error : "obj_num" is over range in self object list

            else:
                _base.tool_for._list.del_obj(self.object_list, obj_num)

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
