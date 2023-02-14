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
from enum import Enum
from typing import List, Union, Tuple
from dataclasses import dataclass, field
import cv2

if __package__ == "":
    from _base import File
    from _numpy import Array_Process, Image_Process, Np_Dtype, ndarray
    import _error as _e

else:
    from ._base import File
    from ._numpy import Array_Process, Image_Process, Np_Dtype, ndarray
    from . import _error as _e

_error_message = _e.Custom_error("AIS_utils", "_cv2")


# -- DEFINE CONSTNAT -- #
class Color_Option(Enum):
    BGR = 0
    RGB = 1
    BGRA = 2
    GRAY = 3


class CVT_option(Enum):
    GRAY2BGR = 0
    BGR2GRAY = 1
    BGR2RGB = 2


class Channel_Style(Enum):
    Last = True
    First = False


class R_option(Enum):
    ZtoFF = 0  # [0, 255]
    ZtoO = 1  # [0, 1.0]


class Image_direction(Enum):
    Hight = 0
    width = 1


class Support_Image_Extension(Enum):
    JPG = "jpg"
    PNG = "png"
    BMP = "bmp"


class Support_Video_Extension(Enum):
    MP4 = "mp4"
    AVI = "avi"


class Segmentation_Style(Enum):
    CLASS_MAP = 0       # (h, w, class count)
    COLOR_MAP = 1       # (h, w, 3)
    CLASSIFICATION = 2  # (h, w)

# -- DEFINE CONFIG -- #


# -- Mation Function -- #
class File_IO():
    @staticmethod
    def _Image_read(file_path: str, color_option: Color_Option = Color_Option.BGR) -> ndarray:
        _exist, file_path = File._extension_check(file_path, [ext.value for ext in Support_Image_Extension], True)

        if not _exist:
            raise ValueError(f"image file {file_path} not exist")

        # data read
        if color_option == Color_Option.BGR:
            _read_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        elif color_option == Color_Option.GRAY:
            _read_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        elif color_option == Color_Option.RGB:
            _read_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            _read_img = cv2.cvtColor(_read_img, cv2.COLOR_BGR2RGB)
        elif color_option == Color_Option.BGRA:
            # this code dosen't check it. if you wnat use it. check it output
            _read_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        return _read_img

    @staticmethod
    def _Image_write(file_path: str, image: ndarray):
        _, file_path = File._extension_check(file_path, [ext.value for ext in Support_Image_Extension], True)
        cv2.imwrite(file_path, image)

    # @staticmethod  # in later fix
    # def _video_capture(location: str, is_file=False):
    #     _location = Directory._make(location)
    #     cap = cv2.VideoCapture(_location)
    #     return cap

    # @classmethod  # in later fix
    # def _video_write(self, filename: str, video_size, frame=30):
    #     video_format = filename.split("/")[-1].split(".")[-1]

    #     if not _base.File._extension_check(video_format, self.VIDEO_EXT):
    #         video_format = "avi"
    #         filename += "avi" if filename[-1] == "." else ".avi"

    #     _h, _w = video_size[:2]

    #     if video_format == "avi":
    #         fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #     elif video_format == "mp4":
    #         fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    #     return cv2.VideoWriter(filename, fourcc, frame, (_w, _h))


class Base_Process():
    @staticmethod
    def _Resize(image: ndarray, size: List[Union[int, float]]):
        _h, _w = image.shape[:2]
        _interpolation = cv2.INTER_AREA

        # ratiol
        if type(size[0]) == float and type(size[1]) == float:
            if size[0] >= 1.0 or size[1] >= 1.0:
                _interpolation = cv2.INTER_LINEAR
            return cv2.resize(image, dsize=(0, 0), fx=size[1], fy=size[0], interpolation=_interpolation)

        # absolute
        elif type(size[0]) == int and type(size[1]) == int:
            if size[0] >= _w or size[1] >= _h:
                _interpolation = cv2.INTER_LINEAR
            return cv2.resize(image, dsize=(size[1], size[0]), interpolation=_interpolation)

        else:
            return image

    @staticmethod
    def image_stack(images, channel_option: Channel_Style):
        if channel_option.value:  # stack to last channel
            _axis = -1
        else:               # stack to first channel
            _axis = 0
        return Array_Process.stack(images, _axis)

    @staticmethod
    def channel_converter(image, channel_option: Channel_Style):
        if channel_option.value:  # [w, h, c]
            return Image_Process._conver_to_last_channel(image)
        else:  # [c, w, h]
            return Image_Process._conver_to_first_channel(image)

    @staticmethod
    def range_converter(image, form_range: R_option, to_range: R_option):
        if form_range == R_option.ZtoO:
            if to_range == R_option.ZtoFF:  # convert to [0.0, 1.0] -> [0, 255]
                return Array_Process._converter(image * 0xff, dtype=Np_Dtype.UINT)
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
    def filtering(image: ndarray, array):
        if len(image.shape) > 2:
            # color image
            holder = Array_Process._converter(image, dtype=Np_Dtype.UINT)
            for _ch_ct in range(image.shape[-1]):
                holder[:, :, _ch_ct] = Base_Process.filtering(image[:, :, _ch_ct], array)
            return holder
        else:
            return cv2.filter2D(image, cv2.CV_64F, array)

    @staticmethod
    def padding(image: ndarray, padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]], value: int):
        """
        padding
            int  A              -> [top : A, bottom: A, left : A, right: A]\n
            list [A, B]         -> [top : A, bottom: A, left : B, right: B]\n
            list [A, B, C, D]   -> [top : A, bottom: B, left : C, right: D]\n
        """
        # make holder -> in later add multi padding option
        if isinstance(padding, int):
            _t_pad, _b_pad, _l_pad, _r_pad = [padding, padding, padding, padding]
        else:
            if len(padding) == 2:
                # [hight_pad, width_pad]
                _t_pad, _b_pad, _l_pad, _r_pad = [padding[0], padding[0], padding[1], padding[1]]
            else:
                # [top, bottom, left, right]
                _t_pad, _b_pad, _l_pad, _r_pad = padding

        _holder_shape = [_v for _v in image.shape]
        _holder_shape[0] += _t_pad + _b_pad  # h padding
        _holder_shape[1] += _l_pad + _r_pad  # w padding

        _holder = Array_Process._converter(_holder_shape, is_shape=True, value=value, dtype=Np_Dtype.UINT)

        _t_pad = _t_pad if _t_pad else None
        _b_pad = -_b_pad if _b_pad else None
        _l_pad = _l_pad if _l_pad else None
        _r_pad = -_r_pad if _r_pad else None

        _holder[_t_pad: _b_pad, _l_pad: _r_pad] = image

        return _holder

    @staticmethod
    def unpadding(image: ndarray, padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]]):
        # make holder -> in later add multi padding option
        if isinstance(padding, int):
            _t_pad, _b_pad, _l_pad, _r_pad = [padding, padding, padding, padding]
        else:  # isinstance(padding, list):
            if len(padding) == 2:
                # [hight_pad, width_pad]
                _t_pad, _b_pad, _l_pad, _r_pad = [padding[0], padding[0], padding[1], padding[1]]
            else:  # len(padding) == 4
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
        def sobel(image: ndarray, is_euclid: bool = True):
            if len(image.shape) > 2:
                # color image
                delta_holder = Array_Process._Make_array(image.shape[:2], 0, dtype=Np_Dtype.FLOAT)
                direction_holder = Array_Process._Make_array(image.shape[:2], 0, dtype=Np_Dtype.FLOAT)

                for _ch_ct in range(image.shape[-1]):
                    result = edge.gradient.sobel(image[:, :, _ch_ct], is_euclid)
                    delta_holder += result[0]
                    direction_holder += result[1]

                return delta_holder / 3, (direction_holder / 3).round()
            else:
                dx = Base_Process.filtering(image, Array_Process._converter([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=Np_Dtype.FLOAT))
                dy = Base_Process.filtering(image, Array_Process._converter([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=Np_Dtype.FLOAT))
                dxy = Image_Process._distance(dx, dy, is_euclid)
                direction = Image_Process._gredient_direction(dx, dy)
                return dxy, direction

    @staticmethod
    def gradient_to_edge(gradient: List[ndarray], threshold: Union[int, Tuple[int, int]] = 1, is_edge_shrink: bool = True, is_active_map: bool = False):
        # gradient -> [delta, direction]
        _delta = gradient[0]
        _filterd = Array_Process.range_cut(Array_Process._Norm(_delta), threshold, "upper")

        if is_edge_shrink:
            _direction = gradient[1]
            _edge = Image_Process._direction_check(_delta * _filterd, _direction, [0, 1, 2, 3])
        else:
            _edge = (_filterd != 0)

        return _edge if is_active_map else Array_Process._converter(0xFF * _edge, dtype=Np_Dtype.UINT)

    @staticmethod
    def sobel(
            image: ndarray,
            threshold: Union[int, Tuple[int, int]] = (-1, 1),
            is_euclid: bool = True,
            is_edge_shrink: bool = True,
            is_active_map: bool = False):
        if len(image.shape) > 2:
            # color image
            holder = Array_Process._converter(image.shape[:2], is_shape=True, value=0, dtype=Np_Dtype.FLOAT)
            for _ch_ct in range(image.shape[-1]):
                holder += edge.sobel(image[:, :, _ch_ct], threshold, is_euclid, is_edge_shrink, True)
            holder = (holder >= 2)
            return holder if is_active_map else Array_Process._converter(0xFF * holder, dtype=Np_Dtype.UINT)
        else:
            dx = cv2.Sobel(image, -1, 1, 0, delta=128)
            dy = cv2.Sobel(image, -1, 0, 1, delta=128)

            dxy = Image_Process._distance(dx, dy, is_euclid) if is_euclid else cv2.addWeighted(dx, 0.5, dy, 0.5, 0)
            _edge = Array_Process.range_cut(Array_Process._Norm(dxy), threshold, ouput_type="value")

            if is_edge_shrink:
                direction = Image_Process._gredient_direction(dx, dy)
                _edge = Image_Process._direction_check(_edge, direction, [0, 1, 2, 3])

            else:
                _edge = (_edge != 0)

            return _edge if is_active_map else Array_Process._converter(0xFF * _edge, dtype=Np_Dtype.UINT)

    @staticmethod
    def canny(gray_image, ths, k_size=3, range=R_option.ZtoFF, channel=Channel_Style.Last):
        _high = ths[0]
        _low = ths[1]

        canny_image = cv2.Canny(gray_image, _low, _high, k_size)  # [h, w]
        if channel is not None:
            canny_image = Base_Process.image_stack([canny_image, canny_image, canny_image], channel)

        return Base_Process.range_converter(canny_image, R_option.ZtoFF, range)


class gui_process():
    @staticmethod
    def _Display(image: ndarray, dispaly_window: str, ms_delay: int = -1):
        cv2.imshow(dispaly_window, image)
        return cv2.waitKeyEx(ms_delay)

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

    # @staticmethod
    # def _Make_text_box(image: ndarray, text: str, position: text_position):
    #     [_text_w, _text_h], _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=2)
    #     text_padding = 10

    #     _holder_shape = [_v for _v in image.shape]
    #     _holder_shape[0] += _t_pad + _b_pad  # h padding
    #     _holder_shape[1] += _l_pad + _r_pad  # w padding

    #     if len(image.shape) == 3:
    #         _h, _w, _c = image.shape
    #         _text_shell_size = [_text_h + (2 * text_padding), _text_w + (2 * text_padding), _c]
    #         _tpye_image_size = [_h + _text_h, _w, _c]

    #     elif len(image.shape) == 2:
    #         _h, _w = image.shape
    #         _text_shell_size = [_text_h + (2 * text_padding), _text_w + (2 * text_padding)]
    #         _tpye_image_size = [_h + _text_h, _w]

    #     _text_shell = Array_Process._converter(_text_shell_size, True, 0xFF)
    #     cv2.putText(img=_text_shell,
    #                 text=text,
    #                 org=(text_padding, _text_h + text_padding),
    #                 fontScale=1,
    #                 thickness=2,
    #                 fontFace=cv2.FONT_HERSHEY_DUPLEX,
    #                 color=0)

    #     _text_shell = Base_Process.resize(_text_shell, [_text_h, _w])
    #     _tpye_image = Array_Process._converter(_tpye_image_size, True, 0xFF)
    #     _tpye_image[:_h, :] = image
    #     _tpye_image[_h:, :] = _text_shell

    #     return _tpye_image

    # @staticmethod
    # def merge(image: List[ndarray], direction: Image_direction, interval: int = 10, interval_color: int = 255):
    #     _merge_img = image[0]
    #     if direction == Image_direction.Hight:
    #         for _img in image[1:]:
    #             _merge_img = Base_Process.padding(_merge_img, (0, interval, 0, 0), interval_color)
    #             _merge_img = cv2.vconcat([_merge_img, _img])
    #     else:  # width
    #         for _img in image[1:]:
    #             _merge_img = Base_Process.padding(_merge_img, (0, 0, 0, interval), interval_color)
    #             _merge_img = cv2.hconcat((_merge_img, _img))

    #     return _merge_img

    # @staticmethod
    # def rectangle():
    #     pass

    # @staticmethod
    # def circle():
    #     pass

    # @staticmethod
    # def oval():
    #     pass

    # @staticmethod
    # def _polygon(image, pts, thick, color, is_close=True):
    #     pts = Image_Process.poly_points(pts)
    #     if thick == -1:
    #         return cv2.fillConvexPoly(image, pts, color)
    #     else:
    #         return cv2.polylines(image, pts, is_close, color, thick)

    # class canvas():
    #     background = None

    #     object_list = []  # draw object list
    #     points = []  # [point, point, point, point...]

    #     def __init__(self, size=None, sample=None) -> None:
    #         self.active_pen = draw.pen()
    #         if size is not None or sample is not None:
    #             self.set_canvas(size, sample, is_color=[0xFF, 0xFF, 0xFF])

    #     def set_pen(self, color, thickness=1):
    #         self.active_pen.color = color
    #         self.active_pen.thickness = thickness

    #     def set_canvas(self, size, sample=None, is_color=0):
    #         self.background = Array_Process._converter(size, True, is_color) if sample is not None \
    #             else Array_Process._converter(sample, False, is_color)

    #     def set_object(self):
    #         pass

    #     # def del_object(self, obj_num):
    #     #     """
    #     #     Arg:\n
    #     #         target (list) : \n
    #     #         obj_num (int, list[int], range) : \n
    #     #     """
    #     #     if _base.Tool_For._list.is_num_over_range(self.object_list, obj_num):
    #     #         pass  # error : "obj_num" is over range in self object list

    #     #     else:
    #     #         _base.Tool_For._list.del_obj(self.object_list, obj_num)

    #     def clear_object(self):
    #         pass

    #     def draw(self):
    #         pass
