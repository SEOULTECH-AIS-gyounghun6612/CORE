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
from __future__ import annotations
from enum import Enum
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass, field
import cv2
import numpy as np
from numpy import ndarray

from ._System import Path, File
from ._Array import Array_IO


class Format_of():
    class Image(Enum):
        BGR = cv2.IMREAD_COLOR
        BGRA = cv2.IMREAD_UNCHANGED
        GRAY = cv2.IMREAD_GRAYSCALE

    class Codex(Enum):
        MP4 = "DIVX"


class Segmentation_Style(Enum):
    CLASS_MAP = 0       # (h, w, class count)
    COLOR_MAP = 1       # (h, w, 3)
    CLASSIFICATION = 2  # (h, w)


@dataclass
class Camera():
    cam_id: int
    image_size: List
    intrinsic: ndarray
    rectification: ndarray = np.eye(4)

    def _world_to_img(self, tr_to_cam: ndarray, points: ndarray, limit_z: Tuple[int, Optional[int]] = (0, None)):
        """
        """
        _extrict = self.rectification @ tr_to_cam
        _points_on_cam = np.matmul(_extrict, points)
        _projection: ndarray = np.matmul(self.intrinsic, _points_on_cam)

        _z_min_limit, _z_max_limit = limit_z

        _points_in_front = _projection[:, _projection[2] >= _z_min_limit]
        _points_in_front = _points_in_front if _z_max_limit is None else _points_in_front[:, _points_in_front[2] < _z_max_limit]

        _depth = _points_in_front[2]
        if _z_min_limit == 0:
            _depth[_depth == 0] = -1e-6

        _u = np.round(_points_in_front[0, :] / _depth).astype(int)
        _v = np.round(_points_in_front[1, :] / _depth).astype(int)

        # filtering the point of that over the image size
        _filter = (_v >= 0) * (_v < self.image_size[1]) * (_u >= 0) * (_u < self.image_size[0])

        return _u[_filter], _v[_filter], _depth[_filter]

    def _cam_to_world(self, pixel):
        ...


class Vision_IO(File.Basement):
    class Image():
        def __init__(self, save_dir: str) -> None:
            Vision_IO._Path_check("", save_dir)  # If pass this code, save_dir is exist
            self.save_dir = save_dir

        def _Make_image_list(self):
            _image_file_list = Path._Search(self.save_dir, Path.Type.FILE)
            self.file_stream_list = [open(_file, "rb") for _file in _image_file_list]
            self.image_file_list = _image_file_list

        def _Read_data_in_list(self, id: int, data_format: Format_of.Image):
            _file_stream = self.file_stream_list[id]
            _encoded_img = Array_IO._Read_from_binary(_file_stream, np.uint8)
            return cv2.imdecode(_encoded_img, data_format.value)

    class Video():
        class Capture():
            def __init__(self, source_name: Union[int, str], save_dir: str = ""):
                # get capute source
                if isinstance(source_name, int):
                    _source = source_name
                else:
                    _, _source = Vision_IO._Path_check(source_name, save_dir, raise_error=True)

                _cap = cv2.VideoCapture(_source)

                if not _cap.isOpened():
                    raise RuntimeError("Camera open failed!")

                self.capture = _cap

            def _Get_video_info(self):
                _w = round(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                _h = round(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                _fps = self.capture.get(cv2.CAP_PROP_FPS)

                return _w, _h, _fps

            def __call__(self):
                _ret, _frame = self.capture.read()

                if not _ret:
                    self.capture.release()
                    return False, None
                else:
                    return True, _frame

        class Writer():
            def __init__(self, img_size: Tuple[int, int], file_name: str, save_dir: str, video_codex: Format_of.Codex, frame: int):
                _img_h, _img_w = img_size

                # make the vidoe codex and string of ext
                _ext = video_codex.name.lower()
                _fourcc = cv2.VideoWriter.fourcc(*video_codex.value)

                _, _file_path = Vision_IO._Path_check(file_name, save_dir, _ext)
                self._writer = cv2.VideoWriter(_file_path, _fourcc, frame, (_img_w, _img_h))

            def __call__(self, frame: ndarray):
                self._writer.write(frame)

            def _Close(self):
                self._writer.release()


class Vision_Toolbox():
    @staticmethod
    def _format_converter(image: ndarray):
        return ...

    @staticmethod
    def _channel_converter(image: ndarray, convert_to_last_ch: bool = False):
        _shape = image.shape

        if len(_shape) == 2:
            # [h, w] -> [h, w, 1] or [1, h, w]
            return image[:, :, np.newaxis] if convert_to_last_ch else image[np.newaxis, :, :]

        else:
            # [h, w, c] -> [c, h, w] or [c, h, w] -> [h, w, c]
            return np.moveaxis(image, 0, -1) if convert_to_last_ch else np.moveaxis(image, -1, 0)

# class Base_Process():
#     @staticmethod
#     def _Resize(image: ndarray, size: List[Union[int, float]]):
#         _h, _w = image.shape[:2]
#         _interpolation = cv2.INTER_AREA

#         # ratiol
#         if isinstance(size[0], float) and isinstance(size[1], float):
#             if size[0] >= 1.0 or size[1] >= 1.0:
#                 _interpolation = cv2.INTER_LINEAR
#             return cv2.resize(image, dsize=(0, 0), fx=size[1], fy=size[0], interpolation=_interpolation)

#         # absolute
#         elif isinstance(size[0], int) and isinstance(size[1], int):
#             if size[0] >= _w or size[1] >= _h:
#                 _interpolation = cv2.INTER_LINEAR
#             return cv2.resize(image, dsize=(size[1], size[0]), interpolation=_interpolation)

#         else:
#             return image

#     @staticmethod
#     def image_stack(images, channel_option: Channel_Style):
#         if channel_option.value:  # stack to last channel
#             _axis = -1
#         else:               # stack to first channel
#             _axis = 0
#         return Custom_Array_Process.stack(images, _axis)

#     @staticmethod
#     def channel_converter(image, channel_option: Channel_Style):
#         if channel_option.value:  # [w, h, c]
#             return Image_Process._conver_to_last_channel(image)
#         else:  # [c, w, h]
#             return Image_Process._conver_to_first_channel(image)

#     @staticmethod
#     def range_converter(image, form_range: R_option, to_range: R_option):
#         if form_range == R_option.ZtoO:
#             if to_range == R_option.ZtoFF:  # convert to [0.0, 1.0] -> [0, 255]
#                 return Custom_Array_Process._Convert_from(image * 0xff, dtype=Np_Dtype.UINT)
#             else:
#                 return image
#         elif form_range == R_option.ZtoFF:
#             if to_range == R_option.ZtoO:  # convert to [0, 255] -> [0.0, 1.0]
#                 return image / 0xFF
#             else:
#                 return image

#     @staticmethod
#     def img_cvt(image, cvt_option: CVT_Option):
#         if cvt_option == CVT_Option.GRAY2BGR:
#             return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#         elif cvt_option == CVT_Option.BGR2GRAY:
#             return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         elif cvt_option == CVT_Option.BGR2RGB:
#             return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     @staticmethod
#     def filtering(image: ndarray, array):
#         if len(image.shape) > 2:
#             # color image
#             holder = Custom_Array_Process._Convert_from(image, dtype=Np_Dtype.UINT)
#             for _ch_ct in range(image.shape[-1]):
#                 holder[:, :, _ch_ct] = Base_Process.filtering(image[:, :, _ch_ct], array)
#             return holder
#         else:
#             return cv2.filter2D(image, cv2.CV_64F, array)

#     @staticmethod
#     def padding(image: ndarray, padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]], value: int):
#         """
#         padding
#             int  A              -> [top : A, bottom: A, left : A, right: A]\n
#             list [A, B]         -> [top : A, bottom: A, left : B, right: B]\n
#             list [A, B, C, D]   -> [top : A, bottom: B, left : C, right: D]\n
#         """
#         # make holder -> in later add multi padding option
#         if isinstance(padding, int):
#             _t_pad, _b_pad, _l_pad, _r_pad = [padding, padding, padding, padding]
#         else:
#             if len(padding) == 2:
#                 # [hight_pad, width_pad]
#                 _t_pad, _b_pad, _l_pad, _r_pad = [padding[0], padding[0], padding[1], padding[1]]
#             else:
#                 # [top, bottom, left, right]
#                 _t_pad, _b_pad, _l_pad, _r_pad = padding

#         _holder_shape = [_v for _v in image.shape]
#         _holder_shape[0] += _t_pad + _b_pad  # h padding
#         _holder_shape[1] += _l_pad + _r_pad  # w padding

#         _holder = Custom_Array_Process._Make_array(_holder_shape, value=value, dtype=Np_Dtype.UINT)

#         _t_pad = _t_pad if _t_pad else None
#         _b_pad = -_b_pad if _b_pad else None
#         _l_pad = _l_pad if _l_pad else None
#         _r_pad = -_r_pad if _r_pad else None

#         _holder[_t_pad: _b_pad, _l_pad: _r_pad] = image

#         return _holder

#     @staticmethod
#     def unpadding(image: ndarray, padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]]):
#         # make holder -> in later add multi padding option
#         if isinstance(padding, int):
#             _t_pad, _b_pad, _l_pad, _r_pad = [padding, padding, padding, padding]
#         else:  # isinstance(padding, list):
#             if len(padding) == 2:
#                 # [hight_pad, width_pad]
#                 _t_pad, _b_pad, _l_pad, _r_pad = [padding[0], padding[0], padding[1], padding[1]]
#             else:  # len(padding) == 4
#                 # [top, bottom, left, right]
#                 _t_pad, _b_pad, _l_pad, _r_pad = padding

#         return image[_t_pad: -_b_pad, _l_pad: -_r_pad]

#     class blur():
#         default = {
#             "gaussian": {
#                 "ksize": (5, 5),
#                 "sigmaX": 0
#             },
#             "bilateral": {
#                 "d": -1,
#                 "sigmaColor": 10,
#                 "sigmaSpace": 5}}

#         def __call__(self, image, style):
#             if style == "gaussian":
#                 return cv2.GaussianBlur(image, **self.default[style])
#             elif style == "bilateral":
#                 return cv2.bilateralFilter(image, **self.default[style])


# class edge():
#     class gradient():
#         @staticmethod
#         def sobel(image: ndarray, is_euclid: bool = True):
#             if len(image.shape) > 2:
#                 # color image
#                 delta_holder = Custom_Array_Process._Make_array(image.shape[:2], 0, dtype=Np_Dtype.FLOAT)
#                 direction_holder = Custom_Array_Process._Make_array(image.shape[:2], 0, dtype=Np_Dtype.FLOAT)

#                 for _ch_ct in range(image.shape[-1]):
#                     result = edge.gradient.sobel(image[:, :, _ch_ct], is_euclid)
#                     delta_holder += result[0]
#                     direction_holder += result[1]

#                 return delta_holder / 3, (direction_holder / 3).round()
#             else:
#                 dx = Base_Process.filtering(image, Custom_Array_Process._Convert_from([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=Np_Dtype.FLOAT))
#                 dy = Base_Process.filtering(image, Custom_Array_Process._Convert_from([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=Np_Dtype.FLOAT))
#                 dxy = Image_Process._distance(dx, dy, is_euclid)
#                 direction = Image_Process._gredient_direction(dx, dy)
#                 return dxy, direction

#     @staticmethod
#     def gradient_to_edge(gradient: List[ndarray], threshold: Union[int, Tuple[int, int]] = 1, is_edge_shrink: bool = True, is_active_map: bool = False):
#         # gradient -> [delta, direction]
#         _delta = gradient[0]
#         _filterd = Custom_Array_Process.range_cut(Custom_Array_Process._Norm(_delta), threshold, "upper")

#         if is_edge_shrink:
#             _direction = gradient[1]
#             _edge = Image_Process._direction_check(_delta * _filterd, _direction, [0, 1, 2, 3])
#         else:
#             _edge = (_filterd != 0)

#         return _edge if is_active_map else Custom_Array_Process._converter(0xFF * _edge, dtype=Np_Dtype.UINT)

#     @staticmethod
#     def sobel(
#             image: ndarray,
#             threshold: Union[int, Tuple[int, int]] = (-1, 1),
#             is_euclid: bool = True,
#             is_edge_shrink: bool = True,
#             is_active_map: bool = False):
#         if len(image.shape) > 2:
#             # color image
#             holder = Custom_Array_Process._Make_array(image.shape[:2], value=0, dtype=Np_Dtype.FLOAT)
#             for _ch_ct in range(image.shape[-1]):
#                 holder += edge.sobel(image[:, :, _ch_ct], threshold, is_euclid, is_edge_shrink, True)
#             holder = (holder >= 2)
#             return holder if is_active_map else Custom_Array_Process._Convert_from(0xFF * holder, dtype=Np_Dtype.UINT)
#         else:
#             dx = cv2.Sobel(image, -1, 1, 0, delta=128)
#             dy = cv2.Sobel(image, -1, 0, 1, delta=128)

#             dxy = Image_Process._distance(dx, dy, is_euclid) if is_euclid else cv2.addWeighted(dx, 0.5, dy, 0.5, 0)
#             _edge = Custom_Array_Process.range_cut(Custom_Array_Process._Norm(dxy), threshold, ouput_type="value")

#             if is_edge_shrink:
#                 direction = Image_Process._gredient_direction(dx, dy)
#                 _edge = Image_Process._direction_check(_edge, direction, [0, 1, 2, 3])

#             else:
#                 _edge = (_edge != 0)

#             return _edge if is_active_map else Custom_Array_Process._Convert_from(0xFF * _edge, dtype=Np_Dtype.UINT)

#     @staticmethod
#     def canny(gray_image, ths, k_size=3, range=R_option.ZtoFF, channel=Channel_Style.Last):
#         _high = ths[0]
#         _low = ths[1]

#         canny_image = cv2.Canny(gray_image, _low, _high, k_size)  # [h, w]
#         if channel is not None:
#             canny_image = Base_Process.image_stack([canny_image, canny_image, canny_image], channel)

#         return Base_Process.range_converter(canny_image, R_option.ZtoFF, range)


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

# class Image_Process():
#     @staticmethod
#     def _Normalization(image: ndarray, mean: List[float] = [0.485, 0.456, 0.40], std: List[float] = [0.229, 0.224, 0.225]):
#         _, _, _c = image.shape
#         _norm_image = Custom_Array_Process._Make_array_like(image, 0, dtype=Np_Dtype.FLOAT)

#         for _ct_c in range(_c):
#             _norm_image[:, :, _ct_c] = ((image[:, :, _ct_c] / 255) - mean[_ct_c]) / std[_ct_c]

#         return _norm_image

#     @staticmethod
#     def _Un_normalization(norm_image: ndarray, mean: List[float] = [0.485, 0.456, 0.40], std: List[float] = [0.229, 0.224, 0.225]):
#         _, _, _c = norm_image.shape
#         _denorm_image = Custom_Array_Process._Make_array_like(norm_image, 0, dtype=Np_Dtype.UINT)

#         for _ct_c in range(_c):
#             _denorm_image[:, :, _ct_c] = np.round(((norm_image[:, :, _ct_c] * std[_ct_c]) + mean[_ct_c]) * 255)

#         return _denorm_image

#     @staticmethod
#     def _distance(delta_x: ndarray, delta_y: ndarray, is_euclid: bool):
#         if is_euclid:
#             return np.sqrt(np.square(delta_x, dtype=np.float32) + np.square(delta_y, dtype=np.float32), dtype=np.float32)
#         else:
#             return np.abs(delta_x, dtype=np.float32) + np.abs(delta_y, dtype=np.float32)

#     @staticmethod
#     def _gredient_direction(delta_x: ndarray, delta_y: ndarray, sector: int = 8, is_half_cut: bool = True):
#         """
#         if sector count is even (2n)
#         +0 ~ +pi -> n sector, n block (size pi / n)
#         -0 ~ -pi -> n sector, n block (size pi / n)

#         if sector count is odd (2n + 1) => 2 * (2n + 1)
#         +0 ~ +pi -> 2n + 1 sector, n block (size 2 * pi / (2n + 1)) -> left pi / (2n + 1)
#         -0 ~ -pi -> 2m + 1 sector, n block (size 2 * pi / (2nm + 1)) -> left pi / (2n + 1)
#         last one is left sector (2 * pi / (2n + 1))
#         """
#         _block = sector if sector % 2 else sector / 2
#         _block_to_sector = 2 if sector % 2 else 1
#         _term = _block_to_sector * np.pi / _block
#         _bais = _term / 2

#         # get theta
#         theta = np.arctan2(delta_y, delta_x)

#         if is_half_cut and not (sector % 2):
#             theta = theta + np.pi * (theta < 0)
#             sector = int(sector / 2)

#         # theta convert to direction -> start in 180 dgree (clockwize)
#         _tmp_holder = []
#         _theta = theta.copy()
#         for _ct in range(sector):
#             _th = np.pi - (_bais + _term * _ct)
#             _filterd = (_theta >= _th)
#             _tmp_holder.append(_filterd)
#             _theta = _theta - _theta * _filterd

#         _tmp_holder[0] = np.logical_or(_tmp_holder[0], theta == _theta)
#         _tmp_holder = Custom_Array_Process.stack(_tmp_holder)
#         _direction = np.argmax(_tmp_holder, -1)

#         return _direction

#     @staticmethod
#     def _image_shift(image: ndarray, direction: int, step_size: int = 1):
#         holder = np.zeros_like(image)
#         if direction == 0:
#             holder[:, :-step_size] = image[:, step_size:]  # left
#         elif direction == 1:
#             holder[:-step_size, :-step_size] = image[step_size:, step_size:]  # left top
#         elif direction == 2:
#             holder[:-step_size, :] = image[step_size:, :]  # top
#         elif direction == 3:
#             holder[:-step_size, step_size:] = image[step_size:, :-step_size]  # right top
#         elif direction == 4:
#             holder[:, step_size:] = image[:, :-step_size]  # right
#         elif direction == 5:
#             holder[step_size:, step_size:] = image[:-step_size, :-step_size]  # right down
#         elif direction == 6:
#             holder[step_size:, :] = image[:-step_size, :]  # down
#         elif direction == 7:
#             holder[step_size:, :-step_size] = image[:-step_size, step_size:]  # left down

#         return holder

#     @staticmethod
#     def _conver_to_last_channel(image: ndarray):
#         img_shape = image.shape
#         if len(img_shape) == 2:
#             # gray iamge
#             return image[:, :, np.newaxis]
#         else:
#             # else image
#             divide_data = [image[ct] for ct in range(img_shape[0])]
#             return Custom_Array_Process.stack(divide_data)

#     @staticmethod
#     def _conver_to_first_channel(image: ndarray):
#         img_shape = image.shape
#         if len(img_shape) == 2:
#             # gray iamge
#             return image[np.newaxis, :, :]
#         else:
#             # else image
#             divide_data = [image[:, :, ct] for ct in range(img_shape[-1])]
#             return Custom_Array_Process.stack(divide_data, 0)

#     @staticmethod
#     def _direction_check(object_data: ndarray, direction_array: ndarray, check_list: List[int], is_bidirectional: bool = True):
#         filtered = np.zeros_like(object_data)
#         for _direction in check_list:
#             _tmp_direction_check = np.ones_like(object_data)
#             _label = Image_Process._image_shift(object_data, _direction, 1)
#             _tmp_direction_check *= (object_data * (direction_array == _direction)) > _label
#             if is_bidirectional:
#                 _label = Image_Process._image_shift(object_data, _direction + 4, 1)
#                 _tmp_direction_check *= (object_data * (direction_array == _direction)) > _label

#             filtered += _tmp_direction_check

#         return filtered

#     @staticmethod
#     def _color_finder(image: ndarray, color_list: List[Union[int, Tuple[int, int, int]]]):
#         _holder: List[ndarray] = []
#         for _color in color_list:
#             _finded = np.all((image == _color), 2) if isinstance(_color, list) else (image == _color)
#             _holder.append(_finded)

#         return _holder[0].astype(int) if len(_holder) == 1 else np.logical_or(*[data for data in _holder])

#     @staticmethod
#     def _classfication_resize(original: ndarray, size: List[int]):
#         _new = Custom_Array_Process._Make_array(size + [original.shape[-1], ], 0)

#         _pos = np.where(original == 1)
#         _new_pos = []
#         for _ct in range(len(_pos) - 1):
#             _new_pos.append(np.round((size[_ct] - 1) * _pos[_ct] / original.shape[_ct]).astype(int))
#         _new_pos.append(_pos[-1])

#         _new[tuple(_new_pos)] = 1

#         return _new

#     # @staticmethod
#     # def _string_to_img(string: str, shape: Union[Tuple[int, int], Tuple[int, int, int]]):
#     #     return np.fromstring(string, dtype=np.uint8).reshape(shape)
