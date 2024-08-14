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
from typing import (
    Type, Any
    # Literal
)
from dataclasses import dataclass, field

# import math
import cv2
import numpy as np
from numpy.typing import NDArray

from .system import Path  # , File


class Camera_Model():
    """ ### Description of class functionality
    Note

    ---------------------------------------------------------------------
    ### Args
    - Super
        - `arg_name`: Description of the input argument from parents
    - This
        - `arg_name`: Description of the input argument

    ### Attributes
    - `attribute_name`: Description of the class attribute name

    ### Structure
    - `SubClassName` or `Function_name`: Description of each object

    ### Todo
    - `SubClassName` or `Function_name`: Description of each object

    """
    @dataclass
    class Camera():
        """ ### Description of class functionality
        Note

        ---------------------------------------------------------------------
        ### Args
        - Super
            - `arg_name`: Description of the input argument from parents
        - This
            - `arg_name`: Description of the input argument

        ### Attributes
        - `attribute_name`: Description of the class attribute name

        ### Structure
        - `SubClassName` or `Function_name`: Description of each object

        ### Todo
        - `SubClassName` or `Function_name`: Description of each object

        """
        cam_id: int = 0
        fps: int = 30
        intrinsic: NDArray = field(default_factory=lambda: np.eye(3))

        scene_list: list[Camera_Model.Scene] = field(default_factory=list)

        def Set_intrinsic(self, intrinsic_array: NDArray):
            _h, _w = intrinsic_array.shape

            _sp_h = (_h - 3) if (_h - 3) else None
            _sp_w = (_w - 3) if (_w - 3) else None
            self.intrinsic[:_sp_h, :_sp_w] = intrinsic_array

        def Get_camera_params(self, **kwarg) -> dict[str, Any]:
            """ ### Function feature description
            Note

            ------------------------------------------------------------------
            ### Args
            - `arg_name`: Description of the input argument

            ### Returns or Yields
            - `data_format`: Description of the output argument

            ### Raises
            - `error_type`: Method of handling according to error issues

            """
            _hoder = {}
            for _key in kwarg:
                if _key == "fx":
                    ...
                elif _key == "fy":
                    ...
                elif _key == "cx":
                    ...
                elif _key == "cy":
                    ...

            return _hoder

        def Capture(self) -> Camera_Model.Scene:
            """ ### Function feature description
            Note

            ------------------------------------------------------------------
            ### Args
            - `arg_name`: Description of the input argument

            ### Returns or Yields
            - `data_format`: Description of the output argument

            ### Raises
            - `error_type`: Method of handling according to error issues

            """
            raise NotImplementedError

        # def Get_projection_from_f_length(self):
        #     """ ### Function feature description
        #     Note

        #     ------------------------------------------------------------------
        #     ### Args
        #     - `arg_name`: Description of the input argument

        #     ### Returns or Yields
        #     - `data_format`: Description of the output argument

        #     ### Raises
        #     - `error_type`: Method of handling according to error issues

        #     """
        #     _projection = np.zeros((3, 3))
        #     _projection[0, 0] = self.fx
        #     _projection[1, 1] = self.fy
        #     _projection[0, 2] = self.cx
        #     _projection[1, 2] = self.cy

        #     return _projection

        # def Get_projection_from_fov(self):
        #     raise NotImplementedError

        # def Get_perspective_projection_from_f_length(
        #     self,
        #     znear: int,
        #     zfar: int
        # ):
        #     """ ### Function feature description
        #     Note

        #     ------------------------------------------------------------------
        #     ### Args
        #     - `arg_name`: Description of the input argument

        #     ### Returns or Yields
        #     - `data_format`: Description of the output argument

        #     ### Raises
        #     - `error_type`: Method of handling according to error issues

        #     """
        #     # _tan_half_FoV_y = math.tan((self.fov_y / 2))
        #     # _tan_half_FoV_x = math.tan((self.fov_x / 2))

        #     # _projection = np.zeros((4, 4))

        #     # top = _tan_half_FoV_y * znear
        #     # bottom = -top
        #     # right = _tan_half_FoV_x * znear
        #     # left = -right

        #     # z_sign = 1.0

        #     # _projection[0, 0] = 2.0 * znear / (right - left)
        #     # _projection[1, 1] = 2.0 * znear / (top - bottom)
        #     # _projection[0, 2] = (right + left) / (right - left)
        #     # _projection[1, 2] = (top + bottom) / (top - bottom)
        #     # _projection[3, 2] = z_sign
        #     # _projection[2, 2] = z_sign * zfar / (zfar - znear)
        #     # _projection[2, 3] = -(zfar * znear) / (zfar - znear)

        #     raise NotImplementedError

    @dataclass
    class Scene():
        """ ### Description of class functionality
        Note

        ---------------------------------------------------------------------
        ### Args
        - Super
            - `arg_name`: Description of the input argument from parents
        - This
            - `arg_name`: Description of the input argument

        ### Attributes
        - `attribute_name`: Description of the class attribute name

        ### Structure
        - `SubClassName` or `Function_name`: Description of each object

        ### Todo
        - `SubClassName` or `Function_name`: Description of each object

        """
        cam: Camera_Model.Camera
        frame_id: int

        extrinsic: NDArray = field(default_factory=lambda: np.eye(4))
        rotate_order: str = ""
        size: tuple[int, int] = (0, 0)
        images: dict[str, NDArray] = field(
            default_factory=lambda: {
                "rgb": np.empty(0),
                "depth": np.empty(0)
            })

        # rotate: NDArray = field(
        #     default_factory=lambda: np.empty((3, 3)))
        # transfer: NDArray = field(
        #     default_factory=lambda: np.empty((3, 1)))

        def Set_frame_id(self, frame_id: int):
            if frame_id < 0:
                raise ValueError("Frame id MUST 0 or higher")

            self.frame_id = frame_id

        def Set_extrinsic(self, extrinsic_array: NDArray):
            _h, _w = extrinsic_array.shape

            _sp_h = (_h - 4) if (_h - 4) else None
            _sp_w = (_w - 4) if (_w - 4) else None
            self.extrinsic[:_sp_h, :_sp_w] = extrinsic_array

        def Set_size(self, height: int, width: int):
            if height <= 0 or width <= 0:
                raise ValueError("Image size MUST BIGGER THAN 0")

            self.size = (height, width)

        def Set_images(self, is_update: bool = False, **frame_images: NDArray):
            """ ### Function feature description
            Note

            ------------------------------------------------------------------
            ### Args
            - `arg_name`: Description of the input argument

            ### Returns or Yields
            - `data_format`: Description of the output argument

            ### Raises
            - `error_type`: Method of handling according to error issues

            """
            _size = self.size

            if is_update:
                # update
                _images_key = self.images.keys()
                _holder = dict(
                    (_k, _img) for _k, _img in frame_images.items() if (
                        _img.shape == _size and _k in _images_key
                    )
                )
                self.images.update(_holder)

            else:
                # override
                self.images = dict(
                    (_k, _img) for _k, _img in frame_images.items() if (
                        _img.shape == _size
                    )
                )

        def Get_scene_params(self, **kwarg) -> dict[str, Any]:
            """ ### Function feature description
            Note

            ------------------------------------------------------------------
            ### Args
            - `arg_name`: Description of the input argument

            ### Returns or Yields
            - `data_format`: Description of the output argument

            ### Raises
            - `error_type`: Method of handling according to error issues

            """
            _hoder = {}
            for _k, _v in kwarg.items():
                if _k == "rotate_matrix":
                    _hoder[_k] = self.extrinsic[:3, :3]
                if _k == "rotate_angle":
                    _angle = Vision_Toolbox.Get_angle_from_rotate_matrix(
                        self.extrinsic[:3, :3]
                    )[0]
                    _hoder[_k] = [
                        _angle[0] * 180 / np.pi,
                        _angle[1] * 180 / np.pi,
                        _angle[2] * 180 / np.pi,
                    ]
                elif _k == "transfer":
                    _hoder[_k] = self.extrinsic[:3, 3]
                elif _k == "size":
                    ...
                elif _k == "height":
                    ...
                elif _k == "width":
                    ...

            return _hoder

        def Set_rotate_from_list(self, rotate: list[float]):
            if len(rotate) >= 9:
                _array = np.array(rotate[:9])
                self.rotate = _array.reshape((3, 3))
            else:
                raise ValueError(
                    "Rotate array's entry size must be bigger then 9"
                )

        def Set_transfer_from_list(self, tansfer: list[float]):
            if len(tansfer) >= 3:
                _array = np.array(tansfer[:3])
                self.transfer = _array.reshape((3, 1))
            else:
                raise ValueError(
                    "Rotate array's entry size must be bigger then 3"
                )

        def Get_W2C(self, shift: NDArray = np.array([.0, .0, .0]), scale=1.0):
            """ ### Function feature description
            Note

            ------------------------------------------------------------------
            ### Args
            - `arg_name`: Description of the input argument

            ### Returns or Yields
            - `data_format`: Description of the output argument

            ### Raises
            - `error_type`: Method of handling according to error issues

            """
            _rt = np.eye(4)
            _rt[:3, :3] = self.rotate.transpose()
            _rt[:3, 3] = self.transfer

            _c2w = np.linalg.inv(_rt)
            _c2w[:3, 3] = (_c2w[:3, 3] + shift.transpose()) * scale

            return np.float32(np.linalg.inv(_c2w))

        # def Rotate_to_angle(self):
        #     """ ### Function feature description
        #     Note

        #     ------------------------------------------------------------------
        #     ### Args
        #     - `arg_name`: Description of the input argument

        #     ### Returns or Yields
        #     - `data_format`: Description of the output argument

        #     ### Raises
        #     - `error_type`: Method of handling according to error issues

        #     """
        #     raise NotImplementedError


class Flag():
    class Aligan(Enum):
        """ ### Flag for img or text in visualize process"""
        Left = 0
        Center = 1
        Right = 2

    class Axis(Enum):
        X = "x"
        Y = "y"
        Z = "z"

    class Padding(Enum):
        """ ### Data type for each data"""
        ZERO = cv2.BORDER_CONSTANT
        OUTLINE = cv2.BORDER_REPLICATE
        REFLECT = cv2.BORDER_REFLECT
        REFLECT_101 = cv2.BORDER_REFLECT101


class Codex_For(Enum):
    """ ### Video codex for each video extention"""
    MP4 = "DIVX"


class Data_Format(Enum):
    """ ### Data type for each data"""
    UNCHANGE = cv2.IMREAD_UNCHANGED
    BGR = cv2.IMREAD_COLOR
    GRAY = cv2.IMREAD_GRAYSCALE


class Convert_Flag(Enum):
    """ ### Data type for each data"""
    BGR2RGB = cv2.COLOR_BGR2RGB
    BGR2GRAY = cv2.COLOR_BGR2GRAY
    RGB2BGR = cv2.COLOR_RGB2BGR


class File_IO():
    """ ### Description of class functionality
    Note

    ---------------------------------------------------------------------
    ### Args
    - Super
        - `arg_name`: Description of the input argument from parents
    - This
        - `arg_name`: Description of the input argument

    ### Attributes
    - `attribute_name`: Description of the class attribute name

    ### Structure
    - `SubClassName` or `Function_name`: Description of each object

    ### Todo
    - `SubClassName` or `Function_name`: Description of each object

    """
    @staticmethod
    def Get_img_file_list_from_dir(
        data_dir: str,
        keyword: str,
        ext: str | list[str]
    ):
        """ ### Function feature description
        Note

        ------------------------------------------------------------------
        ### Args
        - `arg_name`: Description of the input argument

        ### Returns or Yields
        - `data_format`: Description of the output argument

        ### Raises
        - `error_type`: Method of handling according to error issues

        """
        return Path.Search(data_dir, Path.Type.FILE, keyword, ext_filter=ext)

    @staticmethod
    def File_to_img(
        file_name: str,
        data_format: Data_Format | None = None
    ):
        """ ### Function feature description
        Note

        ------------------------------------------------------------------
        ### Args
        - `arg_name`: Description of the input argument

        ### Returns or Yields
        - `data_format`: Description of the output argument

        ### Raises
        - `error_type`: Method of handling according to error issues

        """
        return cv2.imread(
            file_name,
            cv2.IMREAD_UNCHANGED if data_format is None else data_format.value
        )

    @staticmethod
    def Img_to_file(file_name: str, image: NDArray):
        """ ### Function feature description
        Note

        ------------------------------------------------------------------
        ### Args
        - `arg_name`: Description of the input argument

        ### Returns or Yields
        - `data_format`: Description of the output argument

        ### Raises
        - `error_type`: Method of handling according to error issues

        """
        return cv2.imwrite(file_name, image)

    # in later fix it
    # class RLE():
    #     @staticmethod
    #    def _from_nparray(
    #        data: NDArray,
    #        order: Literal['A', 'C', 'F'] = 'F'
    #    ):
    #         if data is not None:
    #             _size = data.shape
    #             _size = (int(_size[0]), int(_size[1]))

    #             return_RLE = []
    #             # zeros
    #             if (data == np.zeros_like(data)).all():
    #                 return_RLE.append(_size[0] * _size[1])
    #             # ones
    #             elif (data == np.ones_like(data)).all():
    #                 return_RLE.append(0)
    #                 return_RLE.append(_size[0] * _size[1])
    #             # else
    #             else:
    #                 _line = data.reshape(_size[0] * _size[1], order=order)
    #                 _count_list = []

    #                 # in later add annotation
    #                 for _type in range(2):
    #                     _points = np.where(_line == _type)[0]
    #                     _filter = _points[1:] - _points[:-1]
    #                     _filter = _filter[np.where(_filter != 1)[0]]
    #                     _count = _filter[np.where(_filter != 1)[0]] - 1

    #                     if _points[0]:
    #                         _count = np.append((_points[0], ), _count)
    #                     _count_list.append(_count)

    #                 _one_count, _zero_count = _count_list

    #                 if _line[0]:
    #                     _zero_count = np.append((0, ), _zero_count)

    #                 for _ct in range(len(_one_count)):
    #                     return_RLE.append(int(_zero_count[_ct]))
    #                     return_RLE.append(int(_one_count[_ct]))

    #                 _last_count = int(len(_line) - sum(return_RLE))
    #                 return_RLE.append(_last_count)

    #             return {"size": _size, "counts": return_RLE}

    #         else:
    #             return None

    #     @staticmethod
    #     def _to_nparray(data, order: Literal['A', 'C', 'F'] = "F"):
    #         if data is not None:
    #             _rle_data = data["counts"]
    #             _list = []
    #             for _type, _count in enumerate(_rle_data):
    #                 [_list.append(_type % 2) for _ct in range(_count)]
    #             _list = np.reshape(_list, data["size"], order)
    #             return _list
    #         else:
    #             return None


class Vision_Toolbox():
    """ ### Description of class functionality
    Note

    ---------------------------------------------------------------------
    ### Args
    - Super
        - `arg_name`: Description of the input argument from parents
    - This
        - `arg_name`: Description of the input argument

    ### Attributes
    - `attribute_name`: Description of the class attribute name

    ### Structure
    - `SubClassName` or `Function_name`: Description of each object

    ### Todo
    - `SubClassName` or `Function_name`: Description of each object

    """
    @staticmethod
    def Format_converter(
        image: NDArray,
        convert_flag: Convert_Flag = Convert_Flag.BGR2RGB
    ):
        """ ### Function feature description
        Note

        ------------------------------------------------------------------
        ### Args
        - `arg_name`: Description of the input argument

        ### Returns or Yields
        - `data_format`: Description of the output argument

        ### Raises
        - `error_type`: Method of handling according to error issues

        """
        return cv2.cvtColor(image, convert_flag.value)

    @staticmethod
    def Set_filter(
        src: NDArray[np.uint8],
        kennel: NDArray[np.int_ | np.float_],
        border_type: Flag.Padding = Flag.Padding.REFLECT_101
    ) -> NDArray:
        return cv2.filter2D(
            src,
            ddepth=-1,
            kernel=kennel,
            borderType=border_type.value
        )

    @staticmethod
    def Get_axis_rotate(delta: float, aixs: Flag.Axis, is_rad: bool = False):
        """ ### 축 회전 변환 행렬
        Note

        ------------------------------------------------------------------
        ### Args
        - `delta`: 회전한 각도
        - `is_rad`: 입력된 값의 성격 (true = radian, false = angle)

        ### Returns or Yields
        - `rotate_matrix`: 변환 행렬
        """
        _delta = delta if is_rad else np.pi * (delta / 180)
        _s, _c = np.sin(_delta), np.cos(_delta)

        if aixs == Flag.Axis.X:
            return np.array([[1, 0, 0], [0, _c, -_s], [0, _s, _c]])
        if aixs == Flag.Axis.Y:
            return np.array([[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]])
        return np.array([[_c, -_s, 0], [_s, _c, 0], [0, 0, 1]])

    @staticmethod
    def Get_roate_matrix(
        delta: dict[Flag.Axis, float],
        is_rad: bool = False
    ):
        _matrix = np.eye(3, 3)
        for _order, _value in delta.items():
            _matrix = _matrix @ Vision_Toolbox.Get_axis_rotate(
                _value, _order, is_rad
            )

        return _matrix

    @staticmethod
    def Get_angle_from_rotate_matrix(rotate: NDArray):
        if rotate[2, 0] != 1 and rotate[2, 0] != -1:
            _holder: list[tuple[float, float, float]] = []

            _th_1 = -np.arcsin(rotate[2, 0])
            _th_2 = np.pi - _th_1

            for _th in [_th_1, _th_2]:
                _tan_ps_1 = rotate[2, 1] / np.cos(_th)
                _tan_ps_2 = rotate[2, 2] / np.cos(_th)
                _ps = np.arctan2(_tan_ps_1, _tan_ps_2)

                _tan_ph_1 = rotate[1, 0] / np.cos(_th)
                _tan_ph_2 = rotate[0, 0] / np.cos(_th)
                _ph = np.arctan2(_tan_ph_1, _tan_ph_2)

                _holder.append((_ps, _th, _ph))

            return _holder[0], _holder[1]

        _ph = 0  # can set to anything, it's the gimbal lock case
        if rotate[2, 0] == -1:
            _th = np.pi / 2
            _ps = _ph + np.arctan2(rotate[0, 1], rotate[0, 2])
        else:
            _th = -np.pi / 2
            _ps = -_ph + np.arctan2(-rotate[0, 1], -rotate[0, 2])

        return (_ps, _th, _ph), (None, None, None)


class Debugging():
    """ ### Description of class functionality
    Note

    ---------------------------------------------------------------------
    ### Args
    - Super
        - `arg_name`: Description of the input argument from parents
    - This
        - `arg_name`: Description of the input argument

    ### Attributes
    - `attribute_name`: Description of the class attribute name

    ### Structure
    - `SubClassName` or `Function_name`: Description of each object

    ### Todo
    - `SubClassName` or `Function_name`: Description of each object

    """
    @staticmethod
    def Put_text_to_img(
        draw_img: NDArray,
        location: list[int],
        text_list: list[str],
        padding: int
    ):
        """ ### Function feature description
        Note

        ------------------------------------------------------------------
        ### Args
        - `arg_name`: Description of the input argument

        ### Returns or Yields
        - `data_format`: Description of the output argument

        ### Raises
        - `error_type`: Method of handling according to error issues

        """
        # draw image check
        _shape = draw_img.shape
        _h, _w = _shape[:2]

        # decide text position in image
        _p_y, _p_x = location
        _txt_h = 0
        _txt_w = 0

        for _text in text_list:  # get text box size
            (_size_x, _size_y), _ = cv2.getTextSize(_text, 1, 1, 1)
            _txt_h += _size_y
            _txt_w = _size_x if _size_x > _txt_w else _txt_w

        _is_over_h = 2 * (_txt_h + padding) > _h
        _is_over_w = 2 * (_txt_w + padding) > _w

        if _is_over_h or _is_over_w:  # can't put text box in image
            return False, draw_img
        else:  # default => right under
            # set text box x position
            if (_p_x + padding + _txt_w) < _w:
                _text_x = _p_x + padding
            else:
                _text_x = _p_x - (_txt_w + padding)
            # set text box y position
            if (_p_y + _txt_h + padding) < _h:
                _text_y = _p_y + _txt_h + padding
            else:
                _text_y = _p_y - padding

            for _ct, _text in enumerate(reversed(text_list)):
                cv2.putText(
                    draw_img,
                    _text,
                    [_text_x, _text_y - (_ct * _size_y)],
                    1,
                    fontScale=1,
                    color=(0, 255, 0)
                )
            return True, draw_img

    @staticmethod
    def Image_caption(
        image: NDArray,
        caption: str,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        aligan: Flag.Aligan = Flag.Aligan.Center
    ):
        """ ### Function feature description
        Note

        ------------------------------------------------------------------
        ### Args
        - `arg_name`: Description of the input argument

        ### Returns or Yields
        - `data_format`: Description of the output argument

        ### Raises
        - `error_type`: Method of handling according to error issues

        """
        _p = 10
        _i_h, _i_w = image.shape[:2]

        # make the text box
        _t_w, _t_h = cv2.getTextSize(caption, font, 1, 2)[0]
        _t_holder = np.ones(
            (_t_h + _p, _t_w, 3),
            dtype=np.uint8
        ) * 255
        cv2.putText(_t_holder, caption, (0, _t_h), font, 1, (0, 0, 0), 2)

        if _t_w > (_i_w - 2 * _p):
            _t_h, _t_w = round((_i_w - 2 * _p) / _t_w * _t_h), _i_w - (2 * _p)
            _t_holder: NDArray = cv2.resize(
                _t_holder,
                (_t_w, _t_h),
                interpolation=cv2.INTER_CUBIC
            )

        # make the captioned image
        _draw = np.ones(
            (_i_h + _t_h + (2 * _p), _i_w, 3),
            dtype=np.uint8
        ) * 255
        _draw[:_i_h] = image

        if aligan is Flag.Aligan.Center:
            _l_x = (_i_w - _t_w) // 2
        elif aligan is Flag.Aligan.Left:
            _l_x = _p
        else:  # Right
            _l_x = _i_w - _p - _t_w

        _draw[_i_h + _p: _i_h + _t_h + _p, _l_x: _l_x + _t_w, :] = _t_holder

        return _draw

    @staticmethod
    def Organize_the_image_array(
        img_list: list[NDArray | None],
        img_h: int, img_w: int,
        row: int | None, col: int | None = None,
        padding: int | tuple[int, int] | tuple[int, int, int, int] = 10,
        is_fill_row_first: bool = True,
        background: NDArray = 255 * np.ones(3),
        canvas_format: Type = np.uint8
    ):
        """ ### Function feature description
        Note

        ------------------------------------------------------------------
        ### Args
        - `arg_name`: Description of the input argument

        ### Returns or Yields
        - `data_format`: Description of the output argument

        ### Raises
        - `error_type`: Method of handling according to error issues

        """
        # set the row, col size
        if row is None:
            if col is None:
                raise ValueError(
                    "Must set value of the image array's row or col"
                )
            _col = col
            _row = len(img_list) // col + int(len(img_list) % col != 0)
        else:
            if col is None:
                _col = len(img_list) // row + int(len(img_list) % row != 0)
                _row = row
            else:
                _col = col
                _row = row

        if _col * _row < len(img_list):
            _size_error = "{} images can't set in {}x{} array"
            raise ValueError(_size_error.format(len(img_list), _row, _col))

        # set the canvas
        if isinstance(padding, int):
            _lf_pad, _hm_pad = padding, padding
            _up_pad, _vm_pad = padding, padding
        elif len(padding) == 2:  # vertical, horizental padding
            _lf_pad, _hm_pad = padding[0], padding[0]
            _up_pad, _vm_pad = padding[1], padding[1]
        else:  # upper, lower, right, left padding
            _lf_pad, _hm_pad = padding[3], padding[3] + padding[2]
            _up_pad, _vm_pad = padding[0], padding[0] + padding[1]

        _canvas_shape = (
            (img_h + _hm_pad) * _row + _hm_pad,
            (img_w + _vm_pad) * _col + _vm_pad,
            3
        )
        _canvas = (np.ones(_canvas_shape) * background).astype(canvas_format)

        # img set to canvas
        for _ct_r in range(_row):
            _lt_h = _ct_r * (_hm_pad + img_h) + _up_pad
            _rb_h = _lt_h + img_h
            _num_r = _col if is_fill_row_first else _col * _ct_r

            for _ct_c in range(_col):
                _lt_w = _ct_c * (_vm_pad + img_w) + _lf_pad
                _rb_w = _lt_w + img_w
                _num_c = _ct_c * _row if is_fill_row_first else _ct_c

                try:
                    _img = img_list[_num_r + _num_c]
                    if _img is not None and len(_img.shape) == 3:  # color img
                        _canvas[_lt_h: _rb_h, _lt_w: _rb_w] = _img
                    elif _img is not None:  # gray scale img
                        _canvas[_lt_h: _rb_h, _lt_w: _rb_w] = np.stack(
                            [_img, _img, _img],
                            axis=-1
                        )
                except IndexError:
                    continue

        return _canvas

    class Visualize():
        """ ### Description of class functionality
        Note

        ---------------------------------------------------------------------
        ### Args
        - Super
            - `arg_name`: Description of the input argument from parents
        - This
            - `arg_name`: Description of the input argument

        ### Attributes
        - `attribute_name`: Description of the class attribute name

        ### Structure
        - `SubClassName` or `Function_name`: Description of each object

        ### Todo
        - `SubClassName` or `Function_name`: Description of each object

        """
        @staticmethod
        def Point_cloud():
            """ ### Function feature description
            Note

            ------------------------------------------------------------------
            ### Args
            - `arg_name`: Description of the input argument

            ### Returns or Yields
            - `data_format`: Description of the output argument

            ### Raises
            - `error_type`: Method of handling according to error issues

            """
            raise NotImplementedError
