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
from typing import (Type, TypeVar, Generic)

from enum import Enum, auto, IntEnum
from dataclasses import dataclass

# import math
import cv2
import numpy as np
from numpy.typing import NDArray

from .system import Path, String


class Viewpoints():
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
    class Img_Group():
        def Set_image(self, **frame_images: NDArray):
            raise NotImplementedError

        def Get_image(self):
            raise NotImplementedError

    IMG_GROUP = TypeVar(
        "IMG_GROUP",
        bound=Img_Group
    )

    class Camera(Generic[IMG_GROUP]):
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
        def __init__(
            self,
            cam_id: int,
            fps: float,
            shape: tuple[int, int, int],
        ) -> None:
            self.cam_id: int = cam_id

            self.fps: float = fps
            self.shape: tuple[int, int, int] = shape

            self._intrinsic: NDArray = np.eye(3)

        @property
        def Intrinsic(self):
            return self._intrinsic

        @Intrinsic.setter
        def Intrinsic(self, extrinsic_array: NDArray):
            assert len(extrinsic_array.shape) in (1, 2)

            if len(extrinsic_array.shape) == 1:
                _array = extrinsic_array.reshape(3, 3)
            else:
                _array = extrinsic_array

            _h, _w = _array.shape[:2]

            _sp_h = (_h - 3) if _h < 3 else 3
            _sp_w = (_w - 3) if _w < 3 else 3

            self.Intrinsic[:_sp_h, :_sp_w] = _array

        @property
        def Cams_parameters(self):
            ...

        @Cams_parameters.setter
        def Cams_parameters(self):
            ...

        # def Get_camera_params(self, **kwarg) -> dict[str, Any]:
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
        #     _hoder = {}
        #     for _key in kwarg:
        #         if _key == "fx":
        #             ...
        #         elif _key == "fy":
        #             ...
        #         elif _key == "cx":
        #             ...
        #         elif _key == "cy":
        #             ...

        #     return _hoder

        def Get_data(self):
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
            # get images
            raise NotImplementedError

    class Scene(Generic[IMG_GROUP]):
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
        def __init__(
            self,
            cam_info: Viewpoints.Camera,
            frame_id: int = 0
        ):
            self.cams_info = cam_info
            self._frame_id: int = frame_id

            self._extrinsic: NDArray = np.eye(4)
            self._rotate_order: str = "xyz"

            self.image_block: Viewpoints.IMG_GROUP = Viewpoints.Img_Group()

        @property
        def Frame_id(self):
            """ ### 연속된 장면에서 현재 프레임의 번호를 반환하는 함수"""
            return self._frame_id

        @Frame_id.setter
        def Frame_id(self, frame_id: int):
            if frame_id < 0:
                raise ValueError("Frame id MUST 0 or higher")

            self._frame_id = frame_id

        @property
        def Extrinsic(self):
            """ ### 장면의 위치 정보를 반환하는 함수"""
            return self._extrinsic

        @Extrinsic.setter
        def Extrinsic(self, extrinsic_array: NDArray):
            assert len(extrinsic_array.shape) in (1, 2)

            if len(extrinsic_array.shape) == 1:
                _array = extrinsic_array.reshape(-1, 4)
            else:
                _array = extrinsic_array

            _h, _w = _array.shape[:2]

            _sp_h = (_h - 4) if _h < 4 else 4
            _sp_w = (_w - 4) if _w < 4 else 4

            self.Extrinsic[:_sp_h, :_sp_w] = _array

        @property
        def Cam_to_world(self):
            return np.linalg.inv(self._extrinsic)

        @property
        def World_to_Cam(
            self,
            translate: NDArray = np.zeros(3),
            scale: float = 1.0
        ):
            _c2w = np.linalg.inv(self.Extrinsic)
            cam_center = _c2w[:3, 3]
            cam_center = (cam_center + translate) * scale
            _c2w[:3, 3] = cam_center
            return np.linalg.inv(self.Extrinsic)

        @property
        def Images(self):
            return self.image_block.Get_image()

        @Images.setter
        def Images(self, **frame_images: NDArray):
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
            self.image_block.Set_image(**frame_images)

        def Set_rotate_from_angle(
            self, is_override: bool = False, **delta: tuple[float, bool]
        ):
            _rotate_order = self._rotate_order
            _rotate_metrix = np.eye(
                3
            ) if is_override else self.Extrinsic[:3, :3]

            for _k in _rotate_order:
                if _rotate_order in delta:
                    _rotate_metrix @= Vision_Toolbox.Get_axis_rotate(
                        delta[_k][0], Flag.Axis(_k), delta[_k][1]
                    )

            self.Extrinsic[:3, :3] = _rotate_metrix

        def Set_rotate_from_list(self, rotate: list[float]):
            if len(rotate) == 9:
                _array = np.array(rotate[:9])
                self.Extrinsic[:3, :3] = _array.reshape((3, 3))
            else:
                raise ValueError(
                    "Rotate array's entry size must be 9"
                )

        def Set_transfer_from_list(self, tansfer: list[float]):
            if len(tansfer) >= 3:
                _array = np.array(tansfer[:3])
                self.Extrinsic[3, :3] = _array.reshape((3, 1))
            else:
                raise ValueError(
                    "Rotate array's entry size must be 3"
                )

        def Get_cams_info(self):
            return self.cams_info.Cams_parameters


class Flag():
    class Aligan(IntEnum):
        """ ### Flag for img or text in visualize process"""
        Left = auto()
        Center = auto()
        Right = auto()

    class Axis(String.String_Enum):
        X = auto()
        Y = auto()
        Z = auto()

    class Padding(Enum):
        """ ### Data type for each data"""
        ZERO = cv2.BORDER_CONSTANT
        OUTLINE = cv2.BORDER_REPLICATE
        REFLECT = cv2.BORDER_REFLECT
        REFLECT_101 = cv2.BORDER_REFLECT101

    class Convert(Enum):
        """ ### Data type for each data"""
        BGR2RGB = cv2.COLOR_BGR2RGB
        BGR2GRAY = cv2.COLOR_BGR2GRAY
        RGB2BGR = cv2.COLOR_RGB2BGR
        BGRA2RGBA = cv2.COLOR_BGRA2RGBA

    class Codex(Enum):
        """ ### Video codex for each video extention"""
        MP4 = "DIVX"

    class Format(Enum):
        """ ### Data type for each data"""
        UNCHANGE = cv2.IMREAD_UNCHANGED
        BGR = cv2.IMREAD_COLOR
        GRAY = cv2.IMREAD_GRAYSCALE


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
        data_format: Flag.Format | None = None
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
        convert_flag: Flag.Convert = Flag.Convert.BGR2RGB
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
        is_row_first: bool = True,
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

        _img_ct = len(img_list)

        if not row:
            if not col:
                _e_massage = "Must set value of the image array's row or col"
                raise ValueError(_e_massage)
            _col = col
            _row = _img_ct // col + int(_img_ct % col)
        else:
            _row = row
            _col = _img_ct // row + int(_img_ct % row) if col is None else col

        if _col * _row < _img_ct:
            _size_error = "{} images can't set in {}x{} array"
            raise ValueError(_size_error.format(_img_ct, _row, _col))

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
            _num_r = _col if is_row_first else _col * _ct_r

            for _ct_c in range(_col):
                _lt_w = _ct_c * (_vm_pad + img_w) + _lf_pad
                _rb_w = _lt_w + img_w
                _num_c = _ct_c * _row if is_row_first else _ct_c

                _img = img_list[_num_r + _num_c]

                try:
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
