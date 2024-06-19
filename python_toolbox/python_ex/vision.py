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
from typing import List, Dict, Tuple, Type
from dataclasses import dataclass, field

# import math
import cv2
import numpy as np
from numpy import ndarray

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
        cam_id: int
        fps: int

        fov_x: float = -1.
        fov_y: float = -1.

        fx: float = -1.
        fy: float = -1.

        width: int = 0
        hight: int = 0

        cx: float = 0
        cy: float = 0

        intrinsic: ndarray = field(default_factory=lambda: np.eye(4))

        def Set_camera_info(self, **kwarg):
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

        def Get_camera_info(self, **kwarg):
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

        def Get_projection_from_f_length(self):
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
            _projection = np.zeros((3, 3))
            _projection[0, 0] = self.fx
            _projection[1, 1] = self.fy
            _projection[0, 2] = self.cx
            _projection[1, 2] = self.cy

            return _projection

        def Get_projection_from_fov(self):
            raise NotImplementedError

        def Get_perspective_projection_from_f_length(
            self,
            znear: int,
            zfar: int
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
            # _tan_half_FoV_y = math.tan((self.fov_y / 2))
            # _tan_half_FoV_x = math.tan((self.fov_x / 2))

            # _projection = np.zeros((4, 4))

            # top = _tan_half_FoV_y * znear
            # bottom = -top
            # right = _tan_half_FoV_x * znear
            # left = -right

            # z_sign = 1.0

            # _projection[0, 0] = 2.0 * znear / (right - left)
            # _projection[1, 1] = 2.0 * znear / (top - bottom)
            # _projection[0, 2] = (right + left) / (right - left)
            # _projection[1, 2] = (top + bottom) / (top - bottom)
            # _projection[3, 2] = z_sign
            # _projection[2, 2] = z_sign * zfar / (zfar - znear)
            # _projection[2, 3] = -(zfar * znear) / (zfar - znear)

            raise NotImplementedError

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

        rotate: ndarray = field(
            default_factory=lambda: np.empty((3, 3)))
        transfer: ndarray = field(
            default_factory=lambda: np.empty((3, 1)))

        image: Dict[str, ndarray] = field(
            default_factory=lambda: {
                "image": np.empty(0),
                "depth": np.empty(0)
            })

        def _Set_frame_id(self, frame_id: int):
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
            if frame_id < 0:
                raise ValueError(
                    f"frame is must over 0. this frame: {frame_id} is not"
                )
            self.frame_id = frame_id

        def Set_frame_data(self, frame_id: int, **frame_images):
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
            self._Set_frame_id(frame_id)

            self.image.update(
                dict((
                    _key,
                    _img
                ) for _key, _img in frame_images.items() if _key in self.image)
            )

        def Set_rotate_from_list(self, rotate: List[float]):
            if len(rotate) >= 9:
                _array = np.array(rotate[:9])
                self.rotate = _array.reshape((3, 3))
            else:
                raise ValueError(
                    "Rotate array's entry size must be bigger then 9"
                )

        def Set_transfer_from_list(self, tansfer: List[float]):
            if len(tansfer) >= 3:
                _array = np.array(tansfer[:3])
                self.transfer = _array.reshape((3, 1))
            else:
                raise ValueError(
                    "Rotate array's entry size must be bigger then 3"
                )

        def Get_W2C(self, shift: ndarray = np.array([.0, .0, .0]), scale=1.0):
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


class Codex_For(Enum):
    """ ### Video codex for each video extention"""
    MP4 = "DIVX"


class Data_Format(Enum):
    """ ### Data type for each data"""
    BGR = cv2.IMREAD_COLOR
    BGRA = cv2.IMREAD_UNCHANGED
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
        ext: str | List[str]
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
    def Img_to_file(file_name: str, image: ndarray):
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
        image: ndarray,
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


class Aligan(Enum):
    """ ### Flag for img or text in visualize process"""
    Left = 0
    Center = 1
    Right = 2


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
        draw_img: ndarray,
        location: List[int],
        text_list: List[str],
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
        image: np.ndarray,
        caption: str,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        aligan: Aligan = Aligan.Center
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
            _t_holder: np.ndarray = cv2.resize(
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

        if aligan is Aligan.Center:
            _l_x = (_i_w - _t_w) // 2
        elif aligan is Aligan.Left:
            _l_x = _p
        else:  # Right
            _l_x = _i_w - _p - _t_w

        _draw[_i_h + _p: _i_h + _t_h + _p, _l_x: _l_x + _t_w, :] = _t_holder

        return _draw

    @staticmethod
    def Organize_the_image_array(
        img_list: List[ndarray | None],
        img_h: int, img_w: int,
        row: int | None, col: int | None = None,
        padding: int | Tuple[int, int] | Tuple[int, int, int, int] = 10,
        is_row_by_row: bool = True,
        background: ndarray = 255 * np.ones(3),
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
            (img_w + _vm_pad) * _col + _vm_pad
        )
        _canvas = (np.ones(_canvas_shape) * background).astype(canvas_format)

        # img set to canvas
        for _ct_r in range(_row):
            _lt_h = _ct_r * (_hm_pad + img_h) + _up_pad
            _rb_h = _lt_h + img_h
            _num_r = _row * _ct_r if is_row_by_row else _ct_r

            for _ct_c in range(_col):
                _lt_w = _ct_c * (_vm_pad + img_w) + _lf_pad
                _rb_w = _lt_w + img_w
                _num_c = _ct_c if is_row_by_row else _col * _ct_c

                _img = img_list[_num_r + _num_c]

                if _img is not None:
                    _canvas[_lt_h: _rb_h, _lt_w: _rb_w] = _img

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
