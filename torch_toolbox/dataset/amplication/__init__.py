from typing import Literal, Callable, Any
from pathlib import Path

import numpy as np
import cv2

import torch


class Read_Image():
    """Resize sample to given size (width, height).
    -----------------------

    this code base on depth_anthing_v2 resize code
    """

    def __init__(
        self,
        key_list: list[str],
        keep_key_list: list[str] | None = None,
        width: int = -1,
        height: int = -1,
        keep_ratio=True,
        block_size=1,
        bound: Literal["lower", "upper", "minimal"] = "lower",
        use_padding: bool = False,
        interpolation=cv2.INTER_AREA,
    ):
        self.key_list = key_list
        self.keep_key_list = keep_key_list

        self.resize = width != -1 and height != -1

        self.width = width
        self.height = height

        self.block_size = block_size

        self.keep_ratio = keep_ratio
        self.use_padding = use_padding
        if keep_ratio and use_padding:
            if bound != "upper":
                print((
                    "if want use padding resize for multi size image,"
                    "use upper bound"
                ))
            self.bound = "upper"
        else:
            self.bound = bound
        # else:
        #     raise ValueError(f"resize_method '{_b}' not implemented")
        self.interpolation = interpolation

    def Align_to_block(
        self, v: np.ndarray, min_val=0, max_val=None
    ):
        _b = self.block_size
        _x = np.array(v, dtype=np.int32)
        _y = ((_x + _b // 2) // _b) * _b  # block size round

        if max_val is not None:
            _mask = _y > max_val
            if np.any(_mask):
                _y[_mask] = (_x[_mask] // _b) * _b  # block size floor

        _mask = _y < min_val
        if np.any(_mask):
            _y[_mask] = ((_x[_mask] + _b - 1) // _b) * _b  # block size ceil

        return _y

    def Get_size(self, width: list[int] | int, height: list[int] | int):
        # determine new height and width
        _h, _w = self.height, self.width
        _r_h = _h / np.asarray(height, dtype=np.float32)
        _r_w = _w /np.asarray(width, dtype=np.float32)
        _b = self.bound

        if self.keep_ratio:
            if _b == "lower":
                # scale such that output size is lower bound
                _n = np.maximum(_r_h, _r_w)
            elif _b == "upper":
                _n = np.minimum(_r_h, _r_w)
            else:  # _b -> "minimal"
                _n = np.where(np.abs(1 - _r_w) < np.abs(1 - _r_h), _r_w, _r_h)

            _r_h = _r_w = _n

        if _b == "lower":
            _new_h = self.Align_to_block(_r_h * height, min_val=_h)
            _new_w = self.Align_to_block(_r_w * width, min_val=_w)
        elif _b == "upper":
            _new_h = self.Align_to_block(_r_h * height, max_val=_h)
            _new_w = self.Align_to_block(_r_w * width, max_val=_w)
        else:  # _b -> "minimal"
            _new_h = self.Align_to_block(_r_h * height)
            _new_w = self.Align_to_block(_r_w * width)

        return (_new_w.astype(int), _new_h.astype(int))

    def Read_img(self, file_name: str | Path):
        _img = cv2.imread(str(file_name), cv2.IMREAD_UNCHANGED)
        _dim = _img.ndim

        if _dim == 4:
            return cv2.cvtColor(_img, cv2.COLOR_BGRA2RGB)
        if _dim == 3:
            return cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        return _img

    def __call__(
        self, sample: dict[str, list[str | Path]], value_limit: int = -1
    ):
        _key_list = self.key_list
        _keep_key_list = self.keep_key_list
        _inter = self.interpolation
        _sample = {}
        for _k, _file_list in sample.items():
            if _k not in _key_list:
                continue

            _meta = []
            _size = []
            for _name in _file_list:
                _img = self.Read_img(_name)

                if _img is not None:
                    if value_limit > 0:
                        _img[_img > value_limit] = 0
                    
                    _meta.append(_img)
                    _size.append(_img.shape[:2])

            if self.resize:
                if not _size:
                    raise ValueError
                
                _ori_h, _ori_w = np.array(_size).T
                _new_w, _new_h = self.Get_size(_ori_w, _ori_h)
                _data = [
                    cv2.resize(
                        _img, (_w, _h), interpolation=_inter
                    ) for _img, _w, _h in zip(_meta, _new_w, _new_h)
                ]
                _sample[f"{_k}_size"] = np.array(_size, dtype=np.uint16)
            else:
                _data = _meta

            _sample[_k] = np.array(_data, dtype=np.uint8)
            if _keep_key_list and _k in _keep_key_list:
                _sample[f"{_k}_ori"] = np.array(_meta)

        return _sample


class Norm():
    def __init__(
        self,
        norm_by_key: dict[str, Literal["ILSVRC", "min_max", "div_255"]],
    ):
        self.norm_by_key: dict[str, Callable[[np.ndarray], np.ndarray]] = dict((
            _k, getattr(self, f"Norm_{_v.lower()}")
        ) for _k, _v in norm_by_key.items())

    def Norm_ilsvrc(self, img: np.ndarray):  # ImageNet
        _img = img / 255.0
        return (_img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    def Norm_min_max(self, img: np.ndarray):
        _dim = img.ndim
        _axis = list(range(_dim-1))
        _min = img.min(axis=_axis)
        _max = img.max(axis=_axis)

        return (img - _min) / (_max - _min)
    
    def Norm_div_255(self, img: np.ndarray):
        return img / 255

    def __call__(self, sample: dict[str, np.ndarray]):
        for _k, _norm in self.norm_by_key.items():
            if _k in sample:
                sample[_k] = _norm(sample[_k])

        return sample


# class To_Tenser():
#     def __call__(self, *args, **kwds):
#         raise NotImplementedError

class To_Tenser():
    def __init__(self, color_key: list[str]):
        self.color_key = color_key

    def __call__(self, sample: dict[str, Any]):
        for _k, _v in sample.items():
            if isinstance(_v, np.ndarray):
                if _k in self.color_key:
                    if _v.ndim == 3:
                        _v = np.transpose(_v, (2, 0, 1))
                    if _v.ndim == 4:
                        _v = np.transpose(_v, (0, 3, 1, 2))

                sample[_k] = torch.from_numpy(
                    np.ascontiguousarray(_v).astype(np.float32)
                )

        return sample