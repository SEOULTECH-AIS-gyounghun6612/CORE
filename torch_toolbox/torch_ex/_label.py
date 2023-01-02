from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional

from python_ex._base import Directory, File, Utils
from python_ex._vision import File_IO
from python_ex._numpy import Array_Process, Np_Dtype, ndarray, Image_Process


if __package__ == "":
    # if this file in local project
    from torch_ex._torch_base import Learning_Mode
else:
    # if this file in package folder
    from ._torch_base import Learning_Mode


# -- DEFINE CONSTNAT -- #
class Support_Label(Enum):
    CUSTOM = "CUSTOM"
    BDD_100k = "BDD_100k"


class Data_Style(Enum):
    IMAGE = "image"
    CLASSIFICATION = "classification"
    SEMENTIC_SEG = "mask"
    BBOX = "bboxes"


class File_Style(Enum):
    IMAGE_FILE = 0
    ANNOTATION = 1
    ZIP_FILE = 2


# -- DEFINE STRUCTURE -- #
class Label_Structure():
    @dataclass
    class Basement(Utils.Config):
        identity_num: int
        train_num: int
        cateogry: str
        ignore_in_eval: bool
        name: str
        class_info: Any

    @dataclass
    class Classification_Label(Basement):
        class_info: str

    @dataclass
    class Seg_Label(Basement):
        class_info: Tuple[int, int, int]  # BGR


@dataclass
class File_Profile():
    _data_style: Data_Style
    _data_file: File_Style
    _file_directory: Optional[str] = None

    _file_list: list = field(default_factory=list)


# -- Mation Function -- #
class Label():
    class File_IO():
        class Basement():
            _directory: Dict

            def __init__(self, root: str, file_info: Dict[Learning_Mode, List[Tuple[Data_Style, File_Style, Optional[str]]]]) -> None:
                self._root_dir = root if Directory._exist_check(root) else Directory._relative_root()
                self._file_info = file_info

                self._learning_mode = list(file_info.keys())[0]

            # Freeze function
            def _Set_learning_mode(self, mode: Learning_Mode):
                self._learning_mode = mode

            # Un-Freeze function
            def _Get_file_profiles(self) -> List[File_Profile]:
                file_profiles = []
                for _info in self._file_info[self._learning_mode]:
                    file_profiles.append(File_Profile(_info[0], _info[1], _info[2]))
                return file_profiles

        class BDD_100k(Basement):
            _directory: Dict[Data_Style, Dict[File_Style, str]] = {
                Data_Style.IMAGE: {
                    File_Style.IMAGE_FILE: Directory._divider_check("bdd100k/images/{}/{}/"),
                },
                Data_Style.SEMENTIC_SEG: {
                    File_Style.IMAGE_FILE: Directory._divider_check("bdd100k/labels/sem_seg/colormaps/{}/")
                }
            }

            def _Get_file_profiles(self) -> List[File_Profile]:
                _file_profiles = super()._Get_file_profiles()
                _mode = self._learning_mode.value

                for _profile in _file_profiles:
                    if _profile._data_file == File_Style.IMAGE_FILE:
                        _dir = f"{self._root_dir}{self._directory[_profile._data_style][_profile._data_file]}"
                        _dir = _dir.format(_profile._file_directory, _mode) if _profile._data_style == Data_Style.IMAGE else _dir.format(_mode)
                        _profile._file_list = sorted(Directory._inside_search(_dir))

                    elif _profile._data_file == File_Style.ANNOTATION:
                        ...

                return _file_profiles

    class Process():
        class Basement():
            # initialize
            def __init__(self, name: Support_Label, active_style: List[Data_Style], meta_file: Optional[str] = None):
                # Parameter for Label information
                self._lable_name = name

                # Parameter for Label pre-process
                self._activate_label: Dict[Data_Style, Dict[int, List[Any]]] = {}

                # Get label data
                if meta_file is not None and File._exist_check(meta_file):
                    # from custom meta file
                    [_meta_dir, _meta_file] = File._file_name_from_path(meta_file, just_file_name=False)
                else:
                    # from default meta file
                    _meta_dir = f"{Directory._devide(__file__)[0]}data_file{Directory._Divider}"
                    _meta_file = f"{self._lable_name.value}.json"
                _meta_data: Dict[str, List[Dict]] = File._json(file_dir=_meta_dir, file_name=_meta_file)

                # Make active_label from meta data
                for _style in active_style:
                    if _style == Data_Style.CLASSIFICATION:
                        _label_list = [Label_Structure.Basement(**data) for data in _meta_data[_style.value]]

                    else:  # _style == Label_Style.SEMENTIC_SEG:
                        _label_list = [Label_Structure.Basement(**data) for data in _meta_data[_style.value]]

                    # Conver to Raw to Active
                    self._Make_active_label(_style, _label_list)

            # Freeze function
            def _Make_active_label(self, style: Data_Style, raw_label: List[Label_Structure.Basement]):
                self._activate_label[style] = {}
                for _label in raw_label:
                    if _label.train_num in self._activate_label[style].keys():
                        self._activate_label[style][_label.train_num].append(_label.class_info)
                    else:
                        self._activate_label[style][_label.train_num] = [_label.class_info, ]

            # Un-Freeze function

            def _work(self, file_profiles: List[File_Profile], index: int) -> Dict[str, ndarray]:
                raise NotImplementedError

        class BDD_100k(Basement):
            def _work(self, file_profiles: List[File_Profile], index: int) -> Dict[str, ndarray]:
                _holder = {"index": Array_Process._converter(index, dtype=Np_Dtype.INT)}

                _image_count = 0
                _mask_count = 0
                # _bbox_count = 0

                for profile in file_profiles:
                    _pick_file = profile._file_list[index]
                    _data_style = profile._data_style.value

                    if profile._data_file == File_Style.IMAGE_FILE:
                        _data = File_IO._image_read(_pick_file)

                        if profile._data_style == Data_Style.IMAGE:
                            if _data_style in _holder.keys():
                                _holder.update({f"{_data_style}{_image_count}": _data})
                                _image_count += 1
                            else:
                                _holder.update({_data_style: _data})

                        elif profile._data_style == Data_Style.SEMENTIC_SEG:
                            _data = Label_Img_Process._color_map_to_classification(_data, self._activate_label[profile._data_style])

                            if _data_style in _holder.keys():
                                _holder.update({f"{_data_style}{_image_count}": _data})
                                _mask_count += 1
                            else:
                                _holder.update({_data_style: _data})

                return _holder

        class Imagenet_1k(Basement):
            ...

        class Imagenet_22k(Basement):
            ...

        class PASCAL(Basement):
            ...

        class CDnet(Basement):
            ...

        class COCO(Basement):
            ...


class Label_Img_Process():
    # (h, w, class count) -> (h, w)
    @staticmethod
    def _class_map_to_classification(class_map: ndarray) -> ndarray:
        return class_map.argmax(axis=2)

    # (h, w) -> (h, w, 3)
    @staticmethod
    def _classification_to_color_map(classification: ndarray, activate_label: Dict[int, List]) -> ndarray:
        _label_ids = sorted(activate_label.keys())
        _color_list = Array_Process._converter([activate_label[_id][0] for _id in _label_ids], dtype=Np_Dtype.UINT)
        return _color_list[classification]

    # (h, w, 3) -> (h, w, class count)
    @staticmethod
    def _color_map_to_class_map(color_map: ndarray, activate_label: Dict[int, List]) -> ndarray:
        _h, _w, _ = color_map.shape

        _label_ids = sorted(activate_label.keys())
        _class_map = Array_Process._converter([_h, _w, len(_label_ids)], True, dtype=Np_Dtype.UINT)

        # color compare
        for _label_id in _label_ids[:-1]:  # last channel : ignore
            _color_list = [_label for _label in activate_label[_label_id]]
            _class_map[:, :, _label_id] = Array_Process._converter(Image_Process._color_finder(color_map, _color_list), dtype=Np_Dtype.UINT)

        # make ignore
        _class_map[:, :, -1] = Array_Process._converter(1 - Array_Process._converter(_class_map.sum(axis=2), dtype=Np_Dtype.BOOL), dtype=Np_Dtype.UINT)
        return _class_map

    # (h, w, 3) -> (h, w)
    @staticmethod
    def _color_map_to_classification(color_map: ndarray, activate_label: Dict[int, List]) -> ndarray:
        _class_map = Label_Img_Process._color_map_to_class_map(color_map, activate_label)
        return Label_Img_Process._class_map_to_classification(_class_map)

    # (h, w, class count) -> (h, w, 3)
    @staticmethod
    def _class_map_to_color_map(class_map: ndarray, activate_label: Dict[int, List]) -> ndarray:
        _classification = Label_Img_Process._class_map_to_classification(class_map)
        return Label_Img_Process._classification_to_color_map(_classification, activate_label)

    # (h, w) -> (h, w, class count)
    @staticmethod
    def _classification_to_class_map(classification: ndarray, num_id: int) -> ndarray:
        _h, _w = classification.shape
        _class_map = Array_Process._converter([_h, _w, num_id], True, dtype=Np_Dtype.UINT)

        for _id in range(num_id):
            _class_map[:, :, _id] = classification == _id
        return _class_map
