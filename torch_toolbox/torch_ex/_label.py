from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional

from python_ex._base import Directory, File, Utils
from python_ex._vision import File_IO
from python_ex._numpy import Array_Process, Np_Dtype, ndarray, Image_Process


if __package__ == "":
    # if this file in local project
    from torch_ex._torch_base import Process_Name
else:
    # if this file in package folder
    from ._torch_base import Process_Name


# -- DEFINE CONSTNAT -- #
class Support_Label(Enum):
    CUSTOM = "custom_dataset"
    BDD_100k = "BDD_100k"
    COCO = "COCO"


class Data_Category(Enum):
    IMAGE = "image"
    CLASSIFICATION = "classification"
    SEMENTIC_SEG = "sementic"
    INSTANCES_SEG = "instances"
    BBOX = "bboxes"
    KEYPOINTS = "keypoints"


class File_Style(Enum):
    IMAGE_FILE = "image"
    ANNOTATION = "annotation"
    ZIP_FILE = "zip"


# -- DEFINE STRUCTURE -- #
class Label_Structure():
    class Label_Style(Enum):
        CLASSIFICATION = "classification"
        MASK = "mask"

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
class Data_Profile():
    _data_category: Data_Category
    _file_style: File_Style
    _optional_info: Optional[Any] = None

    _data_list: list = field(default_factory=list)


# -- Mation Function -- #
class Label():
    class File_IO():
        class Basement():
            _directory: Dict

            def __init__(self, file_info: Dict[Process_Name, List[Tuple[Data_Category, File_Style, Any]]], root: str) -> None:
                self._root_dir = root if Directory._Exist_check(root) else Directory._Relative_root()
                self._file_info = file_info

            # Un-Freeze function
            def _Get_file_profiles(self, mode: Process_Name) -> List[Data_Profile]:
                _file_profiles = []

                if mode in self._file_info.keys():
                    for _info in self._file_info[mode]:
                        _file_profiles.append(Data_Profile(_info[0], _info[1], _info[2]))

                return _file_profiles

        class BDD_100k(Basement):
            _directory: Dict[Data_Category, Dict[File_Style, str]] = {
                Data_Category.IMAGE: {
                    File_Style.IMAGE_FILE: Directory._Divider_check("bdd100k/images/{}/{}/"),
                },
                Data_Category.SEMENTIC_SEG: {
                    File_Style.IMAGE_FILE: Directory._Divider_check("bdd100k/labels/sem_seg/colormaps/{}/")
                }
            }

            def _Get_file_profiles(self, mode: Process_Name) -> List[Data_Profile]:
                assert mode in self._file_info.keys()
                _file_profiles = []

                for _info in self._file_info[mode]:
                    if _info[1] is File_Style.IMAGE_FILE:
                        _dir = f"{self._root_dir}{self._directory[_info[0]][_info[1]]}"
                        _dir = _dir.format(_info[2], mode.value) if _info[0] is Data_Category.IMAGE else _dir.format(mode.value)

                        _data_list = sorted(Directory._Search(_dir))
                    else:
                        _data_list = []

                    _file_profiles.append(Data_Profile(_info[0], _info[1], _info[2], _data_list))

                return _file_profiles

        class COCO(Basement):
            _directory: Dict[Data_Category, Dict[File_Style, str]] = {
                Data_Category.IMAGE: {
                    File_Style.IMAGE_FILE: Directory._Divider_check("coco/{}2017/"),
                }
            }

            def _Get_file_profiles(self, mode: Process_Name) -> List[Data_Profile]:
                _annotation_data: Dict[str, Dict] = {
                    "captions": {},
                    "instances": {},
                    "person_keypoints": {}
                }

                assert mode in self._file_info.keys()
                _file_profiles = []

                for _info in self._file_info[mode]:
                    assert _info[1] == File_Style.ANNOTATION, "COCO label style must be use annotation"  # In later fix this error message

                    _annotation_type: str = _info[2]
                    if _annotation_data[_annotation_type] == {}:
                        _meta_ann = File._Json(f"{self._root_dir}coco/annotations/", f"{_annotation_type}_{mode.value}2017.json")

                        if _annotation_type == "instances":
                            _holder = {}
                            for _image_info in _meta_ann["images"]:
                                _holder.update({_image_info["id"]: {"file_name": _image_info["file_name"], "bbox": [], "instances": [], "sementic": {}}})

                            for _label in _meta_ann["annotations"]:
                                if isinstance(_label["segmentation"], dict):
                                    ...  # in later fix it
                                else:
                                    _holder[_label["image_id"]]["bbox"].append(_label["bbox"] + [_label["category_id"], ])
                                    _holder[_label["image_id"]]["instances"].append(_label["segmentation"])

                                    if _label["category_id"] in _holder[_label["image_id"]]["sementic"].keys():
                                        _holder[_label["image_id"]]["sementic"][_label["category_id"]].append(_label["segmentation"])
                                    else:
                                        _holder[_label["image_id"]]["sementic"][_label["category_id"]] = [_label["segmentation"]]
                        else:
                            _holder = {}

                        _annotation_data[_annotation_type] = _holder

                    _data_category = _info[0]
                    if _data_category is Data_Category.IMAGE:
                        _file_style = File_Style.IMAGE_FILE
                        _dir = f"{self._root_dir}{self._directory[_data_category][_file_style]}"
                        _data_list = [f"{_dir.format(mode.value)}{_value['file_name']}" for _, _value in _annotation_data[_annotation_type].items()]
                    else:
                        _file_style = File_Style.ANNOTATION

                        if _data_category is Data_Category.BBOX:
                            _data_list = [_value["bbox"] for _, _value in _annotation_data[_annotation_type].items()]
                        elif _data_category is Data_Category.INSTANCES_SEG:
                            _data_list = [_value["instances"] for _, _value in _annotation_data[_annotation_type].items()]
                        elif _data_category is Data_Category.SEMENTIC_SEG:
                            _data_list = [_value["sementic"] for _, _value in _annotation_data[_annotation_type].items()]
                        else:
                            _data_list = []

                    _file_profiles.append(
                        Data_Profile(
                            _data_category,
                            _file_style,
                            _data_list=_data_list)
                    )

                return _file_profiles

    class Process():
        class Basement():
            # initialize
            def __init__(self, name: Support_Label, active_style: List[Label_Structure.Label_Style], meta_file: Optional[str] = None):
                # Parameter for Label information
                self._lable_name = name

                # Parameter for Label pre-process
                # Get label data
                if meta_file is not None and File._Exist_check(meta_file):
                    # from custom meta file
                    [_meta_dir, _meta_file] = File._Extrect_file_name(meta_file, just_file_name=False)
                else:
                    # from default meta file
                    _meta_dir = f"{Directory._Devide(__file__)[0]}data_file{Directory._Divider}"
                    _meta_file = f"{self._lable_name.value}.json"
                _meta_data: Dict[str, List[Dict]] = File._Json(file_dir=_meta_dir, file_name=_meta_file)

                self._activate_label: Dict[Label_Structure.Label_Style, Dict[int, List[Any]]] = {}
                # Make active_label from meta data
                for _style in active_style:
                    _label_list = [Label_Structure.Basement(**data) for data in _meta_data[_style.value]]

                    # Conver to Raw to Active
                    self._Make_active_label(_style, _label_list)

            # Freeze function
            def _Make_active_label(self, style: Label_Structure.Label_Style, raw_label: List[Label_Structure.Basement]):
                self._activate_label[style] = {}
                for _label in raw_label:
                    if _label.train_num in self._activate_label[style].keys():
                        self._activate_label[style][_label.train_num].append(_label.class_info)
                    else:
                        self._activate_label[style][_label.train_num] = [_label.class_info, ]

            def _Data_to_Label(self, data_category: Data_Category):
                if data_category is Data_Category.IMAGE:
                    _type = "image"
                elif data_category is Data_Category.CLASSIFICATION:
                    _type = "classification"
                elif data_category is Data_Category.BBOX:
                    _type = "bboxes"
                elif data_category in [Data_Category.INSTANCES_SEG, Data_Category.SEMENTIC_SEG]:
                    _type = "mask"
                else:
                    _type = "keypoints"

                return _type

            # Un-Freeze function
            def _Work(self, file_profiles: List[Data_Profile], index: int) -> Dict[str, ndarray]:
                raise NotImplementedError

        class BDD_100k(Basement):
            def _Work(self, file_profiles: List[Data_Profile], index: int) -> Dict[str, Optional[ndarray]]:
                _holder: Dict[str, Optional[ndarray]] = {"index": Array_Process._Convert_from(index, dtype=Np_Dtype.INT)}
                _count: Dict[str, int] = {
                    "image": 0,
                    "classification": 0,
                    "mask": 0,
                    "bboxes": 0,
                    "keypoints": 0
                }

                for _profile in file_profiles:
                    _pick_file = _profile._data_list[index]
                    if _profile._file_style == File_Style.IMAGE_FILE:
                        _data = File_IO._Image_read(_pick_file)
                        if _profile._data_category is Data_Category.SEMENTIC_SEG:
                            _data = Label_Img_Process._color_map_to_classification(_data, self._activate_label[Label_Structure.Label_Style.MASK])
                        elif _profile._data_category is Data_Category.INSTANCES_SEG:
                            ...
                    else:
                        _data = None

                    _type = self._Data_to_Label(_profile._data_category)
                    _key = f"{_type}{_count[_type]}" if _type in _holder.keys() else _type
                    _count[_type] += 1

                    _holder.update({_key: _data})

                return _holder

        class COCO(Basement):
            def _Work(self, file_profiles: List[Data_Profile], index: int) -> Dict[str, Optional[ndarray]]:
                _holder: Dict[str, Optional[ndarray]] = {"index": Array_Process._Convert_from(index, dtype=Np_Dtype.INT)}
                _count: Dict[str, int] = dict((_style.value, 0) for _style in Data_Category)

                for profile in file_profiles:
                    _pick_file = profile._data_list[index]
                    _data_style = profile._data_category.value

                    if profile._file_style == File_Style.IMAGE_FILE:
                        _data = File_IO._Image_read(_pick_file)

                    elif profile._file_style == File_Style.ANNOTATION:
                        _data = _pick_file
                    else:
                        _data = None

                    _key = f"{_data_style}{_count[_data_style]}" if _data_style in _holder.keys() else _data_style
                    if _data_style in _holder.keys():
                        _key = f"{_data_style}{_count[_data_style]}"
                        _count[_data_style] += 1
                    else:
                        _key = _data_style

                    _holder.update({_key: _data})

                return _holder

        class Imagenet_1k(Basement):
            ...

        class Imagenet_22k(Basement):
            ...

        class PASCAL(Basement):
            ...

        class CDnet(Basement):
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
        _color_list = Array_Process._Convert_from([activate_label[_id][0] for _id in _label_ids], dtype=Np_Dtype.UINT)
        return _color_list[classification]

    # (h, w, 3) -> (h, w, class count)
    @staticmethod
    def _color_map_to_class_map(color_map: ndarray, activate_label: Dict[int, List]) -> ndarray:
        _h, _w, _ = color_map.shape

        _label_ids = sorted(activate_label.keys())
        _class_map = Array_Process._Make_array([_h, _w, len(_label_ids)], 0, dtype=Np_Dtype.UINT)

        # color compare
        for _label_id in _label_ids[:-1]:  # last channel : ignore
            _color_list = [_label for _label in activate_label[_label_id]]
            _class_map[:, :, _label_id] = Array_Process._Convert_from(Image_Process._color_finder(color_map, _color_list), dtype=Np_Dtype.UINT)

        # make ignore
        _class_map[:, :, -1] = Array_Process._Convert_from(1 - Array_Process._Convert_from(_class_map.sum(axis=2), dtype=Np_Dtype.BOOL), dtype=Np_Dtype.UINT)
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
        _class_map = Array_Process._Make_array([_h, _w, num_id], 0, dtype=Np_Dtype.UINT)

        for _id in range(num_id):
            _class_map[:, :, _id] = classification == _id
        return _class_map
