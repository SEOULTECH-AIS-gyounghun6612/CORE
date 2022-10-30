from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from re import U
from tkinter.tix import AUTO
from typing import Dict, List, Tuple, Union, Any

if __package__ == "":
    # if this file in local project
    from _base import Directory, File, Utils
    import _cv2
    from _numpy import np_base, np_dtype
    import _error as _e

else:
    # if this file in package folder
    from ._base import Directory, File, Utils
    from . import _cv2
    from ._numpy import np_base, np_dtype
    from . import _error as _e


# Set constant
DEBUG = False
_error = _e.Custom_error(
    module_name="ais_custom_utils_v 2.x",
    file_name="_label.py")

label = namedtuple(
    "label",
    ["id", "train_id", "categoryId", "hasInstances", "ignoreInEval", "color", "name"])


# -- DEFINE CONSTNAT -- #
class Suported_Label(Enum):
    BDD_100K = "BDD-100k"


class Learning_Mode(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class Label_Style(Enum):
    CLASSIFICATION = "classification"
    SEM_SEG = "sem_seg"
    DETECTION = "detection"


class IO_Style(Enum):
    ANNOTATION = 0
    IMAGE_FILE = 1
    ZIP_FILE = 2


class Normalize_Style(Enum):
    AUTO = 0
    FIXED = 1


# -- CUSTOM DATA TYPE -- #
@dataclass
class Data_Profile():
    _Label_style: Label_Style
    _IO_style: IO_Style
    _Input: List = field(default_factory=list)
    _Label: List = field(default_factory=list)


# -- DEFINE STRUCTURE -- #
class Label_Structure():
    @dataclass
    class Basement():
        _Identity_num: int
        _Train_num: int
        _Cateogry_num: str
        _Ignore_in_eval: bool
        _Name: str

    @dataclass
    class Classification_Label(Basement):
        _Class_info: str

    @dataclass
    class Seg_Label(Basement):
        _Class_info: Tuple[int, int, int]  # BGR


# -- DEFINE CONFIG -- #
@dataclass
class Augmentation_Config(Utils.Config):
    _Resize: Union[List[int], List[float]] = field(default_factory=lambda: [1.0, 1.0])
    _Flip: int = 0  # horizontal, vertical
    _Rotate: float = 0.0  # 0 ~ 90 -> 0.0 ~ 1.0

    _Normalize = Normalize_Style.FIXED
    _Mean: List[int] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    _Std: List[int] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    def _convert_to_dict(self) -> Dict[str, Any]:
        return super()._convert_to_dict()

    def _restore_from_dict(self, data: Dict[str, Any]):
        return super()._restore_from_dict(data)


@dataclass
class Label_Config(Utils.Config):
    """
    """
    _Name: Suported_Label
    _Load_meta_file: str
    _Load_label_style: List[Label_Style]  # Label style list that get from meta data.

    _Data_root: str  # Where input and label data exist
    _Data_size: List[int]  # Get data size from source if that can change
    _Data_augmentation: Augmentation_Config

    def _convert_to_dict(self):
        #  "_Data_augmentation": self._Data_augmentation._convert_to_dict()}
        return {
            "_Name": self._Name.value,
            "_Load_meta_file": self._Load_meta_file,
            "_Load_label_style": [__style.value for __style in self._Load_label_style],
            "_Data_root": self._Data_root,
            "_Data_size": self._Data_size, }

    def _restore_from_dict(self, data: Dict[str, Any]):
        self._Name = Suported_Label(data["_Name"])
        self._Load_meta_file, = data["_Load_meta_file"]
        self._Load_label_style = [Label_Style(__style_name) for __style_name in data["_Load_label_style"]]
        self._Data_root = data["_Data_root"]
        self._Data_size = data["_Data_size"]
        # self._Data_augmentation = self._Data_augmentation._restore_from_dict(data["_Data_size"])

    def _root_directory_check(self) -> bool:
        if Directory._exist_check(self._Data_root):
            return True
        else:
            __default_root = Directory._relative_root() + f"Data{Directory._Divider}"
            return Directory._exist_check(__default_root)

    def _meta_file_exist_check(self) -> bool:
        if File._exist_check(self._Load_meta_file):
            return File._name_from_path(self._Load_meta_file)
        else:
            return None


# -- Mation Function -- #
class Label_Process():
    class Basement():
        # Parameter for Label information
        _Class_category: Dict[Label_Style, List[str]] = {}  # each class's category

        # Parameter for Label pre-process
        _Data_size: List[int]
        _Data_augmentation: Augmentation_Config

        # initialize
        def __init__(self, config: Label_Config):
            self._Lable_name = config._Name
            self._Root_dir = config._Data_root
            self._Data_size = config._Data_size

            # Get label data from meta file
            __meta = config._meta_file_exist_check()
            if __meta is not None:
                [__meta_dir, __meta_file] = __meta
            else:
                __meta_dir = f"{Directory._devide(__file__)[0]}/data_file/"
                __meta_file = f"{self._Lable_name.value}.json"
            __meta_data: Dict[str, Label_Structure] = File._json(file_dir=__meta_dir, file_name=__meta_file)

            # Make raw label data
            self._Raw_label: Dict[Label_Style, List[Union[Label_Structure.Classification_Label, Label_Structure.Seg_Label]]] = {}
            for _style in config._Load_label_style:
                if _style == Label_Style.CLASSIFICATION:
                    _label_list: List[Label_Structure.Classification_Label] = [Label_Structure.Classification_Label(**data) for data in __meta_data[_style.value]]

                elif _style == Label_Style.SEM_SEG:
                    _label_list: List[Label_Structure.Seg_Label] = [Label_Structure.Seg_Label(**data) for data in __meta_data[_style.value]]

                self._Raw_label[_style] = _label_list

            # Conver to Raw to Active
            self.make_label_info(_style)

        # Freeze function
        def make_label_info(self, style: Label_Style):
            self._Activated_label: Dict[int, List] = {}
            for _label in self._Raw_label[style]:
                _info = [_label._Class_info, _label._Name]
                if _label._Train_num in self._Activated_label.keys():
                    self._Activated_label[_label._Train_num].append(_info)
                else:
                    self._Activated_label[_label._Train_num] = [_info, ]

        def set_learning_mode(self, mode: Learning_Mode):
            self._Active_mode = mode

        # Un-Freeze function
        def get_data_profile(self, label_style: Label_Style, io_style: IO_Style) -> Data_Profile:
            return Data_Profile(_Label_style=label_style, _IO_style=io_style, _Input=[], _Label=[])  # Make data profile holder

        def work(self, data: Data_Profile, index: int):
            ...

        def label_data_convert(self, label, from_style, to_style):
            ...

    class Imagenet_1k(Basement):
        Directory: Dict[Label_Style, Dict[IO_Style, Union[List[str], str]]] = {
            Label_Style.CLASSIFICATION: {
                IO_Style.IMAGE_FILE: "ILSVRC/2012/{}/{}/",
                IO_Style.ZIP_FILE: "ILSVRC/2012/",
                IO_Style.ANNOTATION: "ILSVRC/2012/"}
        }
        Annotation: Dict[Label_Style, Dict[IO_Style, Union[List[str], str]]] = {
            Label_Style.CLASSIFICATION: {
                IO_Style.IMAGE_FILE: None,
                IO_Style.ZIP_FILE: "{}_map_for_zip.txt",
                IO_Style.ANNOTATION: "{}_map_for_annotation.json"}
        }

        def __init__(self, import_style: List[Label_Style], data_size: List[int], root: str = None) -> None:
            self._Lable_name = "imagenet_1k"
            super().__init__(import_style, data_size, root)

        def get_data_profile(self, IO_label_style: Label_Style, IO_file_style: IO_Style) -> Data_Profile:
            data_profile = super().get_data_profile(IO_label_style, IO_file_style)

            if data_profile._IO_style == IO_Style.IMAGE_FILE:
                _label_list = self._Raw_label[IO_label_style]
                _class_info = [[_label.Class_info, _label._Train_num] for _label in _label_list]

                for _name, train_id in _class_info:
                    _file_list = Directory._inside_search(
                        self._Root_dir + self.Directory[IO_label_style][IO_file_style].format(self.learning_mode, _name))
                    data_profile._Input += _file_list
                    data_profile._Label += [train_id for _ in range(len(_file_list))]

            elif data_profile._IO_style == IO_Style.ANNOTATION:
                _annotation = File._json(
                    self._Root_dir + self.Directory[IO_label_style][IO_file_style],
                    self.Annotation[IO_label_style][IO_file_style].format(self.learning_mode))
                data_profile._Input += _annotation["input"]
                data_profile._Label += _annotation["label"]

            return data_profile

        def work(self, data: Data_Profile, index):
            _label_list = self._Raw_label[data._Label_style]

            if data._IO_style == IO_Style.IMAGE_FILE:
                picked_image = _cv2.file.image_read(data._Input[index])
                picked_image = _cv2.cv_base.resize(picked_image, [224, 224])
                picked_label = np_base.get_array_from(len(_label_list), True, dtype=np_dtype.np_float32)

            return [picked_image, picked_label]

    class Imagenet_22k(Basement):
        ...

    class BDD_100k(Basement):
        _Class_category: Dict[Label_Style, List[str]] = {Label_Style.SEM_SEG: ["void", "flat", "construction", "object", "nature", "sky", "human", "vehicle"]}
        _Activated_label: Dict[int, List] = {}

        _Directory: Dict[Label_Style, Dict[IO_Style, Union[List[str], str]]] = {
            Label_Style.SEM_SEG: {
                IO_Style.IMAGE_FILE: "bdd-100k/{}/{}/",
                IO_Style.ZIP_FILE: "",
                IO_Style.ANNOTATION: ""}}

        def __init__(self, config: Label_Config):
            super().__init__(config)

        def get_data_profile(self, label_style: Label_Style, io_style: IO_Style) -> Data_Profile:
            data_profile = super().get_data_profile(label_style, io_style)

            if data_profile._Label_style == Label_Style.SEM_SEG:
                if data_profile._IO_style == IO_Style.IMAGE_FILE:
                    __input_dir = self._Directory[label_style][io_style].format("images/10k", self._Active_mode)
                    data_profile._Input += sorted(Directory._inside_search(self._Root_dir + __input_dir))

                    __label_dir = self._Directory[label_style][io_style].format("labels/sem_seg/colormaps", self._Active_mode)
                    data_profile._Label += sorted(Directory._inside_search(self._Root_dir + __label_dir))

                elif data_profile._IO_style == IO_Style.ANNOTATION:
                    ...

            return data_profile

        def work(self, data: Data_Profile, index: int):
            if data._Label_style == Label_Style.SEM_SEG:
                # Get data from each data source
                if data._IO_style == IO_Style.IMAGE_FILE:
                    picked_input = _cv2.file.image_read(data._Input[index])
                    picked_label = _cv2.file.image_read(data._Label[index])

                elif data._IO_style == IO_Style.ANNOTATION:
                    ...

                # Data pre-process
                # In later fix it -> using config augmentation
                picked_input = _cv2.cv_base.resize(picked_input, self._Data_size)
                picked_input = _cv2.cv_base.img_cvt(picked_input, _cv2.CVT_option.BGR2RGB)
                picked_input = np_base.type_converter(picked_input, np_dtype.np_float32)

                picked_label = _cv2.augmentation._colormap_to_classification(picked_label, self._Activated_label, self._Data_size)
                picked_label = np_base.type_converter(picked_label, np_dtype.np_float32)

                return {"input": picked_input, "label": picked_label, "info": index}

        def unwork(self, image):
            ...

        def label_data_convert(self, label, from_style, to_style):
            return _cv2.augmentation._classification_to_colormap(label, self._Activated_label, self._Data_size)

    class PASCAL():
        ...

    class CDnet():
        Label_category: Dict[Label_Style, List[str]] = {Label_Style.SEM_SEG: ["void", "flat", "construction", "object", "nature", "sky", "human", "vehicle"]}

        Directory: Dict[Label_Style, Dict[IO_Style, Union[List[str], str]]] = {
            Label_Style.SEM_SEG: {
                IO_Style.IMAGE_FILE: "",
                IO_Style.ZIP_FILE: "",
                IO_Style.ANNOTATION: ""}
        }
        Annotation: Dict[Label_Style, Dict[IO_Style, Union[List[str], str]]] = {
            Label_Style.SEM_SEG: {
                IO_Style.ZIP_FILE: "",
                IO_Style.ANNOTATION: ""}
        }

        def __init__(self, import_style: List[Label_Style], data_size: List[int]) -> None:
            self.Lable_name = "CDnet"
            super().__init__(import_style, data_size)

        def get_data_profile(self, holder: Data_Profile) -> Data_Profile:
            ...

        def work(self, data, index):
            ...

    class COCO():
        def __init__(self) -> None:
            pass

    @staticmethod
    def _build(config: Label_Config):
        if config._Name == Suported_Label.BDD_100K:
            return Label_Process.BDD_100k(config)


class Data_Augmentation():
    ...
