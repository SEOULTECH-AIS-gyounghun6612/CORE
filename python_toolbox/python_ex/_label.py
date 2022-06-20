from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Union

if __package__ == "":
    # if this file in local project
    from _base import directory, file
    import _cv2
    import _numpy
    import _error as _e

else:
    # if this file in package folder
    from ._base import directory, file
    from . import _cv2
    from . import _numpy
    from . import _error as _e


# Set constant
DEBUG = False
_error = _e.Custom_error(
    module_name="ais_custom_utils_v 2.x",
    file_name="_label.py")

label = namedtuple(
    "label",
    ["id", "train_id", "categoryId", "hasInstances", "ignoreInEval", "color", "name"])


class File_style(Enum):
    ANNOTATION = 0
    IMAGE_FILE = 1
    ZIP_FILE = 2


class Label_style(Enum):
    CLASSIFICATION = "classification"
    SEG = "seg"
    DETECTION = "detection"


@dataclass
class Data():
    Label_style: Label_style
    File_style: File_style
    Input: List = field(default_factory=list)
    Label: List = field(default_factory=list)


class Labels():
    @dataclass
    class basement():
        ID: int
        Train_ID: int
        Cateogry_ID: str
        Ignore_in_eval: bool
        name: str

    @dataclass
    class classification_label(basement):
        Class_name: str

    @dataclass
    class seg_label(basement):
        Class_color: Tuple[int, int, int]  # BGR


@dataclass
class Augmentation():
    ...


class Label_process():
    class label_basement():
        Lable_name: str = "basement"

        Label_dict: Dict[Label_style, List[Union[Labels.classification_label, Labels.seg_label]]] = {}
        Label_category: Dict[Label_style, List[str]] = {}

        Root_dir: str = directory._make("data", directory._relative_root())
        Data_directory: Dict[Label_style, Dict[File_style, Union[List[str], str]]] = {}
        Data_annotation: Dict[Label_style, Union[str, List[str]]] = {}

        Augmentation_list: Augmentation = {}

        def __init__(self, import_style: List[Label_style], augmentation: Augmentation, root: str = None) -> None:
            self.Root_dir = directory._slash_check(root) if root is not None else self.Root_dir  # set data root dir

            _meta_label: Dict[str, Union[int, str, bool]] = file._json(
                file_dir=f"{directory._devide(__file__)[0]}/data_file/",
                file_name=f"{self.Lable_name}.json")  # get label information from file

            for _style in import_style:  # data set in label dictionary from meta label
                if _style == Label_style.CLASSIFICATION:
                    _label_list: List[Labels.classification_label] = [Labels.classification_label(**data) for data in _meta_label[_style.value]]

                # elif
                # ...

                self.Label_dict[_style] = _label_list

        def set_learning_mode(self, learning_mode: str):
            self.learning_mode = learning_mode

        def get_data_profile(self, IO_label_style: Label_style, IO_file_style: File_style) -> Data:
            data_profile = Data(Label_style=IO_label_style, File_style=IO_file_style, Input=[], Label=[])

            return data_profile

        def work(self, data: Data, index: int):
            ...

    class Imagenet_1k(label_basement):
        Directory: Dict[Label_style, Dict[File_style, Union[List[str], str]]] = {
            Label_style.CLASSIFICATION: {
                File_style.IMAGE_FILE: "ILSVRC/2012/{}/{}/",
                File_style.ZIP_FILE: "ILSVRC/2012/",
                File_style.ANNOTATION: "ILSVRC/2012/"}
        }
        Annotation: Dict[Label_style, Dict[File_style, Union[List[str], str]]] = {
            Label_style.CLASSIFICATION: {
                File_style.IMAGE_FILE: None,
                File_style.ZIP_FILE: "{}_map_for_zip.txt",
                File_style.ANNOTATION: "{}_map_for_annotation.json"}
        }

        def __init__(self, import_style: List[Label_style], augmentation: Augmentation, root: str = None) -> None:
            self.Lable_name = "imagenet_1k"
            super().__init__(import_style, augmentation, root)

        def get_data_profile(self, IO_label_style: Label_style, IO_file_style: File_style) -> Data:
            data_profile = super().get_data_profile(IO_label_style, IO_file_style)

            IO_file_dir = self.Directory[IO_label_style][IO_file_style]
            IO_anno = self.Annotation[IO_label_style][IO_file_style]

            # make data list
            if data_profile.File_style == File_style.IMAGE_FILE:
                _label_list = self.Label_dict[IO_label_style]
                _class_info = [[_label.Class_name, _label.Train_ID] for _label in _label_list]

                for _name, train_id in _class_info:
                    _file_list = directory._inside_search(self.Root_dir + IO_file_dir.format(self.learning_mode, _name))
                    data_profile.Input += _file_list
                    data_profile.Label += [train_id for _ in range(len(_file_list))]

            elif data_profile.File_style == File_style.ANNOTATION:
                _annotation = file._json(self.Root_dir + IO_file_dir, IO_anno.format(self.learning_mode))
                data_profile.Input += _annotation["input"]
                data_profile.Label += _annotation["label"]

            return data_profile

        def work(self, data: Data, index):
            _label_list = self.Label_dict[data.Label_style]

            if data.File_style == File_style.IMAGE_FILE:
                picked_image = _cv2.file.image_read(data.Input[index])
                picked_image = _cv2.cv_base.resize(picked_image, [224, 224])
                picked_label = _numpy.np_base.get_array_from(len(_label_list), True, dtype=_numpy.np_base.np_dtype.np_float32)

                return [picked_image, picked_label]
                ...
            ...

    class Imagenet_22k(label_basement):
        ...

    class BDD_100k(label_basement):
        Label_category: Dict[Label_style, List[str]] = {Label_style.SEG: ["void", "flat", "construction", "object", "nature", "sky", "human", "vehicle"]}

        Directory: Dict[Label_style, Dict[File_style, Union[List[str], str]]] = {
            Label_style.SEG: {
                File_style.IMAGE_FILE: "",
                File_style.ZIP_FILE: "",
                File_style.ANNOTATION: ""}
        }
        Annotation: Dict[Label_style, Dict[File_style, Union[List[str], str]]] = {
            Label_style.SEG: {
                File_style.ZIP_FILE: "",
                File_style.ANNOTATION: ""}
        }

        def __init__(self, import_style: List[Label_style], data_size: List[int]) -> None:
            self.Lable_name = "BDD-100k"
            super().__init__(import_style, data_size)

        def get_data_profile(self, holder: Data) -> Data:
            ...

        def work(self, data, index):
            ...

    class CDnet():
        Label_category: Dict[Label_style, List[str]] = {Label_style.SEG: ["void", "flat", "construction", "object", "nature", "sky", "human", "vehicle"]}

        Directory: Dict[Label_style, Dict[File_style, Union[List[str], str]]] = {
            Label_style.SEG: {
                File_style.IMAGE_FILE: "",
                File_style.ZIP_FILE: "",
                File_style.ANNOTATION: ""}
        }
        Annotation: Dict[Label_style, Dict[File_style, Union[List[str], str]]] = {
            Label_style.SEG: {
                File_style.ZIP_FILE: "",
                File_style.ANNOTATION: ""}
        }

        def __init__(self, import_style: List[Label_style], data_size: List[int]) -> None:
            self.Lable_name = "CDnet"
            super().__init__(import_style, data_size)

        def get_data_profile(self, holder: Data) -> Data:
            ...

        def work(self, data, index):
            ...

    class COCO():
        def __init__(self) -> None:
            pass
