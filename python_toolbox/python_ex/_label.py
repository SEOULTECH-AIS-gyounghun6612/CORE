from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Union

if __package__ == "":
    # if this file in local project
    import _base
    import _cv2
    import _numpy
    import _error as _e

else:
    # if this file in package folder
    from . import _base
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


class label_process():
    class label_basement():
        Lable_name: str = "basement"
        Label_dict: Dict[Label_style, List[Union[Labels.classification_label, Labels.seg_label]]] = {}
        Label_category: Dict[Label_style, List[str]] = {}

        Directory: Dict[Label_style, Dict[File_style, Union[List[str], str]]] = {}
        Annotation: Dict[Label_style, Union[str, List[str]]] = {}

        def __init__(self, import_style: List[Label_style], data_size: List[int]) -> None:
            _meta_label: Dict[str, Union[int, str, bool]] = _base.file._json(
                file_dir=f"{_base.directory._devide(__file__)[0]}/data_file/",
                file_name=f"{self.Lable_name}.json")

            for _style in import_style:
                if _style == Label_style.CLASSIFICATION:
                    _label_list: List[Labels.classification_label] = [Labels.classification_label(**data) for data in _meta_label[_style.value]]

                # elif
                # ...

                self.Label_dict[_style] = _label_list

        def set_learning_mode(self, learning_mode: str):
            self.learning_mode = learning_mode

        def make_file_process_profile(self, label_style: Label_style, file_style: File_style) -> Dict:
            ...

        def work(self, data_list, index):
            ...

    class Imagenet_1k(label_basement):
        Directory: Dict[Label_style, Dict[File_style, Union[List[str], str]]] = {
            Label_style.CLASSIFICATION: {
                File_style.IMAGE_FILE: "ILSVRC/{}/{}",
                File_style.ZIP_FILE: "ILSVRC/",
                File_style.ANNOTATION: "ILSVRC/"}
        }
        Annotation: Dict[Label_style, Dict[File_style, Union[List[str], str]]] = {
            Label_style.CLASSIFICATION: {
                File_style.ZIP_FILE: "{}_map_for_zip.txt",
                File_style.ANNOTATION: "{}_map_for_annotation.json"}
        }

        def __init__(self, import_style: List[Label_style], data_size: List[int]) -> None:
            self.Lable_name = "imagenet_1k"
            super().__init__(import_style, data_size)

        def make_file_process_profile(self, label_style: Label_style, file_style: File_style) -> Dict:
            if file_style == File_style.IMAGE_FILE:
                return {"directory": self.Directory[label_style][file_style], "label_list": self.Label_dict[label_style], "annotation_file_name": None}
            elif file_style == File_style.ANNOTATION:
                return {"directory": self.Directory[label_style][file_style], "label_list": None, "annotation_file_name": self.Annotation[label_style][file_style]}

        def work(self, data_list, index):
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

        def make_file_process_profile(self, label_style: Label_style, file_style: File_style) -> Dict:
            ...

        def work(self, data_list, index):
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

        def make_file_process_profile(self, label_style: Label_style, file_style: File_style) -> Dict:
            ...

        def work(self, data_list, index):
            ...

    class COCO():
        def __init__(self) -> None:
            pass


class file_process():
    class basement():
        def __init__(self, root_directory: str, label_style: Label_style, file_style: File_style) -> None:
            self.root = _base.directory._slash_check(root_directory)
            self.label_style = label_style
            self.file_style = file_style

        def get_profile_option(self):
            return {"label_style": self.label_style, "file_style": self.file_style}

        def work(self, directory: str, **info):
            _dir = _base.directory._slash_check(self.root + directory)
            data_list = {"input": [], "label": [], "file_style": self.file_style}

            return _dir, data_list

    class classification(basement):
        def __init__(self, root_directory: str, file_style: File_style = File_style.IMAGE_FILE) -> None:
            super().__init__(root_directory, file_style)

        def work(self, directory: str, label_list: List[Labels.classification_label], annotation_file_name: str):
            _dir, data_list = super().work(directory)

            if self.file_style == File_style.IMAGE_FILE:
                _class_info = [[_label.Class_name, _label.Train_ID] for _label in label_list]

                for _name, train_id in _class_info:
                    _file_list = _base.directory._inside_search(_dir.format(_name))
                    data_list["input"] += _file_list
                    data_list["label"] += [train_id for _ in range(len(_file_list))]

            elif self.file_style == File_style.ANNOTATION:
                _annotation = _base.file._json(_dir, annotation_file_name)
                data_list["input"] = _annotation["input"]
                data_list["label"] = _annotation["label"]

            return _dir, data_list

    class segmentation(basement):
        ...