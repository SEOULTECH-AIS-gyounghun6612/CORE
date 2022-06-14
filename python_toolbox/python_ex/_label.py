from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union

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
    COLORMAP = 0
    ANNOTATION = 1
    ZIP = 2


class file_process():
    """
    make data list
    """
    class file_basement():
        def __init__(self, data_root_directory: str, data_exts: List[List[str]], is_sequence: List[List[bool]]) -> None:
            self.data_root = _base.directory._slash_check(data_root_directory)
            self.data_exts = data_exts
            self.data_sequence = is_sequence
            self.data_list = []

        def get_len(self):
            return len(self.data_list)

        def pick_data(self, item_num) -> List[List]:
            return self.data_list[item_num]

        def get_file_list(self, data_paths: List[List[str]]):
            pass

    class file_from_directory(file_basement):
        """
        from image file
        """
        def __init__(self, data_root_directory: str, data_exts: List[List[str]], is_sequence: List[List[bool]]) -> None:
            """
            Args:
                data_root_directory  :
                data_ext        :
                file_structure  :
            Returns:
                None
            """
            super().__init__(data_root_directory, data_exts, is_sequence)

        def get_list_in(self, directory: str, ext: str, is_sequence: bool) -> List:
            if is_sequence:
                file_list = [_base.directory._inside_search(_search_dir, ext=ext) for _search_dir in _base.directory._inside_search(self.data_root + directory)]
            else:
                file_list = _base.directory._inside_search(self.data_root + directory, ext=ext)
            return file_list

        def get_file_list(self, data_paths: List[List[str]]):
            # make data list
            # data_paths -> [input -> [data path 0, data path 1, data path 2 ...], label -> [data path 0, data path 1, data path 2 ...]]
            # data_exts -> [input -> [data ext 0, data ext 1, data ext 2 ...], label -> [data ext 0, data ext 1, data ext 2 ...]]

            # outer => pick input or label parameters
            # inter => make data list, that exist in file path. used function zip(), this data re-group about each data 0 channel
            #          ([datas in path 0, datas in path 1, datas in path 2, ...]) --zip()--> ([data 0 in each path, data 1 in each path, data 2 in each path, ...])

            data_list = []
            for _paths, _exts, _squences in zip(data_paths, self.data_exts, self.data_sequence):
                temp_holder = []
                for _path, _ext, _sq in zip(_paths, _exts, _squences):
                    temp_holder.append(self.get_list_in(_path, _ext, _sq))
                data_list.append(zip(*temp_holder))

            self.data_list = list(zip(*data_list))

            return self.data_list

    class file_from_annotation_file(file_basement):
        """
        from image file
        """
        ...


class Label_style(Enum):
    CLASSIFICATION = "classification"
    SEG = "seg"
    DETECTION = "detection"


class Labels():
    @dataclass
    class classification_label():
        ID: int
        Train_ID: int
        Cateogry_ID: str
        Ignore_in_eval: bool
        Class_name: str
        name: str

    @dataclass
    class seg_label():
        ...


class label_process():
    class label_basement():
        Lable_name: str = "basement"
        Directory: Dict[Label_style, str] = {}
        Label_list: Dict[Label_style, List[Labels]] = {}
        Label_category: Dict[Label_style, List[str]] = {}

        def __init__(self, label_style: Label_style, file_style: File_style, data_size: List[int]) -> None:
            self.file_style = file_style

            self.label_style = label_style
            _meta_label: Dict[str, Union[int, str, bool]] = _base.file._json(
                file_dir=f"{_base.directory._devide(__file__)[0]}/data_file/",
                file_name=f"{self.Lable_name}.json")

            if self.label_style == Label_style.CLASSIFICATION:
                self.Label_list: List[Labels.classification_label] = {
                    self.label_style: [Labels.classification_label(**data) for data in _meta_label[self.label_style.value]]}
            # elif
            # ...

        def set_learning_mode(self, learning_mode: str):
            self.learning_mode = learning_mode

        def get_label_list(self) -> Dict:
            return self.Label_list[self.label_style]

        def get_data_directory(self) -> str:
            return self.Directory[self.label_style]

        def work(self, data_list, index):
            ...

    class Imagenet_1k(label_basement):
        Directory: Dict[Label_style, str] = {
            Label_style.CLASSIFICATION: "ILSVRC/2012/{}/"
        }
        Label_category: Dict[Label_style, List[str]] = {}

        def __init__(self, label_style: Label_style, file_style: File_style, data_size) -> None:
            self.Lable_name = "imagenet_1k"
            super().__init__(label_style, file_style, data_size)

        def work(self, data_list, index):
            ...

    class Imagenet_22k(label_basement):
        ...

    class BDD_100k(label_basement):
        Directory: Dict[Label_style, str] = {}
        Label_list: Dict[Label_style, List[label]] = {}
        Label_category: Dict[Label_style, List[str]] = {}

        label_: Dict[str, Union[Dict, str, label]] = {
            "directory": {
                "seg": {
                    "color_map": {
                        "input_dir": "images/10k/{}/",  # data_type
                        "label_dir": "labels/sem_seg/colormaps/{}/"}
                },
                "100k": {
                    "color_map": {
                        "input_dir": "images/100k/{}/"}
                }
            },
            "list": {   # "id", "train_id", "categoryId", "hasInstances", "ignoreInEval", "color", "name"
                "seg": [
                    label(0x00, 0xFF, 0x00, 0x00, 0x01, _numpy.np_base.get_array_from([0x00, 0x00, 0x00]), "unlabeled"),
                    label(0x01, 0xFF, 0x00, 0x00, 0x01, _numpy.np_base.get_array_from([0x00, 0x4A, 0x6F]), "dynamic"),
                    label(0x02, 0xFF, 0x00, 0x00, 0x01, _numpy.np_base.get_array_from([0x00, 0x00, 0x00]), "ego vehicle"),
                    label(0x03, 0xFF, 0x00, 0x00, 0x01, _numpy.np_base.get_array_from([0x51, 0x00, 0x51]), "ground"),
                    label(0x04, 0xFF, 0x00, 0x00, 0x01, _numpy.np_base.get_array_from([0x00, 0x00, 0x00]), "static"),
                    label(0x05, 0xFF, 0x01, 0x00, 0x01, _numpy.np_base.get_array_from([0xA0, 0xAA, 0xFA]), "parking"),
                    label(0x06, 0xFF, 0x01, 0x00, 0x01, _numpy.np_base.get_array_from([0x8C, 0x96, 0xE6]), "rail track"),
                    label(0x07, 0x00, 0x01, 0x00, 0x00, _numpy.np_base.get_array_from([0x80, 0x40, 0x80]), "road"),
                    label(0x08, 0x01, 0x01, 0x00, 0x00, _numpy.np_base.get_array_from([0xE8, 0x23, 0xF4]), "sidewalk"),
                    label(0x09, 0xFF, 0x02, 0x00, 0x01, _numpy.np_base.get_array_from([0x64, 0x64, 0x96]), "bridge"),
                    label(0x0A, 0x02, 0x02, 0x00, 0x00, _numpy.np_base.get_array_from([0x46, 0x46, 0x46]), "building"),
                    label(0x0B, 0x04, 0x02, 0x00, 0x00, _numpy.np_base.get_array_from([0x99, 0x99, 0xBE]), "fence"),
                    label(0x0C, 0xFF, 0x02, 0x00, 0x01, _numpy.np_base.get_array_from([0xB4, 0x64, 0xB4]), "garage"),
                    label(0x0D, 0xFF, 0x02, 0x00, 0x01, _numpy.np_base.get_array_from([0xB4, 0xA5, 0xB4]), "guard rail"),
                    label(0x0E, 0xFF, 0x02, 0x00, 0x01, _numpy.np_base.get_array_from([0x5A, 0x78, 0x96]), "tunnel"),
                    label(0x0F, 0x03, 0x02, 0x00, 0x00, _numpy.np_base.get_array_from([0x9C, 0x66, 0x66]), "wall"),
                    label(0x10, 0xFF, 0x03, 0x00, 0x01, _numpy.np_base.get_array_from([0x64, 0xAA, 0xFA]), "banner"),
                    label(0x11, 0xFF, 0x03, 0x00, 0x01, _numpy.np_base.get_array_from([0xFA, 0xDC, 0xDC]), "billboard"),
                    label(0x12, 0xFF, 0x03, 0x00, 0x01, _numpy.np_base.get_array_from([0x00, 0xA5, 0xFF]), "lane divider"),
                    label(0x13, 0xFF, 0x03, 0x00, 0x00, _numpy.np_base.get_array_from([0x3C, 0x14, 0xDC]), "parking sign"),
                    label(0x14, 0x05, 0x03, 0x00, 0x00, _numpy.np_base.get_array_from([0x99, 0x99, 0x99]), "pole"),
                    label(0x15, 0xFF, 0x03, 0x00, 0x01, _numpy.np_base.get_array_from([0x99, 0x99, 0x99]), "polegroup"),
                    label(0x16, 0xFF, 0x03, 0x00, 0x01, _numpy.np_base.get_array_from([0x64, 0xDC, 0xDC]), "street light"),
                    label(0x17, 0xFF, 0x03, 0x00, 0x01, _numpy.np_base.get_array_from([0x00, 0x46, 0xFF]), "traffic cone"),
                    label(0x18, 0xFF, 0x03, 0x00, 0x01, _numpy.np_base.get_array_from([0xDC, 0xDC, 0xDC]), "traffic device"),
                    label(0x19, 0x06, 0x03, 0x00, 0x00, _numpy.np_base.get_array_from([0x1E, 0xAA, 0xFA]), "traffic light"),
                    label(0x1A, 0x07, 0x03, 0x00, 0x00, _numpy.np_base.get_array_from([0x00, 0xDC, 0xDC]), "traffic sign"),
                    label(0x1B, 0xFF, 0x03, 0x00, 0x01, _numpy.np_base.get_array_from([0xFA, 0xAA, 0xFA]), "traffic sign frame"),
                    label(0x1C, 0x09, 0x04, 0x00, 0x00, _numpy.np_base.get_array_from([0x98, 0xFB, 0x98]), "terrain"),
                    label(0x1D, 0x08, 0x04, 0x00, 0x00, _numpy.np_base.get_array_from([0x23, 0x8E, 0x6B]), "vegetation"),
                    label(0x1E, 0x0A, 0x05, 0x00, 0x00, _numpy.np_base.get_array_from([0xB4, 0x82, 0x46]), "sky"),
                    label(0x1F, 0x0B, 0x06, 0x01, 0x00, _numpy.np_base.get_array_from([0x3C, 0x14, 0xDC]), "person"),
                    label(0x20, 0x0C, 0x06, 0x01, 0x00, _numpy.np_base.get_array_from([0x00, 0x00, 0xFF]), "rider"),
                    label(0x21, 0x12, 0x07, 0x01, 0x00, _numpy.np_base.get_array_from([0x20, 0x0B, 0x77]), "bicycle"),
                    label(0x22, 0x0F, 0x07, 0x01, 0x00, _numpy.np_base.get_array_from([0x64, 0x3C, 0x00]), "bus"),
                    label(0x23, 0x0D, 0x07, 0x01, 0x00, _numpy.np_base.get_array_from([0x8E, 0x00, 0x00]), "car"),
                    label(0x24, 0xFF, 0x07, 0x01, 0x01, _numpy.np_base.get_array_from([0x5A, 0x00, 0x00]), "caravan"),
                    label(0x25, 0x11, 0x07, 0x01, 0x00, _numpy.np_base.get_array_from([0xE6, 0x00, 0x00]), "motorcycle"),
                    label(0x26, 0xFF, 0x07, 0x01, 0x01, _numpy.np_base.get_array_from([0x6E, 0x00, 0x00]), "trailer"),
                    label(0x27, 0x10, 0x07, 0x01, 0x00, _numpy.np_base.get_array_from([0x64, 0x50, 0x00]), "train"),
                    label(0x28, 0x0E, 0x07, 0x01, 0x00, _numpy.np_base.get_array_from([0x46, 0x00, 0x00]), "truck")]
            },
            "category": {
                "seg": ["void", "flat", "construction", "object", "nature", "sky", "human", "vehicle"]
            }
        }

        def __init__(self, label_style, file_style) -> None:
            super().__init__(label_style, file_style)

        def get_data_directory(self, learning_style: str):
            pass

        def make_data(self, file_list: List[str], index, data_size):
            pass

    class CDnet():
        label_ = {
            "directory": {
                "seg": {
                }
            },
            "list": {   # "id", "train_id", "categoryId", "hasInstances", "ignoreInEval", "color", "name"
                "seg": [
                    label(0x00, 0x00, 0x01, 0x00, 0x00, [0x10, 0x10, 0x10], "Static"),
                    label(0x01, 0x00, 0x00, 0x00, 0x00, [0x32, 0x32, 0x32], "Hard shadow"),
                    label(0x02, 0xFF, 0x00, 0x00, 0x01, [0x55, 0x55, 0x55], "Outside region of interest"),
                    label(0x03, 0x01, 0x00, 0x00, 0x00, [0xAA, 0xAA, 0xAA], "Unknown motion"),
                    label(0x04, 0x02, 0x01, 0x00, 0x00, [0xFF, 0xFF, 0xFF], "Motion")]
            },
            "category": {
                "seg": ["void", "flat", "construction", "object", "nature", "sky", "human", "vehicle"]
            }
        }

        def __init__(self) -> None:
            pass

    class COCO():
        def __init__(self) -> None:
            pass
