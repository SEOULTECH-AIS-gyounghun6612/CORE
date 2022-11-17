from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Union, Any, Optional

if __package__ == "":
    # if this file in local project
    from _base import Directory, File, Utils
    from _vision import File_IO, Label_Img_Process
    from _numpy import Array_Process, Np_Dtype, ndarray

else:
    # if this file in package folder
    from ._base import Directory, File, Utils
    from ._vision import File_IO, Label_Img_Process
    from _numpy import Array_Process, Np_Dtype, ndarray


# -- DEFINE CONSTNAT -- #
class Support_Label(Enum):
    BDD_100k = "BDD-100k"


class Learning_Mode(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class Input_Style(Enum):
    IMAGE = "classification"
    NUMPY = "sem_seg"


class Label_Style(Enum):
    CLASSIFICATION = "classification"
    SEMENTIC_SEG = "sem_seg"
    DETECTION = "detection"


class IO_Style(Enum):
    ANNOTATION = 0
    IMAGE_FILE = 1
    ZIP_FILE = 2


class Normalize_Style(Enum):
    AUTO = 0
    FIXED = 1


# -- DEFINE STRUCTURE -- #
@dataclass
class Work_Profile():
    _Label_style: Label_Style
    _Label_IO: IO_Style

    _Input_style: Input_Style = Input_Style.IMAGE
    _Input_IO: IO_Style = IO_Style.IMAGE_FILE

    _Data_list: List = field(default_factory=list)


class Label_Structure():
    @dataclass
    class Basement(Utils.Config):
        _Identity_num: int
        _Train_num: int
        _Cateogry: str
        _Ignore_in_eval: bool
        _Name: str
        _Class_info: Any

    @dataclass
    class Classification_Label(Basement):
        _Class_info: str

    @dataclass
    class Seg_Label(Basement):
        _Class_info: Tuple[int, int, int]  # BGR


# -- DEFINE CONFIG -- #
@dataclass
class Label_Process_Config(Utils.Config):
    """
    """
    _Name: Support_Label = Support_Label.BDD_100k
    _Data_root: Optional[str] = None  # Where input and label data exist
    _Meta_file: Optional[str] = None
    _Active_style: List[Label_Style] = field(default_factory=lambda: [Label_Style.SEMENTIC_SEG])  # Label style list that get from meta data.

    def _get_parameter(self):
        return {
            "name": self._Name,
            "data_root": self._Data_root if self._Data_root is None else f"{Directory._relative_root()}{Directory._Divider}data{Directory._Divider}",
            "active_style": [_style for _style in self._Active_style],
            "meta_file": self._Meta_file}

    def _convert_to_dict(self) -> Dict[str, Union[Dict, str, int, float, bool, None]]:
        return {
            "_Name": self._Name.value,
            "_File_root": self._Data_root,
            "_Active_style": [_style.value for _style in self._Active_style],
            "_Meta_file": self._Meta_file}


# -- Mation Function -- #
class Label_Process():
    class Basement():
        # initialize
        def __init__(self, name: Support_Label, data_root: str, active_style: List[Label_Style], meta_file: Optional[str] = None):
            # Parameter for Label information
            self._Lable_name = name

            # Parameter for Label pre-process
            self._Root_dir = data_root if Directory._exist_check(data_root) else Directory._relative_root()
            self._Activate_label: Dict[Label_Style, Dict[int, List[Any]]] = {}

            # Get label data from meta file
            if meta_file is not None and File._exist_check(meta_file):
                [_meta_dir, _meta_file] = File._file_name_from_path(meta_file, just_file_name=False)
            else:
                _meta_dir = f"{Directory._devide(__file__)[0]}data_file{Directory._Divider}"
                _meta_file = f"{self._Lable_name.value}.json"
            _meta_data: Dict[str, List[Dict]] = File._json(file_dir=_meta_dir, file_name=_meta_file)

            # Make active_label from meta data
            for _style in active_style:
                if _style == Label_Style.CLASSIFICATION:
                    _label_list = [Label_Structure.Basement(**data) for data in _meta_data[_style.value]]

                else:  # _style == Label_Style.SEMENTIC_SEG:
                    _label_list = [Label_Structure.Basement(**data) for data in _meta_data[_style.value]]

                # Conver to Raw to Active
                self._make_active_label(_style, _label_list)

        # Freeze function
        def _make_active_label(self, style: Label_Style, raw_label: List[Label_Structure.Basement]):
            self._Activate_label[style] = {}
            for _label in raw_label:
                if _label._Train_num in self._Activate_label[style].keys():
                    self._Activate_label[style][_label._Train_num].append(_label._Class_info)
                else:
                    self._Activate_label[style][_label._Train_num] = [_label._Class_info, ]

        def _set_learning_mode(self, mode: Learning_Mode):
            self._Active_mode = mode

        # Un-Freeze function
        def _get_work_profile(
                self, label_style: Label_Style, label_io: IO_Style, input_style: Input_Style = Input_Style.IMAGE, input_io: IO_Style = IO_Style.IMAGE_FILE):
            _parameter = {
                "_Label_style": label_style,
                "_Label_IO": label_io,
                "_Input_style": input_style,
                "_Input_IO": input_io,

                "_Data_list": []}

            return Work_Profile(**_parameter)  # Make data profile holder

        def _work(self, data: Work_Profile, index: int) -> Dict[str, ndarray]:
            raise NotImplementedError

    class BDD_100k(Basement):
        _Directory: Dict[Union[Label_Style, Input_Style], Dict[IO_Style, str]] = {
            Input_Style.IMAGE: {
                IO_Style.IMAGE_FILE: "bdd100k/images/{}/{}/",
                IO_Style.ZIP_FILE: "",
                IO_Style.ANNOTATION: ""
            },
            Label_Style.SEMENTIC_SEG: {
                IO_Style.IMAGE_FILE: "bdd100k/labels/sem_seg/colormaps/{}/",
                IO_Style.ZIP_FILE: "",
                IO_Style.ANNOTATION: ""}}

        def _get_work_profile(
                self, label_style: Label_Style, label_io: IO_Style, input_style: Input_Style = Input_Style.IMAGE, input_io: IO_Style = IO_Style.IMAGE_FILE):
            _selected_data = super()._get_work_profile(label_style, label_io, input_style, input_io)

            if label_style == Label_Style.SEMENTIC_SEG:
                if label_io == IO_Style.IMAGE_FILE:
                    # input
                    _input_dir = self._Directory[input_style][input_io].format("10k", self._Active_mode.value)
                    _input_list = sorted(Directory._inside_search(self._Root_dir + _input_dir))
                    # output
                    _label_dir = self._Directory[label_style][label_io].format(self._Active_mode.value)
                    _label_list = sorted(Directory._inside_search(self._Root_dir + _label_dir))

                    _selected_data._Data_list = [{"input": _input_file, "label": _label_file} for _input_file, _label_file in zip(_input_list, _label_list)]

                elif label_io == IO_Style.ANNOTATION:
                    ...

            return _selected_data

        def _work(self, data: Work_Profile, index: int) -> Dict[str, ndarray]:
            _pick_data = data._Data_list[index]
            _holder = {"info": Array_Process._converter(index, dtype=Np_Dtype.INT)}

            if data._Label_style == Label_Style.SEMENTIC_SEG:
                # Get data from each data source
                if data._Label_IO == IO_Style.IMAGE_FILE:
                    _input = File_IO._image_read(_pick_data["input"])
                    _label = File_IO._image_read(_pick_data["label"])

                else:  # data._Label_IO == IO_Style.ANNOTATION
                    _input = None
                    _label = None

                _label = None if _label is None else Label_Img_Process._color_map_to_classification(_label, self._Activate_label[data._Label_style])
                _holder.update({"input": _input}) if _input is not None else ...
                _holder.update({"label": _label}) if _label is not None else ...

            return _holder

    class Imagenet_1k(Basement):
        Directory: Dict[Label_Style, Dict[IO_Style, Optional[Union[List[str], str]]]] = {
            Label_Style.CLASSIFICATION: {
                IO_Style.IMAGE_FILE: "ILSVRC/2012/{}/{}/",
                IO_Style.ZIP_FILE: "ILSVRC/2012/",
                IO_Style.ANNOTATION: "ILSVRC/2012/"}
        }
        Annotation: Dict[Label_Style, Dict[IO_Style, Optional[Union[List[str], str]]]] = {
            Label_Style.CLASSIFICATION: {
                IO_Style.IMAGE_FILE: None,
                IO_Style.ZIP_FILE: "{}_map_for_zip.txt",
                IO_Style.ANNOTATION: "{}_map_for_annotation.json"}
        }

    class Imagenet_22k(Basement):
        ...

    class PASCAL(Basement):
        ...

    class CDnet(Basement):
        Directory: Dict[Label_Style, Dict[IO_Style, Union[List[str], str]]] = {
            Label_Style.SEMENTIC_SEG: {
                IO_Style.IMAGE_FILE: "",
                IO_Style.ZIP_FILE: "",
                IO_Style.ANNOTATION: ""}
        }
        Annotation: Dict[Label_Style, Dict[IO_Style, Union[List[str], str]]] = {
            Label_Style.SEMENTIC_SEG: {
                IO_Style.ZIP_FILE: "",
                IO_Style.ANNOTATION: ""}
        }

        def get_data_profile(self, holder: Work_Profile) -> Work_Profile:
            ...

        def work(self, data, index):
            ...

    class COCO(Basement):
        def __init__(self):
            pass

    @staticmethod
    def _build(name: Support_Label, data_root: str, active_style: List[Label_Style], meta_file: Optional[str] = None) -> Basement:
        return Label_Process.__dict__[name.name](name, data_root, active_style, meta_file)
