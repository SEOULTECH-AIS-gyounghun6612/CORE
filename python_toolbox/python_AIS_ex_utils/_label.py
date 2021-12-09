from collections import namedtuple

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


# Label_lsit template
class information_tool():
    info_dict = {}
    category = {}

    IGNORE_IDs = [255, ]
    active_id_dict = {}
    active_name_dict = {}
    sequence_tpye_label_list = []

    def __init__(self, label_type, ignore_ids=None) -> None:
        self.label_type = label_type
        self.is_sequence = label_type in self.sequence_tpye_label_list
        self.set_active_label(ignore_ids)

    def set_active_label(self, ignore_ids):
        # selected label list check
        for _id_ct, _tem_componant in enumerate(self.info_dict[self.label_type]):
            if _tem_componant.id != _id_ct:
                # label list has some missing
                _error.variable_stop(
                    function_name=self.__class__.__name__ + "__init__",
                    variable_list=["info_dict[{}]".format(self.label_type), ],
                    AA="If not label id start 0, change it.\n\
                        else not, info_dict check again. Label list has some missing.")

        # set ignore list
        _ig_ids = ignore_ids if ignore_ids is not None else self.IGNORE_IDs

        _ignore_list = []  # ignore labels
        # make active dict
        for _data in self.info_dict[self.label_type]:
            _tem_id = _data.train_id
            if _tem_id not in _ig_ids:  # using labels
                self.active_id_dict[_tem_id] = [_data.id, ] if _tem_id not in self.active_id_dict\
                    else self.active_id_dict[_tem_id] + [_data.id, ]
                self.active_name_dict[_data.name] = _tem_id
            else:  # ignore labels
                _ignore_list.append(_data.id)
                self.active_name_dict[_data.name] = -1

        # add ignore data in train_id_dict
        self.active_id_dict[-1] = _ignore_list

    def get_color_list(self):
        _class_color_list = []

        # using labels color
        for _id in range(self.get_label_ct() - 1):
            _class_color_list.append(
                [x.color for x in self.get_data(_id)])

        # ignore label color -> black
        _class_color_list.append([[0x00 for _ct in range(len(_class_color_list[0][0]))], ])

        return _class_color_list

    def get_label_ct(self):
        return len(self.active_id_dict)

    def get_data(self, signal):
        if isinstance(signal, int):
            # make retrun data list from train id
            return [self.info_dict[self.label_type][_ct] for _ct in self.active_id_dict[signal]]

        elif isinstance(signal, str):
            # make retrun data from class name
            # signal input convert to id num
            _tem_id = self.active_name_dict[signal] if signal in self.active_name_dict.keys()\
                else -1
            # get data from original
            return self.get_data(_tem_id)

        else:  # incorrect signal
            pass

    # for data transformation term
    # classfication -> (class count, h, w)
    # class map     -> (h, w, 1)
    # color map     -> (h, w, 3)

    def get_color_map_from(self, class_map):
        return _numpy.image_extention.class_map_to_color_map(class_map, self.get_color_list())

    def get_classfication_from(self, color_map, is_last_ch=True):
        return _numpy.image_extention.color_map_to_classfication(color_map, self.get_color_list())

    def get_class_map_from(self, classfication, is_last_ch=False):
        return _numpy.image_extention.classfication_to_class_map(classfication)


class data_tool():
    # baseline
    data_root = None
    input_len = 0

    data_list = []  # componant: [[input_data, label_data], ...]
    process_config = {}

    def __init__(self, data_root, source_style="file") -> None:
        self.data_root = _base.directory._slash_check(data_root)
        if not _base.directory._exist_check(self.data_root):  # data root folder not found
            _error.variable_stop(
                function_name="file_process_tool.__init__",
                variable_list=["data_root", ],
                AA="Error in parameter 'data_root'.\n \
                    Directory {} not exist".format(data_root)
            )

        self.source_style = source_style

    def get_data_len(self):
        return len(self.data_list)

    def pick_data(self, item_num, shape):
        return self.make_data_from_list(self.data_list[item_num], shape)

    def make_data_from_list(self, pick_data, shape):
        pass

    # *.from_*_process => make data list from each data source
    # source_style : image file (color_map, masks, ...)
    input_ext = ".jpg"
    label_ext = ".png"

    # source_style : polygon
    annotation_ext = ".json"

    # make the data list from each source
    def make_datalist(self, data_type):
        pass

    def detector(self, root, is_sequence, ext="*", name="*"):
        option = {"name": name, "ext": ext}
        if is_sequence:
            folder_list = _base.directory._inside_search(root, "directory")
            return [_base.directory._inside_search(folder, **option) for folder in folder_list]
        else:
            return _base.directory._inside_search(root, **option)


class BDD_100K(information_tool, data_tool):
    info_dict = {
        "seg": [  # "id", "train_id", "categoryId", "hasInstances", "ignoreInEval", "color", "name"
            label(0x00, 0xFF, 0x00, 0x00, 0x01, [0x00, 0x00, 0x00], "unlabeled"),
            label(0x01, 0xFF, 0x00, 0x00, 0x01, [0x00, 0x4A, 0x6F], "dynamic"),
            label(0x02, 0xFF, 0x00, 0x00, 0x01, [0x00, 0x00, 0x00], "ego vehicle"),
            label(0x03, 0xFF, 0x00, 0x00, 0x01, [0x51, 0x00, 0x51], "ground"),
            label(0x04, 0xFF, 0x00, 0x00, 0x01, [0x00, 0x00, 0x00], "static"),
            label(0x05, 0xFF, 0x01, 0x00, 0x01, [0xA0, 0xAA, 0xFA], "parking"),
            label(0x06, 0xFF, 0x01, 0x00, 0x01, [0x8C, 0x96, 0xE6], "rail track"),
            label(0x07, 0x00, 0x01, 0x00, 0x00, [0x80, 0x40, 0x80], "road"),
            label(0x08, 0x01, 0x01, 0x00, 0x00, [0xE8, 0x23, 0xF4], "sidewalk"),
            label(0x09, 0xFF, 0x02, 0x00, 0x01, [0x64, 0x64, 0x96], "bridge"),
            label(0x0A, 0x02, 0x02, 0x00, 0x00, [0x46, 0x46, 0x46], "building"),
            label(0x0B, 0x04, 0x02, 0x00, 0x00, [0x99, 0x99, 0xBE], "fence"),
            label(0x0C, 0xFF, 0x02, 0x00, 0x01, [0xB4, 0x64, 0xB4], "garage"),
            label(0x0D, 0xFF, 0x02, 0x00, 0x01, [0xB4, 0xA5, 0xB4], "guard rail"),
            label(0x0E, 0xFF, 0x02, 0x00, 0x01, [0x5A, 0x78, 0x96], "tunnel"),
            label(0x0F, 0x03, 0x02, 0x00, 0x00, [0x9C, 0x66, 0x66], "wall"),
            label(0x10, 0xFF, 0x03, 0x00, 0x01, [0x64, 0xAA, 0xFA], "banner"),
            label(0x11, 0xFF, 0x03, 0x00, 0x01, [0xFA, 0xDC, 0xDC], "billboard"),
            label(0x12, 0xFF, 0x03, 0x00, 0x01, [0x00, 0xA5, 0xFF], "lane divider"),
            label(0x13, 0xFF, 0x03, 0x00, 0x00, [0x3C, 0x14, 0xDC], "parking sign"),
            label(0x14, 0x05, 0x03, 0x00, 0x00, [0x99, 0x99, 0x99], "pole"),
            label(0x15, 0xFF, 0x03, 0x00, 0x01, [0x99, 0x99, 0x99], "polegroup"),
            label(0x16, 0xFF, 0x03, 0x00, 0x01, [0x64, 0xDC, 0xDC], "street light"),
            label(0x17, 0xFF, 0x03, 0x00, 0x01, [0x00, 0x46, 0xFF], "traffic cone"),
            label(0x18, 0xFF, 0x03, 0x00, 0x01, [0xDC, 0xDC, 0xDC], "traffic device"),
            label(0x19, 0x06, 0x03, 0x00, 0x00, [0x1E, 0xAA, 0xFA], "traffic light"),
            label(0x1A, 0x07, 0x03, 0x00, 0x00, [0x00, 0xDC, 0xDC], "traffic sign"),
            label(0x1B, 0xFF, 0x03, 0x00, 0x01, [0xFA, 0xAA, 0xFA], "traffic sign frame"),
            label(0x1C, 0x09, 0x04, 0x00, 0x00, [0x98, 0xFB, 0x98], "terrain"),
            label(0x1D, 0x08, 0x04, 0x00, 0x00, [0x23, 0x8E, 0x6B], "vegetation"),
            label(0x1E, 0x0A, 0x05, 0x00, 0x00, [0xB4, 0x82, 0x46], "sky"),
            label(0x1F, 0x0B, 0x06, 0x01, 0x00, [0x3C, 0x14, 0xDC], "person"),
            label(0x20, 0x0C, 0x06, 0x01, 0x00, [0x00, 0x00, 0xFF], "rider"),
            label(0x21, 0x12, 0x07, 0x01, 0x00, [0x20, 0x0B, 0x77], "bicycle"),
            label(0x22, 0x0F, 0x07, 0x01, 0x00, [0x64, 0x3C, 0x00], "bus"),
            label(0x23, 0x0D, 0x07, 0x01, 0x00, [0x8E, 0x00, 0x00], "car"),
            label(0x24, 0xFF, 0x07, 0x01, 0x01, [0x5A, 0x00, 0x00], "caravan"),
            label(0x25, 0x11, 0x07, 0x01, 0x00, [0xE6, 0x00, 0x00], "motorcycle"),
            label(0x26, 0xFF, 0x07, 0x01, 0x01, [0x6E, 0x00, 0x00], "trailer"),
            label(0x27, 0x10, 0x07, 0x01, 0x00, [0x64, 0x50, 0x00], "train"),
            label(0x28, 0x0E, 0x07, 0x01, 0x00, [0x46, 0x00, 0x00], "truck")]
    }
    category = {
        "seg": ["void", "flat", "construction", "object", "nature", "sky", "human", "vehicle"]
    }
    process_config = {
        "seg_color_map": {
            "input_dir": "images/10k/{}/",  # data_type
            "label_dir": "labels/sem_seg/colormaps/{}/"
        },
        "seg_polygons": {
            "input_dir": "images/10k/{}/",
            "label_dir": "labels/sem_seg/polygons/{}/",
        },
    }

    sequence_tpye_label_list = ["seg_track_20", ]

    def __init__(self, data_root, source_style, label_type, ignore_ids=None):
        data_tool.__init__(self, data_root, source_style)
        information_tool.__init__(self, label_type, ignore_ids)

        self.active_config = self.process_config["{}_{}".format(label_type, source_style)]

    def make_data_from_list(self, pick_data, shape):
        if self.source_style == "polygons":
            # from annotation
            pass
        elif self.source_style == "color_map":
            if isinstance(pick_data, int):  # test
                pass
            elif isinstance(pick_data, tuple):  # train, validation
                input_array = _cv2.file.image_read(pick_data[0], _cv2.Color_option.BGR)
                input_array = _cv2.base_process.resize(input_array, shape)
                label_array = _cv2.file.image_read(pick_data[1], _cv2.Color_option.BGR)
                label_array = self.get_classfication_from(label_array)
                label_array = _numpy.image_extention.classfication_resize(label_array, shape)

                return input_array, label_array

    def make_datalist(self, data_type):
        _input_root = self.data_root + self.active_config["input_dir"]
        _label_root = self.data_root + self.active_config["label_dir"]
        _sq = self.is_sequence

        if self.source_style == "polygons":
            # from json annotation
            pass
        elif self.source_style == "color_map":
            # from color_map image file
            _label_option = {"root": _label_root.format(data_type), "is_sequence": _sq, "ext": self.label_ext}
            _input_option = {"root": _input_root.format(data_type), "is_sequence": _sq, "ext": self.input_ext}

            self.data_list = self.detector(**_input_option) if data_type == "test"\
                else list(zip(self.detector(**_input_option), self.detector(**_label_option)))


class CDnet(information_tool, data_tool):
    info_dict = {
        "original": [  # "id", "train_id", "categoryId", "hasInstances", "ignoreInEval", "color", "name"
            label(0x00, 0x00, 0x01, 0x00, 0x00, [0x10, 0x10, 0x10], "Static"),
            label(0x01, 0x00, 0x00, 0x00, 0x00, [0x32, 0x32, 0x32], "Hard shadow"),
            label(0x02, 0xFF, 0x00, 0x00, 0x01, [0x55, 0x55, 0x55], "Outside region of interest"),
            label(0x03, 0x01, 0x00, 0x00, 0x00, [0xAA, 0xAA, 0xAA], "Unknown motion"),
            label(0x04, 0x02, 0x01, 0x00, 0x00, [0xFF, 0xFF, 0xFF], "Motion")]
    }
    category = {
        "seg": ["void", "flat", "construction", "object", "nature", "sky", "human", "vehicle"]
    }
    process_config = {
        "original_color_map": {
            "input_dir": "{}/{}/input/",
            "label_dir": "{}/{}/groundtruth/",
            "train_list": [],
            "test_list": []
        }
    }

    def __init__(self, label_type, data_root, source_style, ignore_ids=None) -> None:
        data_tool.__init__(self, data_root, source_style)
        information_tool.__init__(self, label_type, ignore_ids)

    def get_matched_file_name(self, file_name):
        _file_name = _base.file._name_from_directory(file_name)
        return _file_name.replace(self.input_ext, self.label_ext).replace("in", "gt")

    def processing(self, pick_data, call_sign):
        pass

    def from_file_process(self):
        pass


class COCO_style(information_tool, data_tool):
    def __init__(self, label_type, ignore_ids=None) -> None:
        super().__init__(label_type, ignore_ids=ignore_ids)
