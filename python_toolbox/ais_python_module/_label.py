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
    [
        "name",
        "id",
        "train_id",
        "category",
        "categoryId",
        "hasInstances",
        "ignoreInEval",
        "color"
    ])


# Label_lsit template
class label_tool():
    DATA_LIST = {}
    CATEGORY = []

    IGNORE_IDs = [255, ]
    train_id_dict = {}
    name_dict = {}

    def __init__(self, label_type) -> None:
        self.label_type = label_type

        if self._data_list_has_problem():
            _error.variable_stop(
                function_name=self.__class__.__name__ + "__init__",
                variable_list=["DATA_LIST[{}]".format(self.label_type), ],
                AA="DAta list check again"
            )

        _ignore_list = []
        for _data in self.DATA_LIST[self.label_type]:
            # make train id dict
            _tem_id = _data.train_id
            if _tem_id not in self.IGNORE_IDs:
                # using this data
                self.train_id_dict[_tem_id] = [_data.id, ] if _tem_id not in self.train_id_dict\
                    else self.train_id_dict[_tem_id] + [_data.id, ]
            else:
                # ignore label collect
                _ignore_list.append(_data.id)

            # make name dict
            self.name_dict[_data.name] = _data.id

        # add ignore data in train_id_dict
        self.train_id_dict[self.get_label_num()] = _ignore_list

    def _data_list_has_problem(self):
        _flag = True
        _check = self.DATA_LIST[self.label_type]
        for _id_ct, _tem_componant in enumerate(_check):
            if _tem_componant.id != _id_ct:
                _flag = False
                break
        return _flag

    def get_label_num(self):
        return len(self.train_id_dict)

    def data_call(self, signal, signal_type="name"):
        if signal_type == "name":  # default
            # make retrun data from class name
            _tem_id = self.name_dict[signal]
            _dada = self.data_call(_tem_id, "id")

        elif signal_type == "train_id":
            # make retrun data list from train id
            _data = []
            for _tem_id in self.train_id_dict(signal):
                _data.append(self.data_call(_tem_id, "id"))

        else:  # return data
            return self.DATA_LIST[self.label_type][signal]

        return _dada


# Label template dict
class Labels():
    LABEL_DATA_LIST = {
        "BDD-100K": [
            {"name": "unlabeled",
             "id": 0,
             "train_id": 255,
             "category": "void",
             "categoryId": 0,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0x00, 0x00, 0x00]},
            {"name": "dynamic",
             "id": 1,
             "train_id": 255,
             "category": "void",
             "categoryId": 0,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0x00, 0x4A, 0x6F]},
            {"name": "ego vehicle",
             "id": 2,
             "train_id": 255,
             "category": "void",
             "categoryId": 0,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0x00, 0x00, 0x00]},
            {"name": "ground",
             "id": 3,
             "train_id": 255,
             "category": "void",
             "categoryId": 0,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0x51, 0x00, 0x51]},
            {"name": "static",
             "id": 4,
             "train_id": 255,
             "category": "void",
             "categoryId": 0,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0x00, 0x00, 0x00]},
            {"name": "parking",
             "id": 5,
             "train_id": 255,
             "category": "flat",
             "categoryId": 1,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0xA0, 0xAA, 0xFA]},
            {"name": "rail track",
             "id": 6,
             "train_id": 255,
             "category": "flat",
             "categoryId": 1,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0x8C, 0x96, 0xE6]},
            {"name": "road",
             "id": 7,
             "train_id": 0,
             "category": "flat",
             "categoryId": 1,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0x80, 0x40, 0x80]},
            {"name": "sidewalk",
             "id": 8,
             "train_id": 1,
             "category": "flat",
             "categoryId": 1,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0xE8, 0x23, 0xF4]},
            {"name": "bridge",
             "id": 9,
             "train_id": 255,
             "category": "construction",
             "categoryId": 2,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0x64, 0x64, 0x96]},
            {"name": "building",
             "id": 10,
             "train_id": 2,
             "category": "construction",
             "categoryId": 2,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0x46, 0x46, 0x46]},
            {"name": "fence",
             "id": 11,
             "train_id": 4,
             "category": "construction",
             "categoryId": 2,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0x99, 0x99, 0xBE]},
            {"name": "garage",
             "id": 12,
             "train_id": 255,
             "category": "construction",
             "categoryId": 2,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0xB4, 0x64, 0xB4]},
            {"name": "guard rail",
             "id": 13,
             "train_id": 255,
             "category": "construction",
             "categoryId": 2,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0xB4, 0xA5, 0xB4]},
            {"name": "tunnel",
             "id": 14,
             "train_id": 255,
             "category": "construction",
             "categoryId": 2,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0x5A, 0x78, 0x96]},
            {"name": "wall",
             "id": 15,
             "train_id": 3,
             "category": "construction",
             "categoryId": 2,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0x9C, 0x66, 0x66]},
            {"name": "banner",
             "id": 16,
             "train_id": 255,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0x64, 0xAA, 0xFA]},
            {"name": "billboard",
             "id": 17,
             "train_id": 255,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0xFA, 0xDC, 0xDC]},
            {"name": "lane divider",
             "id": 18,
             "train_id": 255,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0x00, 0xA5, 0xFF]},
            {"name": "parking sign",
             "id": 19,
             "train_id": 255,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0x3C, 0x14, 0xDC]},
            {"name": "pole",
             "id": 20,
             "train_id": 5,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0x99, 0x99, 0x99]},
            {"name": "polegroup",
             "id": 21,
             "train_id": 255,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0x99, 0x99, 0x99]},
            {"name": "street light",
             "id": 22,
             "train_id": 255,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0x64, 0xDC, 0xDC]},
            {"name": "traffic cone",
             "id": 23,
             "train_id": 255,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0x00, 0x46, 0xFF]},
            {"name": "traffic device",
             "id": 24,
             "train_id": 255,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0xDC, 0xDC, 0xDC]},
            {"name": "traffic light",
             "id": 25,
             "train_id": 6,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0x1E, 0xAA, 0xFA]},
            {"name": "traffic sign",
             "id": 26,
             "train_id": 7,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0x00, 0xDC, 0xDC]},
            {"name": "traffic sign frame",
             "id": 27,
             "train_id": 255,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0xFA, 0xAA, 0xFA]},
            {"name": "terrain",
             "id": 28,
             "train_id": 9,
             "category": "nature",
             "categoryId": 4,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0x98, 0xFB, 0x98]},
            {"name": "vegetation",
             "id": 29,
             "train_id": 8,
             "category": "nature",
             "categoryId": 4,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0x23, 0x8E, 0x6B]},
            {"name": "sky",
             "id": 30,
             "train_id": 10,
             "category": "sky",
             "categoryId": 5,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0xB4, 0x82, 0x46]},
            {"name": "person",
             "id": 31,
             "train_id": 11,
             "category": "human",
             "categoryId": 6,
             "hasInstances": True,
             "ignoreInEval": False,
             "color": [0x3C, 0x14, 0xDC]},
            {"name": "rider",
             "id": 32,
             "train_id": 12,
             "category": "human",
             "categoryId": 6,
             "hasInstances": True,
             "ignoreInEval": False,
             "color": [0x00, 0x00, 0xFF]},
            {"name": "bicycle",
             "id": 33,
             "train_id": 18,
             "category": "vehicle",
             "categoryId": 7,
             "hasInstances": True,
             "ignoreInEval": False,
             "color": [0x20, 0x0B, 0x77]},
            {"name": "bus",
             "id": 34,
             "train_id": 15,
             "category": "vehicle",
             "categoryId": 7,
             "hasInstances": True,
             "ignoreInEval": False,
             "color": [0x64, 0x3C, 0x00]},
            {"name": "car",
             "id": 35,
             "train_id": 13,
             "category": "vehicle",
             "categoryId": 7,
             "hasInstances": True,
             "ignoreInEval": False,
             "color": [0x8E, 0x00, 0x00]},
            {"name": "caravan",
             "id": 36,
             "train_id": 255,
             "category": "vehicle",
             "categoryId": 7,
             "hasInstances": True,
             "ignoreInEval": True,
             "color": [0x5A, 0x00, 0x00]},
            {"name": "motorcycle",
             "id": 37,
             "train_id": 17,
             "category": "vehicle",
             "categoryId": 7,
             "hasInstances": True,
             "ignoreInEval": False,
             "color": [0xE6, 0x00, 0x00]},
            {"name": "trailer",
             "id": 38,
             "train_id": 255,
             "category": "vehicle",
             "categoryId": 7,
             "hasInstances": True,
             "ignoreInEval": True,
             "color": [0x6E, 0x00, 0x00]},
            {"name": "train",
             "id": 39,
             "train_id": 16,
             "category": "vehicle",
             "categoryId": 7,
             "hasInstances": True,
             "ignoreInEval": False,
             "color": [0x64, 0x50, 0x00]},
            {"name": "truck",
             "id": 40,
             "train_id": 14,
             "category": "vehicle",
             "categoryId": 7,
             "hasInstances": True,
             "ignoreInEval": False,
             "color": [0x46, 0x00, 0x00]}
        ],
        "CDnet_2014": [
            {"name": "Static",
             "id": 0,
             "train_id": 0,
             "category": "void",
             "categoryId": 0,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0x00, 0x00, 0x00], },
            {"name": "Hard shadow",
             "id": 1,
             "train_id": 0,
             "category": "void",
             "categoryId": 0,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0x32, 0x32, 0x32], },
            {"name": "Outside region of interest",
             "id": 2,
             "train_id": 255,
             "category": "void",
             "categoryId": 0,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0x55, 0x55, 0x55], },
            {"name": "Unknown motion",
             "id": 3,
             "train_id": 1,
             "category": "void",
             "categoryId": 0,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0xAA, 0xAA, 0xAA], },
            {"name": "Motion",
             "id": 4,
             "train_id": 2,
             "category": "void",
             "categoryId": 0,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0xFF, 0xFF, 0xFF], }
        ],
    }

    DEFAULT_SETTING = {
        "sem_seg": {
            "name_key": "name",
            "id_key": "train_id",
            "color_key": "color",
            "ignore_id_in_raw": 0,
            "ignore_id_list": [255, ]
        }
    }

    def __init__(self, data_name, using_key) -> None:
        # set data info
        self.raw_data = self.LABEL_DATA_LIST[data_name]
        self.label_style = using_key

        # in later add safty for human error about using_key
        self.data_key = self.DEFAULT_SETTING[using_key]
        # make error

        # selecte data
        self.id_label = {}
        self.name_label = {}

        # make class name key label
        self.label_data_init()

    def label_data_init(self):
        # make train_id_label
        _id_key = self.data_key["id_key"]
        _ignore_id_list = self.data_key["ignore_id_list"]
        _ignore = []

        for _data in self.raw_data:
            _id = _data[_id_key]
            _name = _data[self.data_key["name_key"]]
            if _id not in _ignore_id_list:
                # using this data
                self.id_label[_id] = [_data, ] if _id not in self.id_label\
                    else self.id_label[_id] + [_data, ]
                self.name_label[_name] = _data

            else:
                _ignore.append(_data)
                self.name_label[_name] = _data
        self.id_label[len(self.id_label.keys())] = _ignore

    def data_from(self, call, data):
        # get selected
        selected_label = self.name_label if isinstance(call, str) else self.id_label

        if call not in selected_label.keys():
            _error.variable_stop(
                function_name="Labels.data_from",
                variable_list=["call", ],
                AA="Entered parameter 'call' data {} can't find in any label".format(call)
            )

        # return data
        if data == "color":
            return selected_label[call][self.data_key["color_key"]]
        elif data == "id":
            return selected_label[call][self.data_key["id_key"]]
        elif data == "class":
            return selected_label[call][self.data_key["name_key"]]
        else:
            return selected_label[call]

    def get_color_list(self):
        color_list = []
        for _id in sorted(self.id_label.keys()):
            color_list.append(self.id_label[_id][self.data_key["color_key"]])

        return color_list

    def get_class_list(self):
        return sorted(self.name_label.keys())

    def get_ids_list(self):
        return sorted(self.id_label.keys())

    def get_color_map_from(self, classfication):
        color_list = self.get_color_list()
        return _numpy.image_extention.classfication_to_color_map(classfication, color_list)

    def get_class_map_from(self, color_map, is_last_ch=False):
        return _numpy.image_extention.color_map_to_class_map(
            color_map,
            self.get_color_list(),
            is_last_ch)

    def get_classfication_from(self, class_map, is_last_ch=False):
        return _numpy.image_extention.class_map_to_classfication(class_map, is_last_ch)

    def get_len(self):
        return len(self.id_label.keys())


class Label_data_Worker():
    # label holder
    # label style: annotation
    annotation_data = None

    # label style: color or grayscale image
    image_list = []
    input_ext = None
    label_ext = None

    # setting
    # directory parameter
    data_root = None
    input_dir = None
    label_dir = None

    # data info
    input_len = None
    using_option = None
    source = None
    surport_source = ["Annotation", "Color_map"]

    # label class parameter
    # label data class style must be list type
    label_info = None
    label_profile = {
        "label_style": "",
        "input_dir": "",
        "label_dir": ""
    }

    def __init__(self, data_folder, source, option) -> None:
        # set label data root
        self.data_root = _base.directory._last_slash_in_dir_check(data_folder)
        if not _base.directory._exist_check(self.data_root):
            _error.variable_stop(
                function_name="Label_Style_Worker.__init__",
                variable_list=["data_folder", ],
                AA="Error in parameter 'data_folder'.\n \
                    Directory {} not exist".format(data_folder)
            )

        self.using_option = option if option in ["train", "val", "test"] else "test"
        # in later fix this paramnets name -> is_not_test
        self.is_NOT_test = self.using_option != "test"
        is_surport_source = source in self.surport_source
        self.source = source if (is_surport_source and self.is_NOT_test) else "Color_map"

    # common function
    def _worker_initialize(self, label):
        # label information init
        self._label_info_init(label)

        # data dir init
        self._input_dir_initialize()
        self._label_dir_initialize()

        # get data from file
        if self.source == "Annotation":
            self._get_annotaion_data()
        elif self.source == "Color_map":
            self._get_colormap_data()

    def pick_data(self, item_num):
        # pick item
        if self.source == self.surport_source[0]:  # Annotation
            selected_item = self.annotation_data[item_num]
        elif self.source == self.surport_source[1]:  # Color_map
            selected_item = self.image_list[item_num]

        # conver to data from file for train or val
        return self.data_style_converter(selected_item)

    def _label_info_init(self, label):
        # lebel_style and label_profile compatibility check
        self.label_info = label
        self.label_style = label.label_style

        if self.label_style != self.label_profile["label_style"]:
            _error.variable_stop(
                function_name="Label_Style_Worker.__init__",
                variable_list=["label", ],
                AA="Error in parameter 'label style'.\n \
                    Directory {} not exist".format(self.label_style)
            )

    def _input_dir_initialize(self):
        # parameter "input_dir" data initialize
        self.input_dir = self.data_root + self.label_profile["input_dir"]
        self.input_dir = _base.directory._last_slash_in_dir_check(self.input_dir)

    def _label_dir_initialize(self):
        # parameter "annotaion_dir" data initialize
        # In later, add  function about "annotation file exist check" using _error module
        if self.is_NOT_test:
            self.label_dir = self.data_root + self.label_profile["label_dir"]
            self.label_dir = _base.directory._last_slash_in_dir_check(self.label_dir)

    # Individual function
    # # must define
    def _data_style_converter(self, selected_item):
        pass

    # # optional define
    def _get_annotaion_data(self):
        # parameter "annotaion_dir" data initialize
        pass

    def _get_colormap_data(self):
        # parameter "annotaion_dir" data initialize
        pass

    def _label_file_check(self, input_file):
        # parameter "annotaion_dir" data initialize
        pass


class BDD_100K(Label_data_Worker):
    label_profiles = {
        "sem_seg_Color_map": {
            "label_style": "sem_seg",
            "input_dir": "images/10k/",
            "label_dir": "labels/sem_seg/colormaps/"
        },
        "sem_seg_Annotation": {
            "label_style": "sem_seg",
            "input_dir": "images/10k/",
            "label_dir": "labels/sem_seg/polygons/",
            "name_key": "name",
            "data_key": "labels",
            "class_key": "category"
        },
    }

    # label class parameter

    def __init__(
            self,
            data_folder,
            source_style,
            label_style="sem_seg",
            option="train") -> None:
        """
        Args:
            data_folder     : data root directory
            input_category  : 100k or 10k,
            label_category  : sem_seg,

        """
        # set label_profile
        self.label_profile = self.label_profiles[label_style + "_" + source_style]
        # basement init
        super().__init__(data_folder + "bdd-100k", source_style, option)

        self.input_ext = ".jpg"
        self.label_ext = ".png"

        # init
        self.worker_initialize(Labels("BDD-100K", label_style))

    def data_style_converter(self, selected_item):
        if self.source_style == self.surport_source[0]:  # Annotation
            # read input data
            input_data = _cv2.file.image_read(
                self.input_dir + selected_item[self.label_profile["name_key"]],
                _cv2.Color_option.BGR)
            input_data = _cv2.base_process.channel_converter(input_data, _cv2.C_position.First)
            _, _h, _w = input_data.shape

            # make label data from annotation
            anno_datas = selected_item["labels"]
            ignore_ch = self.label_info.get_len() - 1  # ignore
            label_holder = _numpy.image_extention.get_canvus([_h, _w], background_color=ignore_ch)

            for _data in anno_datas:
                _member_class = _data[self.label_profile["class_key"]]
                if _member_class in self.label_info.get_class_list():
                    _id_num = self.label_info.data_from(_member_class, "id")
                else:
                    _id_num = -1

                if "poly2d" in _data.keys():
                    self.draw(
                        draw_style="poly2d",
                        draw_data=_data["poly2d"],
                        draw_color=_id_num if _id_num != -1 else ignore_ch,
                        base=label_holder)

            return input_data, label_holder

        elif self.source_style == self.surport_source[1]:  # color_map or input only
            # read input data
            input_data = _cv2.file.image_read(selected_item[0], _cv2.Color_option.BGR)
            input_data = _cv2.base_process.channel_converter(input_data, _cv2.C_position.First)

            if self.is_NOT_test:
                # read label color_map
                color_map = _cv2.file.image_read(selected_item[1], _cv2.Color_option.BGR)
                color_map = _cv2.base_process.channel_converter(color_map, _cv2.C_position.First)

                class_map = self.label_info.get_class_map_from(color_map)

                return input_data, class_map
            else:
                return input_data, None

    # # optional define
    def get_annotaion_data(self):
        self.input_dir = _base.directory._last_slash_in_dir_check(self.input_dir + self.using_option)
        self.annotation_data = \
            _base.file._json(self.label_dir, "{}_{}.json".format(self.label_style, self.using_option))
        self.input_len = len(self.annotation_data)

    def get_colormap_data(self):
        # get input list
        input_imgs = _base.directory._inside_search(
            searched_dir=self.input_dir + self.using_option,
            search_option="file",
            ext=".jpg")

        # train, val
        if self.label_dir is not None:
            label_imgs = _base.directory._inside_search(
                searched_dir=self.label_dir + self.using_option,
                search_option="file",
                ext=".png")

            for _input_img in input_imgs:
                label_file = _base.file._name_from_directory(_input_img).replace(".jpg", ".png")
                _label_img = _base.directory._last_slash_in_dir_check(self.label_dir + self.using_option) + \
                    label_file

                if _label_img in label_imgs:
                    self.image_list.append([_input_img, _label_img])

            self.input_len = len(self.image_list)
        # test
        else:
            self.image_list.append([input_imgs, ])
            self.input_len = len(self.image_list)

    def draw(self, draw_style, draw_data, draw_color, base):
        if draw_style == "poly2d":
            for _data in draw_data:
                pts = _data["vertices"]
                base = _cv2.draw._polygon(base, pts, -1, draw_color)

        return base


class CDnet(Label_data_Worker):
    label_profiles = {
        "ori_three_class": {
            "label_style": "sem_seg",
            "input_dir": "{}/{}/input/",
            "label_dir": "{}/{}/groundtruth/",
            "train_list": [
                "badWeather/blizzard",
            ],
            "validation_list": []
        },
    }

    def __init__(
            self,
            data_folder,
            version,
            profile_name="ori_three_class",
            option="train") -> None:
        """
        Args:
            data_folder     : data root directory
            input_category  : 100k or 10k,
            label_category  : sem_seg,

        """
        # set label_profile
        self.label_profile = self.label_profiles[profile_name]
        # basement init
        super().__init__(data_folder + "CDnet/" + version, "Color_map", option)

        self.input_ext = ".jpg"
        self.label_ext = ".png"

        # init
        self._worker_initialize(Labels("CDnet_2014", "sem_seg"))

    def data_style_converter(self, selected_item):
        # read input data
        input_data = _cv2.file.image_read(selected_item[0], _cv2.Color_option.BGR)
        input_data = _cv2.base_process.channel_converter(input_data, _cv2.C_position.First)

        if self.is_NOT_test:
            # read label color_map
            color_map = _cv2.file.image_read(selected_item[1], _cv2.Color_option.BGR)
            color_map = _cv2.base_process.channel_converter(color_map, _cv2.C_position.First)

            class_map = self.label_info.get_class_map_from(color_map)

            return input_data, class_map
        else:
            return input_data, None

    # # optional define
    def get_colormap_data(self):
        data_list = "train_list" if self.using_option == "train" else "validation_list"
        for _category in self.label_profile[data_list]:
            # get info
            category_dir = _base.directory._last_slash_in_dir_check(self.data_root + _category)
            input_dir = self.input_dir.format(*_category.split("/"))
            if not _base.directory._exist_check(input_dir):
                _error.variable_stop(
                    function_name="CDnet.get_colormap_data",
                    variable_list=["input_dir", ],
                    AA="Directory {} not exist".format(input_dir)
                )

            info_file = open(category_dir + "temporalROI.txt", "r")
            info_datas = info_file.readline().split(" ")
            _start_num = int(info_datas[0]) - 1
            _end_num = int(info_datas[-1])

            # get input list
            input_imgs = _base.directory._inside_search(
                searched_dir=input_dir,
                search_option="file",
                ext=self.input_ext)

            if self.label_dir is not None:
                # get label directory
                label_dir = self.label_dir.format(*_category.split("/"))
                # directory check
                if not _base.directory._exist_check(label_dir):
                    _error.variable_stop(
                        function_name="CDnet.get_colormap_data",
                        variable_list=["label_dir", ],
                        AA="Directory {} not exist".format(label_dir)
                    )
                label_imgs = _base.directory._inside_search(
                    searched_dir=label_dir,
                    search_option="file",
                    ext=self.label_ext)

                for _input_img in input_imgs[_start_num:_end_num]:
                    _label_img = label_dir + self.label_file_check(_input_img)

                    if _label_img in label_imgs:
                        self.image_list.append([_input_img, _label_img])    

            # test
            else:
                self.image_list = input_imgs
            self.input_len = len(self.image_list)

    def label_file_check(self, input_file):
        _file_name = _base.file._name_from_directory(input_file)
        return _file_name.replace(self.input_ext, self.label_ext).replace("in", "gt")


def load_check():
    print("!!! custom python module ais_utils _label load Success !!!")
