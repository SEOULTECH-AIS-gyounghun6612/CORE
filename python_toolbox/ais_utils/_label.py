from . import _base
from . import _cv2
from . import _numpy

from . import _error as _e

# Set constant
DEBUG = False
_error = _e.Custom_error(
    module_name="ais_custom_utils_v 2.x",
    file_name="_label.py")


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
             "color": [0x6F, 0x4A, 0x00]},
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
             "color": [0xFA, 0xAA, 0xA0]},
            {"name": "rail track",
             "id": 6,
             "train_id": 255,
             "category": "flat",
             "categoryId": 1,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0xE6, 0x96, 0x8C]},
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
             "color": [0xF4, 0x23, 0xE8]},
            {"name": "bridge",
             "id": 9,
             "train_id": 255,
             "category": "construction",
             "categoryId": 2,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0x96, 0x64, 0x64]},
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
             "color": [0xBE, 0x99, 0x99]},
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
             "color": [0x96, 0x78, 0x5A]},
            {"name": "wall",
             "id": 15,
             "train_id": 3,
             "category": "construction",
             "categoryId": 2,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0x66, 0x66, 0x9C]},
            {"name": "banner",
             "id": 16,
             "train_id": 255,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0xFA, 0xAA, 0x64]},
            {"name": "billboard",
             "id": 17,
             "train_id": 255,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0xDC, 0xDC, 0xFA]},
            {"name": "lane divider",
             "id": 18,
             "train_id": 255,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0xFF, 0xA5, 0x00]},
            {"name": "parking sign",
             "id": 19,
             "train_id": 255,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0xDC, 0x14, 0x3C]},
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
             "color": [0xDC, 0xDC, 0x64]},
            {"name": "traffic cone",
             "id": 23,
             "train_id": 255,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": True,
             "color": [0xFF, 0x46, 0x00]},
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
             "color": [0xFA, 0xAA, 0x1E]},
            {"name": "traffic sign",
             "id": 26,
             "train_id": 7,
             "category": "object",
             "categoryId": 3,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0xDC, 0xDC, 0x00]},
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
             "color": [0x6B, 0x8E, 0x23]},
            {"name": "sky",
             "id": 30,
             "train_id": 10,
             "category": "sky",
             "categoryId": 5,
             "hasInstances": False,
             "ignoreInEval": False,
             "color": [0x46, 0x82, 0xB4]},
            {"name": "person",
             "id": 31,
             "train_id": 11,
             "category": "human",
             "categoryId": 6,
             "hasInstances": True,
             "ignoreInEval": False,
             "color": [0xDC, 0x14, 0x3C]},
            {"name": "rider",
             "id": 32,
             "train_id": 12,
             "category": "human",
             "categoryId": 6,
             "hasInstances": True,
             "ignoreInEval": False,
             "color": [0xFF, 0x00, 0x00]},
            {"name": "bicycle",
             "id": 33,
             "train_id": 18,
             "category": "vehicle",
             "categoryId": 7,
             "hasInstances": True,
             "ignoreInEval": False,
             "color": [0x77, 0x0B, 0x20]},
            {"name": "bus",
             "id": 34,
             "train_id": 15,
             "category": "vehicle",
             "categoryId": 7,
             "hasInstances": True,
             "ignoreInEval": False,
             "color": [0x00, 0x3C, 0x64]},
            {"name": "car",
             "id": 35,
             "train_id": 13,
             "category": "vehicle",
             "categoryId": 7,
             "hasInstances": True,
             "ignoreInEval": False,
             "color": [0x00, 0x00, 0x8E]},
            {"name": "caravan",
             "id": 36,
             "train_id": 255,
             "category": "vehicle",
             "categoryId": 7,
             "hasInstances": True,
             "ignoreInEval": True,
             "color": [0x00, 0x00, 0x5A]},
            {"name": "motorcycle",
             "id": 37,
             "train_id": 17,
             "category": "vehicle",
             "categoryId": 7,
             "hasInstances": True,
             "ignoreInEval": False,
             "color": [0x00, 0x00, 0xE6]},
            {"name": "trailer",
             "id": 38,
             "train_id": 255,
             "category": "vehicle",
             "categoryId": 7,
             "hasInstances": True,
             "ignoreInEval": True,
             "color": [0x00, 0x00, 0x6E]},
            {"name": "train",
             "id": 39,
             "train_id": 16,
             "category": "vehicle",
             "categoryId": 7,
             "hasInstances": True,
             "ignoreInEval": False,
             "color": [0x00, 0x50, 0x64]},
            {"name": "truck",
             "id": 40,
             "train_id": 14,
             "category": "vehicle",
             "categoryId": 7,
             "hasInstances": True,
             "ignoreInEval": False,
             "color": [0x00, 0x00, 0x46]}]
    }

    DATA_KEY_LIST = {
        "BDD-100K_seg": {
            "name_key": "name",
            "id_key": "train_id",
            "color_key": "color",
            "ignore_id_list": [255, ]
        }
    }

    def __init__(self, data_name, using_key) -> None:
        # set data info
        self.raw_data = self.LABEL_DATA_LIST[data_name]
        self.data_key = self.DATA_KEY_LIST[using_key]

        # selecte data
        self.id_label = {}
        self.name_label = {}

        # make class name key label
        self.label_data_init()

    def label_data_init(self):
        # make train_id_label
        for _data in self.raw_data:
            if _data[self.data_key["id_key"]] not in self.data_key["ignore_id_list"]:
                # using this data
                self.id_label[self.data_key["id_key"]] = _data
                self.name_label[self.data_key["name_key"]] = _data

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
            color_list.append(self.id_label(_id)[self.data_key["color_key"]])

        return color_list

    def get_color_map_from(self, class_map, is_last_ch=False):
        color_list = self.get_color_list()
        classfication = _numpy.image_extention.class_map_to_classfication(class_map, is_last_ch)

        return color_list[classfication]

    def get_class_map_form(self, color_map):
        pass

    def get_len(self):
        return len(self.id_label.keys())


class Label_Style_Worker():
    # label holder
    # label style: annotation
    annotation_keys = None
    annotation_data = None

    # label style: color or grayscale image
    input_imgs = []
    label_imgs = []

    # setting
    # directory parameter
    data_root = None
    input_dir = None
    label_dir = None
    annotaion_dir = None

    # data call parameter
    input_len = None
    data_category = None
    label_file_style = None

    # label class parameter
    # label data class style must be list type
    label_class_data = None
    label_val_class_list = []

    def __init__(self, data_folder, label_file_style, is_train) -> None:
        self.data_root = _base.directory._slash_check(data_folder)
        if not _base.directory._exist_check(self.data_root):
            _error.variable_stop(
                function_name="Label_Style_Worker.__init__",
                variable_list=["data_folder", ],
                AA="Entered parameter 'data_folder' directory {} not exist".format(data_folder)
            )

        self.data_category = "train" if is_train else "val"
        self.label_file_style = label_file_style

    # common function
    def worker_initialize(self, label_info):
        # data dir init
        self.input_dir_initialize()
        # lable data init
        self.set_label_class_info(label_info)

        # get data from file
        if self.label_file_style == "Annotation":
            self.annotaion_dir_initialize()
            self.get_annotaion_data()
        elif self.label_file_style == "Seg_Map":
            pass

    def pick_data(self, item_num):
        # pick item
        if self.label_file_style == "Annotation":
            selected_item = self.annotation_data[item_num]
        elif self.label_file_style == "Seg_Map":
            pass

        # conver to data from file for train or val
        return self.data_style_converter(selected_item)

    # Individual function
    # # must define
    def input_dir_initialize(self):
        # parameter "input_dir" data initialize
        pass

    def set_label_class_info(self, label_info):
        pass

    def data_style_converter(self, selected_item):
        pass

    # # optional define
    def annotaion_dir_initialize(self):
        # parameter "annotaion_dir" data initialize
        pass

    def get_annotaion_data(self):
        # parameter "annotaion_dir" data initialize
        pass


class BDD_100K(Label_Style_Worker):
    # directory parameter formatting
    input_dir = "{}images/{}/{}/"  # root , input_category
    annotaion_dir = "{0}labels/{1}/polygons/{1}_{2}.json"  # root , label_category, data_style

    # additional data call parameter
    input_category = None
    label_category = None

    # label class parameter

    def __init__(
            self,
            data_folder: str,
            input_category: str,
            label_style,
            label_category,
            is_train) -> None:
        """
        Args:
            data_folder     : data root directory
            input_category  : 100k or 10k,
            label_category  : sem_seg,

        """
        # basement init
        super().__init__(data_folder, label_style, is_train)

        # addditional paramenter init
        self.input_category = input_category
        self.label_category = label_category

        # init
        self.worker_initialize()

    # # must define
    def input_dir_initialize(self):
        self.input_dir = self.input_dir.format(
            self.data_root,
            self.input_category,
            self.data_category)

    def set_label_class_info(self, label_info):
        self.label_class_data = label_info

        if self.label_category == "sem_seg":
            self.annotation_keys = {"class_name": "category", "draw_style": "poly2d"}
            self.label_val_class_list = []

    def data_style_converter(self, selected_item):
        if self.label_file_style == "Annotation":
            # read input data
            input_data = _cv2.read_img(
                self.input_dir + selected_item["name"],
                color_type=_cv2.COLOR_BGR,
                is_last_channel=False)
            _, _h, _w = input_data.shape

            # make label data from annotation
            anno_datas = selected_item["labels"]
            holder_ch = self.label_class_data.get_len()
            label_holder = _numpy.image_extention.get_canvus([holder_ch, _h, _w])

            for _data in anno_datas:
                _anno_class = _data[self.annotation_keys["class_name"]]
                _data_num = self.label_class_data.class_name_to_id(_anno_class)

                self.draw(
                    draw_style=self.annotation_keys["draw_style"],
                    draw_data=_data,
                    draw_color=255,
                    base=label_holder[_data_num]) if _data_num != -1 else None

            return input_data, label_holder

    # # optional define
    def annotaion_dir_initialize(self):
        # In later, add  function about "annotation file exist check" using _error module
        self.annotaion_dir = \
            self.annotaion_dir.format(self.data_root, self.label_category, self.data_category)

    def get_annotaion_data(self):
        self.annotation_data = _base.json_file(self.annotaion_dir)
        self.input_len = len(self.annotation_data)

    def draw(self, draw_style, draw_data, draw_color, base):
        if draw_style in draw_data.keys():
            draw_point_info = draw_data[draw_style]

            if draw_style == "poly2d":
                for _data in draw_point_info:
                    pts = _cv2.np.around(_data["vertices"]).astype(_cv2.np.int32)
                    base = _cv2.cv2.fillConvexPoly(base, pts, draw_color).astype(_cv2.np.uint8)
        return base


def load_check():
    print("!!! custom python module ais_utils _label load Success !!!")
