from ais_utils import _error
import numpy as np

BDD_100k = {
    1: {"color": (0x00, 0x00, 0x00), "name": "unlabeled or statick"},
    2: {"color": (0x00, 0x4A, 0x6F), "name": "dynamic"},
    3: {"color": (0x51, 0x00, 0x51), "name": "ground"},
    4: {"color": (0xA0, 0xAA, 0xFA), "name": "parking"},
    5: {"color": (0x8C, 0x96, 0xE6), "name": "rail track"},
    6: {"color": (0x80, 0x40, 0x80), "name": "road"},
    7: {"color": (0xE8, 0x23, 0xF4), "name": "sidewalk"},
    8: {"color": (0x64, 0x64, 0x96), "name": "bridge"},
    9: {"color": (0x46, 0x46, 0x46), "name": "building"},
    10: {"color": (0x99, 0x99, 0xBE), "name": "fence"},
    11: {"color": (0xB4, 0x64, 0xB4), "name": "garage"},
    12: {"color": (0xB4, 0xA5, 0xB4), "name": "guard rail"},
    13: {"color": (0x5A, 0x78, 0x96), "name": "tunnel"},
    14: {"color": (0x9C, 0x66, 0x66), "name": "wall"},
    15: {"color": (0x64, 0xAA, 0xFA), "name": "banner"},
    16: {"color": (0xFA, 0xDC, 0xDC), "name": "billboard"},
    17: {"color": (0x00, 0xA5, 0xFF), "name": "lane divider"},
    18: {"color": (0x99, 0x99, 0x99), "name": "pole"},
    19: {"color": (0x64, 0xDC, 0xDC), "name": "street light"},
    20: {"color": (0x00, 0x46, 0xFF), "name": "traffic cone"},
    21: {"color": (0xDC, 0xDC, 0xDC), "name": "traffic device"},
    22: {"color": (0x1E, 0xAA, 0xFA), "name": "traffic light"},
    23: {"color": (0x00, 0xDC, 0xDC), "name": "traffic sign"},
    24: {"color": (0xFA, 0xAA, 0xFA), "name": "traffic sign frame"},
    25: {"color": (0x98, 0xFB, 0x98), "name": "terrain"},
    26: {"color": (0x23, 0x8E, 0x6B), "name": "vegetation"},
    27: {"color": (0xB4, 0x82, 0x46), "name": "sky"},
    28: {"color": (0x3C, 0x14, 0xDC), "name": "person or parking sign"},
    29: {"color": (0x00, 0x00, 0xFF), "name": "rider"},
    30: {"color": (0x20, 0x0B, 0x77), "name": "bicycle"},
    31: {"color": (0x64, 0x3C, 0x00), "name": "bus"},
    32: {"color": (0x8E, 0x00, 0x00), "name": "car"},
    33: {"color": (0x5A, 0x00, 0x00), "name": "caravan"},
    34: {"color": (0xE6, 0x00, 0x00), "name": "motorcycle"},
    35: {"color": (0x6E, 0x00, 0x00), "name": "trailer"},
    36: {"color": (0x64, 0x50, 0x00), "name": "train"},
    37: {"color": (0x46, 0x00, 0x00), "name": "truck"}}

CD_net_2014 = {
    1: {"color": (0x00, 0x00, 0x00), "name": "static"},
    2: {"color": (0x32, 0x32, 0x32), "name": "Hard shadow"},
    3: {"color": (0x55, 0x55, 0x55), "name": "Outside region of interest"},
    4: {"color": (0xAA, 0xAA, 0xAA), "name": "Unknown motion"},
    5: {"color": (0xFF, 0xFF, 0xFF), "name": "Motion"}}

COCO = {
    1: {"color": (0x38, 0xA4, 0xC4), "name": "person"},
    2: {"color": (0xB8, 0x24, 0x63), "name": "bicycle"},
    3: {"color": (0xB8, 0xC4, 0x59), "name": "car"},
    4: {"color": (0xE2, 0x3D, 0x72), "name": "motorcycle"},
    5: {"color": (0xEA, 0x59, 0xDD), "name": "airplane"},
    6: {"color": (0x3D, 0x90, 0x6D), "name": "bus"},
    7: {"color": (0x38, 0xB0, 0x24), "name": "train"},
    8: {"color": (0xCE, 0x3B, 0x42), "name": "truck"},
    9: {"color": (0x24, 0x45, 0x86), "name": "boat"},
    10: {"color": (0xE7, 0x31, 0x36), "name": "traffic light"},
    11: {"color": (0x8B, 0x0B, 0xEC), "name": "fire hydrant"},
    13: {"color": (0xAE, 0x1A, 0x9F), "name": "stop sign"},
    14: {"color": (0xAB, 0x9C, 0xC2), "name": "parking meter"},
    15: {"color": (0x90, 0xBD, 0xBF), "name": "bench"},
    16: {"color": (0xF1, 0x3B, 0xB3), "name": "bird"},
    17: {"color": (0x24, 0x9A, 0xB5), "name": "cat"},
    18: {"color": (0x40, 0x9C, 0x65), "name": "dog"},
    19: {"color": (0x95, 0xC2, 0x65), "name": "horse"},
    20: {"color": (0xB8, 0x04, 0x3B), "name": "sheep"},
    21: {"color": (0xA6, 0xDD, 0x92), "name": "cow"},
    22: {"color": (0x9A, 0x3B, 0x79), "name": "elephant"},
    23: {"color": (0xEA, 0x1A, 0xCE), "name": "bear"},
    24: {"color": (0xE2, 0x42, 0xFB), "name": "zebra"},
    25: {"color": (0xCC, 0x77, 0x74), "name": "giraffe"},
    27: {"color": (0x6D, 0xD6, 0xB3), "name": "backpack"},
    28: {"color": (0x18, 0x3B, 0x31), "name": "umbrella"},
    31: {"color": (0x72, 0x74, 0x1D), "name": "handbag"},
    32: {"color": (0x60, 0xFE, 0x79), "name": "tie"},
    33: {"color": (0x63, 0x63, 0xA6), "name": "suitcase"},
    34: {"color": (0x3B, 0xE5, 0xA1), "name": "frisbee"},
    35: {"color": (0x24, 0x1F, 0xFB), "name": "skis"},
    36: {"color": (0xB3, 0x29, 0xD1), "name": "snowboard"},
    37: {"color": (0x88, 0xB0, 0xD8), "name": "sports ball"},
    38: {"color": (0x0E, 0x06, 0x3D), "name": "kite"},
    39: {"color": (0x24, 0x83, 0x2E), "name": "baseball bat"},
    40: {"color": (0x6A, 0xB3, 0x15), "name": "baseball glove"},
    41: {"color": (0x6D, 0x8B, 0x92), "name": "skateboard"},
    42: {"color": (0xBF, 0x51, 0x40), "name": "surfboard"},
    43: {"color": (0xE0, 0x8B, 0xB3), "name": "tennis racket"},
    44: {"color": (0x22, 0xCC, 0x2E), "name": "bottle"},
    46: {"color": (0xF9, 0x36, 0x7E), "name": "wine glass"},
    47: {"color": (0x9C, 0x65, 0xA1), "name": "cup"},
    48: {"color": (0xB5, 0x79, 0x9F), "name": "fork"},
    49: {"color": (0xA1, 0xE7, 0x72), "name": "knife"},
    50: {"color": (0x15, 0x40, 0xB0), "name": "spoon"},
    51: {"color": (0x2C, 0x0E, 0x97), "name": "bowl"},
    52: {"color": (0xF9, 0xBA, 0xB5), "name": "banana"},
    53: {"color": (0xC7, 0x36, 0x6F), "name": "apple"},
    54: {"color": (0x47, 0x15, 0x4A), "name": "sandwich"},
    55: {"color": (0x65, 0x3D, 0xBA), "name": "orange"},
    56: {"color": (0x63, 0xDB, 0xC2), "name": "broccoli"},
    57: {"color": (0x77, 0xCC, 0x31), "name": "carrot"},
    58: {"color": (0x8D, 0x95, 0xB0), "name": "hot dog"},
    59: {"color": (0x81, 0xAE, 0xE0), "name": "pizza"},
    60: {"color": (0x79, 0x36, 0xEA), "name": "donut"},
    61: {"color": (0xEA, 0x97, 0x6A), "name": "cake"},
    62: {"color": (0x3B, 0x4F, 0x63), "name": "chair"},
    63: {"color": (0x18, 0x51, 0xE5), "name": "couch"},
    64: {"color": (0xAB, 0x60, 0x95), "name": "potted plant"},
    65: {"color": (0x72, 0x45, 0x72), "name": "bed"},
    67: {"color": (0x77, 0xC2, 0x31), "name": "dining table"},
    70: {"color": (0x01, 0x29, 0x09), "name": "toilet"},
    72: {"color": (0xA9, 0x81, 0x7C), "name": "tv"},
    73: {"color": (0x4F, 0x51, 0x0E), "name": "laptop"},
    74: {"color": (0xA6, 0xFB, 0x4C), "name": "mouse"},
    75: {"color": (0x60, 0x77, 0xD1), "name": "remote"},
    76: {"color": (0xA4, 0xEC, 0x01), "name": "keyboard"},
    77: {"color": (0xE5, 0x7C, 0x5E), "name": "cell phone"},
    78: {"color": (0x4A, 0x4C, 0x54), "name": "microwave"},
    79: {"color": (0xF6, 0x9C, 0x0B), "name": "oven"},
    80: {"color": (0x59, 0x1F, 0x31), "name": "toaster"},
    81: {"color": (0xD1, 0xE2, 0xE0), "name": "sink"},
    82: {"color": (0xA1, 0xA9, 0x3B), "name": "refrigerator"},
    84: {"color": (0x33, 0x88, 0xF1), "name": "book"},
    85: {"color": (0x7E, 0xA9, 0xC9), "name": "clock"},
    86: {"color": (0xF1, 0x95, 0xC2), "name": "vase"},
    87: {"color": (0xD8, 0x77, 0x79), "name": "scissors"},
    88: {"color": (0xD8, 0xF1, 0x06), "name": "teddy bear"},
    89: {"color": (0xBA, 0xF4, 0x9C), "name": "hair drier"},
    90: {"color": (0x31, 0x83, 0xD3), "name": "toothbrush"},
    92: {"color": (0x31, 0x6D, 0xEF), "name": "banner"},
    93: {"color": (0x74, 0xBF, 0xC4), "name": "blanket"},
    95: {"color": (0x95, 0xEF, 0x92), "name": "bridge"},
    100: {"color": (0xAB, 0x33, 0x92), "name": "cardboard"},
    107: {"color": (0xA1, 0x9A, 0x04), "name": "counter"},
    109: {"color": (0x92, 0x27, 0xAB), "name": "curtain"},
    112: {"color": (0x45, 0xA6, 0x97), "name": "door-stuff"},
    118: {"color": (0x60, 0xF6, 0x83), "name": "floor-wood"},
    119: {"color": (0x56, 0x27, 0x59), "name": "flower"},
    122: {"color": (0x92, 0x81, 0x51), "name": "fruit"},
    125: {"color": (0xA9, 0xC2, 0xEC), "name": "gravel"},
    128: {"color": (0x10, 0xA9, 0x97), "name": "house"},
    130: {"color": (0xD3, 0xC4, 0xB8), "name": "light"},
    133: {"color": (0x54, 0xF4, 0x04), "name": "mirror-stuff"},
    138: {"color": (0x47, 0x3D, 0x4A), "name": "net"},
    141: {"color": (0x33, 0x22, 0x45), "name": "pillow"},
    144: {"color": (0x27, 0x45, 0xBF), "name": "platform"},
    145: {"color": (0xE2, 0x1A, 0x27), "name": "playingfield"},
    147: {"color": (0x36, 0xEF, 0x3D), "name": "railroad"},
    148: {"color": (0xE5, 0x8D, 0xB0), "name": "river"},
    149: {"color": (0x72, 0xFB, 0x7E), "name": "road"},
    151: {"color": (0x59, 0xDB, 0x5B), "name": "roof"},
    154: {"color": (0x97, 0xA4, 0xE5), "name": "sand"},
    155: {"color": (0x60, 0x15, 0x7C), "name": "sea"},
    156: {"color": (0xD8, 0xC7, 0xDD), "name": "shelf"},
    159: {"color": (0x72, 0x9A, 0xEA), "name": "snow"},
    161: {"color": (0x86, 0xC9, 0x9F), "name": "stairs"},
    166: {"color": (0xBA, 0x5B, 0xA1), "name": "tent"},
    168: {"color": (0x81, 0xAB, 0x9C), "name": "towel"},
    171: {"color": (0x59, 0xEC, 0x97), "name": "wall-brick"},
    175: {"color": (0x45, 0x2C, 0xE7), "name": "wall-stone"},
    176: {"color": (0x72, 0x97, 0x24), "name": "wall-tile"},
    177: {"color": (0x90, 0x6D, 0x1A), "name": "wall-wood"},
    178: {"color": (0x9F, 0x2E, 0xEF), "name": "water-other"},
    180: {"color": (0xEC, 0x22, 0x83), "name": "window-blind"},
    181: {"color": (0x92, 0x09, 0x92), "name": "window-other"},
    184: {"color": (0xF6, 0xAB, 0x9F), "name": "tree-merged"},
    185: {"color": (0x01, 0x3B, 0xA6), "name": "fence-merged"},
    186: {"color": (0x4F, 0xE7, 0xDD), "name": "ceiling-merged"},
    187: {"color": (0x8B, 0x81, 0x88), "name": "sky-other-merged"},
    188: {"color": (0x4C, 0x40, 0x95), "name": "cabinet-merged"},
    189: {"color": (0x51, 0xAB, 0x2E), "name": "table-merged"},
    190: {"color": (0x97, 0x27, 0xB5), "name": "floor-other-merged"},
    191: {"color": (0xA1, 0x13, 0x3B), "name": "pavement-merged"},
    192: {"color": (0xD6, 0x2E, 0x3B), "name": "mountain-merged"},
    193: {"color": (0x68, 0xFE, 0x2C), "name": "grass-merged"},
    194: {"color": (0xE7, 0x4A, 0x31), "name": "dirt-merged"},
    195: {"color": (0x4F, 0x9A, 0x86), "name": "paper-merged"},
    196: {"color": (0x38, 0xEA, 0xE0), "name": "food-other-merged"},
    197: {"color": (0xEF, 0x95, 0xD6), "name": "building-other-merged"},
    198: {"color": (0x97, 0x6D, 0x2C), "name": "rock-merged"},
    199: {"color": (0x56, 0x47, 0xE2), "name": "wall-other-merged"},
    200: {"color": (0x72, 0x0B, 0xF6), "name": "rug-merged"}}

YTOVS = {
    1: {"color": (0x51, 0x46, 0x6B), "name": "person"},
    2: {"color": (0x00, 0x4A, 0x6F), "name": "giant_panda"},
    3: {"color": (0x51, 0x00, 0x51), "name": "lizard"},
    4: {"color": (0xA0, 0xAA, 0xFA), "name": "parrot"},
    5: {"color": (0x8C, 0x96, 0xE6), "name": "skateboard"},
    6: {"color": (0x80, 0x40, 0x80), "name": "sedan"},
    7: {"color": (0xE8, 0x23, 0xF4), "name": "ape"},
    8: {"color": (0x64, 0x64, 0x96), "name": "dog"},
    9: {"color": (0x46, 0x46, 0x46), "name": "snake"},
    10: {"color": (0x99, 0x99, 0xBE), "name": "monkey"},
    11: {"color": (0xB4, 0x64, 0xB4), "name": "hand"},
    12: {"color": (0xB4, 0xA5, 0xB4), "name": "rabbit"},
    13: {"color": (0x5A, 0x78, 0x96), "name": "duck"},
    14: {"color": (0x9C, 0x66, 0x66), "name": "cat"},
    15: {"color": (0x64, 0xAA, 0xFA), "name": "cow"},
    16: {"color": (0xFA, 0xDC, 0xDC), "name": "fish"},
    17: {"color": (0x00, 0xA5, 0xFF), "name": "train"},
    18: {"color": (0x99, 0x99, 0x99), "name": "horse"},
    19: {"color": (0x64, 0xDC, 0xDC), "name": "turtle"},
    20: {"color": (0x00, 0x46, 0xFF), "name": "bear"},
    21: {"color": (0xDC, 0xDC, 0xDC), "name": "motorbike"},
    22: {"color": (0x1E, 0xAA, 0xFA), "name": "giraffe"},
    23: {"color": (0x00, 0xDC, 0xDC), "name": "leopard"},
    24: {"color": (0xFA, 0xAA, 0xFA), "name": "fox"},
    25: {"color": (0x98, 0xFB, 0x98), "name": "deer"},
    26: {"color": (0x23, 0x8E, 0x6B), "name": "owl"},
    27: {"color": (0xB4, 0x82, 0x46), "name": "surfboard"},
    28: {"color": (0x3C, 0x14, 0xDC), "name": "airplane"},
    29: {"color": (0x00, 0x00, 0xFF), "name": "truck"},
    30: {"color": (0x20, 0x0B, 0x77), "name": "zebra"},
    31: {"color": (0x64, 0x3C, 0x00), "name": "tiger"},
    32: {"color": (0x8E, 0x00, 0x00), "name": "elephant"},
    33: {"color": (0x5A, 0x00, 0x00), "name": "snowboard"},
    34: {"color": (0xE6, 0x00, 0x00), "name": "boat"},
    35: {"color": (0x6E, 0x00, 0x00), "name": "shark"},
    36: {"color": (0x64, 0x50, 0x00), "name": "mouse"},
    37: {"color": (0x46, 0x00, 0x00), "name": "frog"},
    38: {"color": (0xDE, 0x21, 0x3D), "name": "eagle"},
    39: {"color": (0xCE, 0x46, 0x47), "name": "earless_seal"},
    40: {"color": (0x0C, 0x29, 0xB9), "name": "tennis_racket"}}

SUPORT_LIST = {
    "CDnet-2014": CD_net_2014,
    "YTOVS": YTOVS,
    "BDD-100k": BDD_100k,
    "COCO": COCO}


class _label_dict():
    def __init__(self, label_style):
        self.make_label_dict(label_style)

    def make_label_dict(self, label_style):
        self.label_style = label_style
        if label_style in SUPORT_LIST.keys():
            self.label_dict = SUPORT_LIST[label_style]

        else:
            _error.Custom_Variable_Error(
                loacation="label utils / visualize",
                parameters=str(label_style),
                detail="this data set label is not supported")

        self.inverse_dict = {}
        for _tmp_key in self.label_dict.keys():
            _name = self.label_dict[_tmp_key]["name"]
            self.inverse_dict[_name] = _tmp_key

    def id_check(self, ID):
        return ID in self.label_dict.keys()

    def Class_to_Id(self, name):
        _is_exist = name in self.inverse_dict.keys()
        id_num = self.inverse_dict[name] if _is_exist else -1
        return _is_exist, id_num

    def Id_to_Calss(self, id_num):
        _is_exist = id_num in self.label_dict.keys()
        class_dict = self.label_dict[id_num] if _is_exist else \
            {"color": (0x00, 0x00, 0x00), "name": "Error"}
        return _is_exist, class_dict

    def make_label_image(self, class_channel_data):
        _h, _w, _ = np.shape(class_channel_data)

        _base = np.zeros((_h, _w, 3), np.uint8)
        for _ct, _key in enumerate(self.label_dict.keys()):
            _color = self.label_dict[_key]["color"]
            _base += (_color * class_channel_data[:, :, _ct][:, :, np.newaxis]).astype(np.uint8)

        return _base


def load_success():
    print("!!! custom python module ais_utils _label load Success !!!")
