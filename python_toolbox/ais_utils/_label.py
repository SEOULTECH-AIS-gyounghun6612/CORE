from . import _base
from . import _cv2
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


# Label template dict
class Labels():
    LABEL_DATA_LIST = {
        "BDD-100K_Object": [
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

    def __init__(self, data_name, class_name_key, id_key, color_key) -> None:
        # set data key info
        self.class_name_key = class_name_key
        self.id_key = id_key
        self.color_key = color_key

        # selecte data
        self.raw_label = self.LABEL_DATA_LIST[data_name]
        self.id_label = {}
        self.name_label = {}

        # make class name key label
        for _data in self.raw_label:
            self.name_label[_data[self.class_name_key]] = _data
            self.id_label[_data[self.id_key]] = _data

    def id_to_data(self, id):
        if len(self.raw_label) > id and id >= 0:
            return self.raw_label[id]
        else:
            return None

    def class_name_to_data(self, class_name):
        if class_name not in self.name_label.keys():
            return None
        else:
            return self.name_label[class_name]

    def class_name_to_id(self, class_name):
        _data = self.class_name_to_data(class_name)
        if _data is not None:
            return self.raw_label.index(_data)
        else:
            return -1

    def get_color(self, call):
        if isinstance(call, str):
            data = self.class_name_to_data(call)
        elif isinstance(call, int):
            data = self.id_to_data(call)

        return data[self.color_key] if data is not None else [0, 0, 0]

    def get_len(self):
        return len(self.raw_label)

    def class_map_to_seg_map(self):
        pass

    def seg_map_to_class_map(self):
        pass


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
        self.data_root = data_folder if data_folder[:-1] == "/" else data_folder + "/"

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
            label_holder = _cv2.np.zeros([holder_ch, _h, _w]).astype(_cv2.np.uint8)
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



def load_success():
    print("!!! custom python module ais_utils _label load Success !!!")
