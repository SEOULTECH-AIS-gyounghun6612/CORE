from enum import auto
from python_ex.system import String


class Mode(String.String_Enum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()
