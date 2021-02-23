"""
File object
=====
    Some custom error folder for make python custom utils.

Requirement
=====
    None

File information
=====
    first edit date   : 2020/10/30
    first editer      : choi keonghun
    last edit date    : 2020/10/30
    last editer       : choi keonghun

Import module
=====
"""
ERROR_Info_Sequence = [
    "Directory",  # file directory
    "File_name",  # file name
    "Function",   # function in which the error occurred
    "Detail"
]


def make_code_info_text(**kwagrs):
    temp = "Error information\n"
    for _info in ERROR_Info_Sequence:
        temp += "\t {Info} : {data} \n".format(
            Info=_info,
            data=kwagrs[_info]
        ) if _info in kwagrs else ""
    return temp


class Not_yet_Error(Exception):
    def __init__(self, **kwagrs):
        self._data = kwagrs
        self._data["Detail"] += "This code is Not yet. m(__)m"

    def __str__(self):
        print(make_code_info_text(self._data))


class Variable_Error(Exception):
    def __init__(self, **kwagrs):
        self._data = kwagrs

    def __str__(self):
        print(make_code_info_text(self._data))


class No_model_Error(Exception):
    def __init__(self, **kwagrs):
        self._data = kwagrs
        self._data["Detail"] += "Check again your model folder."

    def __str__(self):
        print(make_code_info_text(self._data))
