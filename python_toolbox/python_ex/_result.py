from enum import Enum
from typing import Any, List, Dict, Union


if __package__ == "":
    import _base
    import _error as _e
else:
    from . import _error as _e
    from . import _base


# -- DEFINE CONSTNAT -- #
class Logging_option(Enum):
    ADD = 0
    READDING = 1
    DELETE = 2


class Log():
    _Annotation: Dict[str, Union[list, str, int]] = {}
    _Data: Dict[str, Union[list, dict]] = {}

    def __init__(self, info: Dict = {}, data: Dict = {}, save_dir: str = None, file_name: str = "log.json"):
        self._Save_dir = save_dir
        self._File_name = file_name

        if _base.File._exist_check(_base.Directory._slash_check(f"{save_dir}{file_name}", True)):
            self._load()
        else:
            self._insert(info)
            self._insert(data, self._Data)

    def _insert(self, data_block: Dict, access_point: Dict = None, is_overwrite: bool = True):
        # check parameter; save point
        if access_point is None:
            access_point = self._Annotation

        # pick data in search point
        for __key in data_block.keys():
            __key_exist = __key in access_point.keys()

            if isinstance(data_block[__key], dict):
                if not __key_exist:
                    access_point[__key] = {}  # Make new key in holder that not be exist key
                self._insert(data_block[__key], access_point[__key], is_overwrite)  # go to deep

            else:
                # add
                if __key_exist and not is_overwrite:
                    if isinstance(access_point[__key], str):
                        access_point[__key] = f'{access_point[__key]}\n{data_block[__key]}'
                    elif isinstance(access_point[__key], list):
                        access_point[__key].append(data_block[__key])
                    else:
                        access_point[__key] = [access_point[__key], ]
                        access_point[__key].append(data_block[__key])
                # (over)write
                else:
                    access_point[__key] = data_block[__key]

    def _get(self, holder: Dict, access_point: Dict = None, is_pop: bool = True):
        # check parameter; save point
        if access_point is None:
            access_point = self._Annotation

        # pick data in search point
        for __key in holder.keys():
            __key_exist = __key in access_point.keys()

            if __key_exist:
                if isinstance(holder[__key], dict):
                    self._get(holder[__key], access_point[__key], is_pop)  # go to deep
                elif isinstance(holder[__key], list):
                    holder[__key] = access_point[__key][holder[__key][0]: holder[__key][1]]
                    if is_pop:
                        del access_point[__key][holder[__key][0]: holder[__key][1]]
                    ...
                else:
                    holder[__key] = access_point[__key]
                    if is_pop:
                        del access_point[__key]

    def _load(self):
        save_pakage = _base.File._json(self._Save_dir, self._File_name)
        self._insert(save_pakage["annotation"])
        self._insert(save_pakage["data"], self._Data)

    def _save(self):
        save_pakage = {
            "annotation": self._Annotation,
            "data": self._Data}

        _base.File._json(self._Save_dir, self._File_name, save_pakage, True)


class ploter():
    ...
