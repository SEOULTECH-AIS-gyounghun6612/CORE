from enum import Enum
from typing import Any, List, Dict, Union


if __package__ == "":
    from _base import Directory, File
    from _numpy import Array_Process, Dtype, ndarray
else:
    from ._base import Directory, File
    from ._numpy import Array_Process, Dtype, ndarray


# -- DEFINE CONSTNAT -- #


# -- Mation Function -- #
class Log():
    _Annotation: Dict[str, Union[list, str, int, Dict]] = {}
    _Data: Dict[str, Union[list, Dict]] = {}

    def __init__(self, info: Dict = {}, data: Dict = {}, file_dir: str = None, file_name: str = "log.json"):

        if file_dir is None:
            self._insert(info)
            self._insert(data, self._Data)

        else:
            self._load(file_dir, file_name) if File._exist_check(Directory._divider_check(f"{file_dir}{file_name}", True)) else ...

    def _insert(self, data_block: Dict, access_point: Dict = None, is_overwrite: bool = True):
        # check parameter; save point
        if access_point is None:
            access_point = self._Annotation

        # pick data in search point
        for _key in data_block.keys():
            _key_exist = _key in access_point.keys()

            if isinstance(data_block[_key], dict):
                if not _key_exist:
                    access_point[_key] = {}  # Make new key in holder that not be exist key
                self._insert(data_block[_key], access_point[_key], is_overwrite)  # go to deep

            else:
                # add
                if _key_exist and not is_overwrite:
                    if not isinstance(access_point[_key], list):
                        access_point[_key] = [access_point[_key], ]

                    if isinstance(data_block[_key], list):
                        access_point[_key] += data_block[_key]
                    else:
                        access_point[_key].append(data_block[_key])

                # (over)write
                else:
                    access_point[_key] = data_block[_key]

    def _get(self, place: Dict, access_point: Dict = None, is_pop: bool = False):
        # check parameter; save point
        if access_point is None:
            access_point = self._Annotation

        _pick_data = {}

        # pick data in search point
        for _key in place.keys():
            if _key not in access_point.keys():
                ...

            # if key in acces point
            elif isinstance(place[_key], dict):
                _pick_data[_key] = self._get(place[_key], access_point[_key])  # go to deep

            elif isinstance(place[_key], (list, tuple)):
                _pick_data[_key] = {}
                for _sub_key in place[_key]:
                    if _sub_key in access_point[_key].keys():
                        _pick_data[_key][_sub_key] = access_point[_key][_sub_key]
            else:
                _pick_data[_key] = access_point[_key]

    def _load(self, file_dir, file_name):
        save_pakage = File._json(file_dir, file_name)
        self._insert(save_pakage["annotation"])
        self._insert(save_pakage["data"], self._Data)

    def _save(self, file_dir, file_name):
        save_pakage = {
            "annotation": self._Annotation,
            "data": self._Data}

        File._json(file_dir, file_name, save_pakage, True)


class ploter():
    ...
