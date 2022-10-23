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
    ADD = True
    OVERWRITE = False


class Log():
    _Annotation: Dict[str, Union[list, str, int]] = {}
    _Data: Dict[str, Union[list, dict]] = {}

    def __init__(self, info: Dict = {}, data: Dict = {}, save_dir: str = None, file_name: str = "log.json"):
        self._Save_dir = save_dir
        self._File_name = file_name

        if _base.File._exist_check(_base.Directory._slash_check(f"{save_dir}{file_name}", True)):
            self._load()
        else:
            self._add(info)
            self._add(data, Logging_option.OVERWRITE, self._Data)

    def _add(self, block: Dict, mode: Logging_option = Logging_option.ADD, save_point: Dict = None):
        if save_point is None:
            save_point = self._Annotation

        for _key in block.keys():
            _flag = _key in save_point.keys() and mode.value  # if added data's key already exist in log's it, check that write mode

            if isinstance(block[_key], Dict):
                if not _flag:
                    save_point[_key] = {}
                self._add(block[_key], mode, save_point[_key])

            else:
                # add
                if _flag and mode.value:
                    if isinstance(save_point[_key], str):
                        save_point[_key] = f'{self._Annotation[_key]}\n{block[_key]}'
                    else:
                        # if save_point is not a list, convert that to list for add new data.
                        if not isinstance(save_point[_key], list):
                            save_point[_key] = [save_point[_key], ]

                        if not isinstance(block[_key], list):
                            save_point[_key].append(block[_key])
                        else:
                            save_point[_key] += block[_key]

                # (over)write
                else:
                    save_point[_key] = block[_key]

    def _get_annotation(self, name: str = None) -> Dict:
        return self._Annotation[name] if name in self._Annotation.keys() else self._Annotation

    def _get_data(self, name: Union[List[str], str] = None, serch_point: Dict = None, range_st: int = 0, range_ed: int = None) -> Dict[str, list]:
        serch_point = self._Data if serch_point is None else serch_point

        if name is None:
            # get all data
            return serch_point

        __name_list = [name, ] if not isinstance(name, list) else name
        picked_data = {}

        for __name in __name_list:
            if __name not in serch_point.keys():
                # unsuitable data name
                ...
            elif isinstance(serch_point[__name], Dict):
                # got ot deep
                picked_data[__name] = self._get_data(__name_list[1:], serch_point[__name], range_st, range_ed)

                if not len(picked_data[__name]):
                    picked_data[__name] = serch_point[__name]

            else:
                picked_data[__name] = serch_point[__name][range_st: range_ed]

        return picked_data

    def _load(self):
        save_pakage = _base.File._json(self._Save_dir, self._File_name)
        self._add(save_pakage["annotation"])
        self._add(save_pakage["data"], self._Data)

    def _save(self):
        save_pakage = {
            "annotation": self._Annotation,
            "data": self._Data}

        _base.File._json(self._Save_dir, self._File_name, save_pakage, True)


class ploter():
    ...
