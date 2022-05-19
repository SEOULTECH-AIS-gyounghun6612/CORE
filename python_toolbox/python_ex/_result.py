from enum import Enum
from typing import Any, List, Dict, Union


if __package__ == "":
    import _base
    import _error as _e
else:
    from . import _error as _e
    from . import _base


class logging_option(Enum):
    ADD = True
    OVERWRITE = False


class log():
    def __init__(self, info: Dict = {}, data: Dict = {}, save_dir: str = None, file_name: str = "log.json"):

        if save_dir is not None:
            self.load(save_dir, file_name)
        else:
            self.info: Dict[str, Union[list, str, int]] = {}
            self.add_info(info)
            self.data: Dict[str, Union[list, dict]] = {}
            self.add_data(self.data, data, logging_option.OVERWRITE)

    def add_info(self, info: Dict, mode: logging_option = logging_option.OVERWRITE):
        for _key in info.keys():
            _flag = _key in self.info.keys() and mode.value  # if added data's key already exist in log's it, check that write mode
            if _flag:
                if isinstance(self.info[_key], str):
                    self.info[_key] = f'{self.info[_key]}\n{info[_key]}'
                else:
                    # Due to "ADD" write option, data type must be set "list"
                    if not isinstance(self.info[_key], list):
                        self.info[_key] = [self.info[_key], ]

                    self.info[_key].append(info[_key])

            else:
                self.info[_key] = info[_key]  # False -> (over)write

    def get_info(self, name: str = None):
        return self.info[name] if name in self.info.keys() else self.info

    def add_data(self, data: Dict, mode: logging_option = logging_option.ADD, save_point: Dict = None):
        if save_point is None:
            save_point = self.data

        for _key in data.keys():
            _flag = _key in save_point.keys() and mode.value  # if added data's key already exist in log's it, check that write mode

            if isinstance(data[_key], Dict):
                save_point[_key] = {}
                self.add_data(data[_key], mode, save_point[_key])

            else:
                _contents = data[_key] if isinstance(data[_key], list) else [data[_key], ]
                if _flag:
                    # Due to "ADD" write option, data type must be set "list"
                    save_point[_key] += _contents
                else:
                    save_point[_key] = _contents  # False -> (over)write

    def get_data(self, name: Union[List[str], str] = None, serch_point: Dict = None):
        serch_point = self.data if serch_point is None else serch_point
        name = [name, ] if not isinstance(name, list) else name
        serch_name = name[0]

        if name is None:
            # get all data
            return serch_point
        elif serch_name not in serch_point.keys():
            # unsuitable data name
            return None
        elif isinstance(serch_point[serch_name], Dict) and len(name[2:]):
            # got ot deep
            return self.get_data(name[1:], serch_point[serch_name])
        else:
            return serch_point[serch_name]

    def load(self, save_dir: str, file_name: str = "log.json"):
        save_pakage = _base.file._json(save_dir, file_name)
        self.add_info(save_pakage["info"])
        self.add_data(save_pakage["data"], logging_option.OVERWRITE)

    def save(self, save_dir: str, file_name: str = "log.json"):
        save_pakage = {
            "info": self.info,
            "data": self.data}

        _base.file._json(save_dir, file_name, save_pakage, True)


class ploter():
    ...
