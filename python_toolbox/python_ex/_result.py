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
        self.save_dir = save_dir
        self.file_name = file_name

        _diretory = _base.directory._slash_check(f"{save_dir}/{file_name}", True)

        if _base.file._exist_check(_diretory):
            self.load()
        else:
            self.info: Dict[str, Union[list, str, int]] = {}
            self.add_info(info)
            self.data: Dict[str, Union[list, dict]] = {}
            self.add_data(self.data, data, logging_option.OVERWRITE)

    def add_info(self, info: Dict, mode: logging_option = logging_option.OVERWRITE, save_point: Dict = None):
        if save_point is None:
            save_point = self.info

        for _key in info.keys():
            _flag = _key in self.info.keys() and mode.value  # if added data's key already exist in log's it, check that write mode

            if isinstance(info[_key], Dict):
                save_point[_key] = {}
                self.add_info(info[_key], mode, save_point[_key])

            else:
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

    def get_info(self, name: str = None) -> Dict:
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

    def get_data(self, name: Union[List[str], str] = None, serch_point: Dict = None, range_st: int = 0, range_ed: int = None) -> Dict[str, list]:
        serch_point = self.data if serch_point is None else serch_point

        if name is None:
            # get all data
            return serch_point

        name_list = [name, ] if not isinstance(name, list) else name
        picked_data = {}

        for _name in name_list:
            if _name not in serch_point.keys():
                # unsuitable data name
                ...
            elif isinstance(serch_point[_name], Dict):
                # got ot deep
                picked_data[_name] = self.get_data(name_list[1:], serch_point[_name], range_st, range_ed)

                if not len(picked_data[_name]):
                    picked_data[_name] = serch_point[_name]

            else:
                picked_data[_name] = serch_point[_name][range_st: range_ed]

        return picked_data

    def load(self):
        save_pakage = _base.file._json(self.save_dir, self.file_name)
        self.add_info(save_pakage["info"])
        self.add_data(save_pakage["data"], logging_option.OVERWRITE)

    def save(self):
        save_pakage = {
            "info": self.info,
            "data": self.data}

        _base.file._json(self.save_dir, self.file_name, save_pakage, True)


class ploter():
    ...
