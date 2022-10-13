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
    def __init__(self, info: Dict = {}, data: Dict = {}, save_dir: str = None, file_name: str = "log.json", is_resotre: bool = False):
        self.save_dir = save_dir
        self.file_name = file_name

        _diretory = _base.directory._slash_check(f"{save_dir}{file_name}", True)

        if _base.file._exist_check(_diretory) and is_resotre:
            self.load()
        else:
            self.annotation: Dict[str, Union[list, str, int]] = {}
            self.add(info)
            self.data: Dict[str, Union[list, dict]] = {}
            self.add(data, logging_option.OVERWRITE, self.data)

    def add(self, block: Dict, mode: logging_option = logging_option.ADD, save_point: Dict = None):
        if save_point is None:
            save_point = self.annotation

        for _key in block.keys():
            _flag = _key in save_point.keys() and mode.value  # if added data's key already exist in log's it, check that write mode

            if isinstance(block[_key], Dict):
                if not _flag:
                    save_point[_key] = {}
                self.add(block[_key], mode, save_point[_key])

            else:
                # add
                if _flag and mode.value:
                    if isinstance(save_point[_key], str):
                        save_point[_key] = f'{self.annotation[_key]}\n{block[_key]}'
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

    def get_annotation(self, name: str = None) -> Dict:
        return self.annotation[name] if name in self.annotation.keys() else self.annotation

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
        self.add(save_pakage["annotation"])
        self.add(save_pakage["data"], self.data)

    def save(self):
        save_pakage = {
            "annotation": self.annotation,
            "data": self.data}

        _base.file._json(self.save_dir, self.file_name, save_pakage, True)


class ploter():
    ...
