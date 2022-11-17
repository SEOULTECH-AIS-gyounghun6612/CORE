"""
File object
-----
    When write down python program, be used custom function.

Requirement
-----
    None
"""
# Import module
from dataclasses import asdict, dataclass
from enum import Enum
import json
import yaml
import platform
import time

from math import log10, floor
from glob import glob
from os import path, system, getcwd, mkdir
from typing import Any, Dict, List, Union, Optional, Tuple


if __package__ == "":
    # if this file in local project
    import _error as _e

else:
    # if this file in package folder
    from . import _error as _e

# Set constant
DEBUG = False
_error = _e.Custom_error(
    module_name="ais_custom_utils_v 2.x",
    file_name="_base.py")


# -- DEFINE CONSTNAT -- #
JSON_WRITEABLE = Optional[Union[str, int, float, bool, Tuple, List, Dict]]


class OS_Style(Enum):
    OS_WINDOW = "Windows"
    OS_UBUNTU = "Linux"


# -- DEFINE CONFIG -- #


# -- Mation Function -- #
class Directory():
    _OS_THIS = platform.system()
    _Divider = "/" if _OS_THIS == OS_Style.OS_UBUNTU.value else "\\"

    @classmethod  # fix it
    def _divider_check(cls, directory: str, is_file: bool = False):
        # each os's directory divide slash fix
        if cls._Divider == "\\":
            from_dived = "/"
        else:
            from_dived = "\\"
        directory.replace(from_dived, cls._Divider)

        if not is_file:
            # if checked path is dir, check last slach exist in end of text
            return directory if directory[-len(cls._Divider):] == cls._Divider \
                else directory + cls._Divider

        return directory

    @staticmethod
    def _exist_check(directory: Optional[str]):
        return False if directory is None else path.isdir(directory)

    @classmethod
    def _devide(cls, directory: str, point: int = -1):
        _dir = cls._divider_check(directory)
        _comp = _dir.split(cls._Divider)[:-1]

        _front = ""
        _back = ""
        for _data in _comp[:point]:
            _front += _data + cls._Divider
        for _data in _comp[point:]:
            _back += _data + cls._Divider

        return _front, _back

    @classmethod
    def _relative_root(cls, just_name: bool = False) -> str:
        return cls._devide(getcwd())[-1] if just_name else cls._divider_check(getcwd())

    @classmethod
    def _make(cls, obj_dir: str, root_dir: Optional[str] = None):
        if root_dir is not None:
            # use root directory
            # root directory check
            _dir = cls._divider_check(root_dir)
            if not cls._exist_check(_dir):
                # if root directory not exist, make it
                _front, _back = cls._devide(_dir, -1)
                cls._make(_back, _front)
        else:
            # use relative root directory (= cwd)
            _dir = cls._relative_root()

        # make directory
        for _part in cls._divider_check(obj_dir).split(cls._Divider):
            _dir = cls._divider_check(_dir + _part)
            mkdir(_dir) if not cls._exist_check(_dir) else None

        return _dir

    @classmethod
    def _inside_search(cls, searched_dir: str, search_option: str = "all", name: str = "*", ext: str = "*"):
        serch_all = search_option == "all"
        _component_name = "*" if serch_all else "*" + name + "*"
        _component_ext = "" if serch_all else (ext if ext[0] == "." else "." + ext)

        search_list = sorted(glob(cls._divider_check(searched_dir) + _component_name + _component_ext))

        if search_option in ["directory", "dir"]:
            search_list = [data for data in search_list if cls._exist_check(data)]
        elif search_option in ["file", ]:
            search_list = [data for data in search_list if File._exist_check(data)]

        return sorted(search_list)

    @staticmethod  # Not yet
    def _compare():
        _error.not_yet("directory._compare")
        # compare_obj = dir_checker(compare_obj, True)
        # compare_data = compare_obj.split(SLASH)
        # base_dir = dir_checker(base_dir, True)
        # base_data = base_dir.split(SLASH)

        # tmp_dir = "." + SLASH
        # same_count = 0

        # for _tmp_folder in base_data:
        #     if _tmp_folder in compare_data:
        #         same_count += 1
        #     else:
        #         break
        # if len(base_data) - same_count:
        #     for _ct in range(len(base_data) - same_count):
        #         tmp_dir += ".." + SLASH

        # for _folder in compare_data[same_count:]:
        #     tmp_dir += _folder + SLASH

    @staticmethod
    def _del():
        _error.not_yet("directory._del")

    @staticmethod
    def _copy():
        _error.not_yet("directory._copy")

    class Server():
        @staticmethod
        def _local_connect(ip_num: str, user_id: str, password: str, root_dir: str, mount_dir: str, is_container: bool = False):
            if is_container:
                # in later make function for docker container
                _error.not_yet("not support for container system at function server.local_connect")
                pass
            else:
                _command = ""
                if Directory._OS_THIS == OS_Style.OS_WINDOW:
                    _command = f"NET USE {mount_dir}: \\\\{ip_num}\\{root_dir} {password} /user:{user_id}"

                elif Directory._OS_THIS == OS_Style.OS_UBUNTU:
                    _command = "sudo -S mount -t cifs -o uid=1000,gid=1000,"
                    _command += f"username={user_id},password={password} //{ip_num}/{root_dir} {mount_dir}"

                system(_command)
                return mount_dir + ":" if Directory._OS_THIS == OS_Style.OS_WINDOW else mount_dir

        @staticmethod
        def _unconnect(mounted_dir: str):
            if Directory._OS_THIS == OS_Style.OS_WINDOW:
                system("NET USE {}: /DELETE".format(mounted_dir))
            elif Directory._OS_THIS == OS_Style.OS_UBUNTU:
                system("fuser -ck {}".format(mounted_dir))
                system("sudo umount {}".format(mounted_dir))


class File():
    @staticmethod
    def _exist_check(file_path: Optional[str]) -> bool:
        return False if file_path is None else path.isfile(file_path)

    @classmethod
    def _file_name_from_path(cls, file_path: str, just_file_name: bool = True):
        file_path = Directory._divider_check(file_path, is_file=True)
        _file_dir, _file_name = path.split(file_path)
        return _file_name if just_file_name else [_file_dir, _file_name]

    @classmethod
    def _extension_check(cls, file_path: str, exts: List[str], is_fix: bool = False) -> Tuple[bool, str]:
        _file_dir, _file_name = cls._file_name_from_path(file_path, False)

        if _file_name == "" or _file_name.split(".")[-1] == "":  # "file_path" is dir or extension not exist in that
            return (True, f"{file_path}.{exts[0]}") if is_fix else (False, file_path)
        elif _file_name.split(".")[-1] not in exts:  # path extension not exist in exts
            _file_name = _file_name.replace(_file_name.split(".")[-1], exts[0])
            return (True, f"{_file_dir}{Directory._Divider}{_file_name}") if is_fix else (False, file_path)
        else:
            return (True, file_path)

    @classmethod
    def _json(cls, file_dir: str, file_name: str, is_save: bool = False, data_dict: Optional[Dict] = None) -> Dict:
        """
        Args:
            save_dir        :
            file_name       :
            data_dict       :
        Returns:
            return (dict)   :
        """
        # directory check
        file_dir = Directory._divider_check(file_dir)
        if not Directory._exist_check(file_dir):
            if is_save:
                # !!!WARING!!! save directory not exist
                _error.variable(
                    function_name="file.json_file",
                    variable_list=["file_dir", ],
                    AA="Entered directory '{}' not exist. In first make that".format(file_dir))
                Directory._make(file_dir)

            else:
                # !!!ERROR!!! load directory not exist
                _error.variable_stop(
                    function_name="file.json_file",
                    variable_list=["file_dir", ],
                    AA="Entered directory '{}' not exist".format(file_dir)
                )

        # file_name check
        _, file_name = File._extension_check(file_name, ["json", ], True)

        # json file process load or save
        if is_save and data_dict is not None:
            # json file save
            _file = open(file_dir + file_name, "w")
            json.dump(data_dict, _file, indent=4)
            return data_dict
        else:
            # json file load
            if cls._exist_check(file_dir + file_name):
                # json file exist
                _file = open(file_dir + file_name, "r")
                return json.load(_file)

            else:
                # !!!ERROR!!! load json file not exist
                _error.variable_stop(
                    function_name="file.json_file",
                    variable_list=["file_dir", "file_name"],
                    AA="Load file '{}' not exist".format(file_dir + file_name)
                )

    @classmethod
    def _yaml(cls, file_dir: str, file_name: str, is_save: bool = False, data_dict: Optional[Dict] = None):
        """
        Args:
            save_dir        :
            file_name       :
            data_dict       :
        Returns:
            return (dict)   :
        """
        # # directory check
        # file_dir = Directory._divider_check(file_dir)
        # if not Directory._exist_check(file_dir):
        #     if is_save:
        #         # !!!WARING!!! save directory not exist
        #         _error.variable(
        #             function_name="file.yaml_file",
        #             variable_list=["file_dir", ],
        #             AA="Entered directory '{}' not exist. In first make that".format(file_dir))
        #         Directory._make(file_dir)

        #     else:
        #         # !!!ERROR!!! load directory not exist
        #         _error.variable_stop(
        #             function_name="file.yaml_file",
        #             variable_list=["file_dir", ],
        #             AA="Entered directory '{}' not exist".format(file_dir)
        #         )

        # # file_name check
        # _, file_name = File._extension_check(file_name, ["yml", 'yaml'], True)

        # # yaml file process load or save
        # if is_save:
        #     # json file save
        #     _file = open(file_dir + file_name, "w")
        #     yaml.dump(data_dict, _file, indent=4)
        # else:
        #     # yaml file load
        #     if cls._exist_check(file_dir + file_name):
        #         # yaml file exist
        #         _file = open(file_dir + file_name, "r")
        #         return yaml.load(_file)

        #     else:
        #         # !!!ERROR!!! load yaml file not exist
        #         _error.variable_stop(
        #             function_name="file.yaml_file",
        #             variable_list=["file_dir", "file_name"],
        #             AA="Load file '{}' not exist".format(file_dir + file_name)
        #         )
        raise NotImplementedError

    @staticmethod
    def _xml(file_dir, file_name, data_dict=None, is_save=False):
        pass

    @staticmethod
    def _del():
        _error.not_yet("file._del")

    @staticmethod
    def _copy_to(dir, file):
        _error.not_yet("file._copy_to")


class Utils():
    @dataclass
    class Config():
        def _get_parameter(self) -> Dict[str, Any]:
            """

            """
            return {}

        def _convert_to_dict(self) -> Dict[str, Optional[Union[Dict, str, int, float, bool]]]:
            """
            Returned dictionary value type must be can dumped in json file
            """
            return asdict(self)

        def _restore_from_dict(self, data: Dict[str, Optional[Union[Dict, str, int, float, bool]]]):
            """

            """
            raise NotImplementedError

    @staticmethod
    def _progress_board(this_count: int, max_count: int):  # [3/25] -> [03/25]
        _string_ct = floor(log10(max_count)) + 1
        _this = f"{this_count}".rjust(_string_ct, " ")
        _max = f"{max_count}".rjust(_string_ct, " ")

        return f"{_this}/{_max}"

    @staticmethod
    def _progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', decimals: int = 1, length: int = 100, fill: str = '█'):
        """
        Call in a loop to create terminal progress bar
        Args:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        Returns:
            Empty
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end="\r")
        # Print New Line on Complete
        if iteration == total:
            print()

    class Time():
        @staticmethod
        def _stemp(source: Optional[float] = None):
            return time.time() if source is None else source

        @staticmethod
        def _apply_text_form(source: float, is_local: bool = False, text_format: str = "%Y-%m-%d-%H:%M:%S"):
            return time.strftime(text_format, time.localtime(source) if is_local else time.gmtime(source))


class Tool_For():
    class _list():
        @staticmethod
        def is_num_over_range(target: list, obj_num: Union[int, list, range]) -> bool:
            """
            Arg:\n
                target (list) : \n
                obj_num (int, list[int], range) : \n
            """
            if isinstance(obj_num, list):
                _max = max(obj_num)
            elif isinstance(obj_num, int):
                _max = obj_num
                obj_num = [obj_num, ]
            elif isinstance(obj_num, range):
                _max = max(obj_num)
            else:
                # !!!ERROR!!! wrong type entered
                _error.data_type(
                    function_name="tool.list_tool.is_num_over_range",
                    variable_list=["obj_num", ],
                    AA="Error in parameter 'obj_num'.\n \
                        'obj_dirs' has unsuitable type data"
                )
                return True

            return _max > (len(target) + 1)

        @staticmethod
        def del_obj(target: list, obj_num: Union[int, list, range]):
            """
            Arg:
                target (list) : \n
                obj_num (int, list[int], range) : \n
            """
            if isinstance(obj_num, (list, range)):
                for _ct, _num in enumerate(obj_num):
                    del target[_num - _ct]
            elif isinstance(obj_num, int):
                del target[obj_num]

        @staticmethod
        def clear_list(target: list):
            del target[:]

    # class _dict():
    #     @classmethod
    #     def dict_to_labeling_holder(cls, dictionary: Dict[str, Union[Dict, str, list]], _root: Dict, Endpoint_holder: Any = None):
    #         for _node_name in dictionary.keys():
    #             _node_value = dictionary[_node_name]

    #         if isinstance(_node_value, dict):  # Nodes exist over 2 levels below.
    #             _root[_node_name] = cls.dict_to_labeling_holder(_node_value, {})

    #         elif isinstance(_node_value, list):  # Nodes exist in 1 level below.
    #             _root[_node_name] = {}
    #             for _name in _node_value:
    #                 _root[_node_name][_name] = Endpoint_holder

    #         elif isinstance(_node_value, str):  # end point node
    #             _root[_node_name] = {}
    #             _root[_node_name][_node_value] = Endpoint_holder

    #         else:
    #             pass

    #         return _root
