"""
File object
-----
    When write down python program, be used custom function.

Requirement
-----
    None
"""
# Import module
from enum import Enum
import json
# import yaml
import platform

from glob import glob
from os import path, system, getcwd, mkdir
from typing import Dict, List, Tuple


# -- DEFINE CONSTNAT -- #
# Data type for hint
TYPE_NUMBER = int | float


class OS_Style(Enum):
    OS_WINDOW = "Windows"
    OS_UBUNTU = "Linux"


# -- Mation Function -- #
class Directory():
    _OS_THIS = platform.system()
    _Divider = "/" if _OS_THIS == OS_Style.OS_UBUNTU.value else "\\"

    @classmethod  # fix it
    def _Divider_check(cls, directory: str, is_file: bool = False):
        """{description}

        {information of function}

        Parameters
        --------------------
        {arg}
            {description}
        """
        # each os's directory divide slash fix
        if cls._Divider == "\\":
            _from_dived = "/"
        else:
            _from_dived = "\\"
        _dir = directory.replace(_from_dived, cls._Divider)

        if not is_file:
            # if checked path is dir, check last slach exist in end of text
            return _dir if _dir[-len(cls._Divider):] == cls._Divider else f"{_dir}{cls._Divider}"

        return _dir

    @staticmethod
    def _Exist_check(directory: str):
        return path.isdir(directory)

    @classmethod
    def _Devide(cls, directory: str, level: int = -1):
        _dir = cls._Divider_check(directory)
        _comp = _dir.split(cls._Divider)[:-1]

        _front = cls._Divider_check(cls._Divider.join(_comp[:level]))
        _back = cls._Divider_check(cls._Divider.join(_comp[level:]))

        return _front, _back

    @staticmethod
    def _Relative_root(just_name: bool = False) -> str:
        return Directory._Devide(getcwd())[-1] if just_name else Directory._Divider_check(getcwd())

    @staticmethod
    def _Make(obj_dir: str, root_dir: str | None = None):
        if root_dir is not None:
            # use root directory
            # root directory check
            _dir = Directory._Divider_check(root_dir)
            if not Directory._Exist_check(_dir):
                # if root directory not exist, make it
                _front, _back = Directory._Devide(_dir, -1)
                Directory._Make(_back, _front)
        else:
            # use relative root directory (= cwd)
            _dir = Directory._Relative_root()

        # make directory
        for _part in Directory._Divider_check(obj_dir).split(Directory._Divider):
            _dir = Directory._Divider_check(f"{_dir}{_part}")
            mkdir(_dir) if not Directory._Exist_check(_dir) else None

        return _dir

    @staticmethod
    def _Search(object_dir: str, keyword: str | None = None, ext_filter: str | List[str] = "Directory"):
        # Make fillter string
        _dir = Directory._Divider_check(object_dir)
        _name = "*" if keyword is None else f"*{keyword}*"
        _ext_info = "" if ext_filter == "Directory" else [f".{_ext}" for _ext in ext_filter]

        # return directory list or all data
        if isinstance(_ext_info, str):
            _search_list = sorted(glob(f"{_dir}{_name}{_ext_info}"))
            return _search_list if ext_filter is None else [data for data in _search_list if Directory._Exist_check(data)]

        # retrun file list
        else:
            _search_list = []
            for _ext in _ext_info:
                _search_list += sorted(glob(f"{_dir}{_name}{_ext}"))

            return _search_list

    class Server():
        @staticmethod
        def _Connect(ip_num: str, user_id: str, password: str, root_dir: str, mount_dir: str, is_container: bool = False):
            if is_container:
                # in later make function for docker container
                raise NotImplementedError
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
        def _Deconnect(mounted_dir: str):
            if Directory._OS_THIS == OS_Style.OS_WINDOW:
                system("NET USE {}: /DELETE".format(mounted_dir))
            elif Directory._OS_THIS == OS_Style.OS_UBUNTU:
                system("fuser -ck {}".format(mounted_dir))
                system("sudo umount {}".format(mounted_dir))

    # @staticmethod  # Not yet
    # def _compare():
    #     raise NotImplementedError
    #     # compare_obj = dir_checker(compare_obj, True)
    #     # compare_data = compare_obj.split(SLASH)
    #     # base_dir = dir_checker(base_dir, True)
    #     # base_data = base_dir.split(SLASH)

    #     # tmp_dir = "." + SLASH
    #     # same_count = 0

    #     # for _tmp_folder in base_data:
    #     #     if _tmp_folder in compare_data:
    #     #         same_count += 1
    #     #     else:
    #     #         break
    #     # if len(base_data) - same_count:
    #     #     for _ct in range(len(base_data) - same_count):
    #     #         tmp_dir += ".." + SLASH

    #     # for _folder in compare_data[same_count:]:
    #     #     tmp_dir += _folder + SLASH

    # @staticmethod
    # def _del():
    #     raise NotImplementedError

    # @staticmethod
    # def _copy():
    #     raise NotImplementedError


TYPE_JSON_KEYABLE = TYPE_NUMBER | bool | str
TYPE_JSON_VALUEABLE = TYPE_JSON_KEYABLE | Tuple | List | Dict | None
TYPE_JSON_WRITEABLE = Dict[TYPE_JSON_KEYABLE, TYPE_JSON_VALUEABLE]


class File():
    @staticmethod
    def _Exist_check(file_path: str | None) -> bool:
        return False if file_path is None else path.isfile(file_path)

    @staticmethod
    def _Extrect_file_name(file_path: str, just_file_name: bool = True):
        _file_dir, _file_name = path.split(file_path)
        return _file_name if just_file_name else [Directory._Divider_check(_file_dir), _file_name]

    @staticmethod
    def _Extension_check(file_path: str, ext_filter: List[str], is_fix: bool = False) -> Tuple[bool, str]:
        # _file_ext = "npy" if isinstance(array, ndarray) else "npz"

        # if file_name.find(".") == -1:
        #     _file_name = f"{file_name}.{_file_ext}"
        # elif file_name.split(".")[-1] is _file_ext:
        #     _file_name = file_name
        # else:
        #     _file_name = file_name.replace(file_name.split(".")[-1], _file_ext)

        _file_dir, _file_name = File._Extrect_file_name(file_path, False)

        if _file_name == "" or _file_name.split(".")[-1] == "":  # "file_path" is dir or extension not exist in that
            return (True, f"{file_path}.{ext_filter[0]}") if is_fix else (False, file_path)
        elif _file_name.split(".")[-1] not in ext_filter:  # path extension not exist in exts
            _file_name = _file_name.replace(_file_name.split(".")[-1], ext_filter[0])
            return (True, f"{_file_dir}{Directory._Divider}{_file_name}") if is_fix else (False, file_path)
        else:
            return (True, file_path)

    @staticmethod
    def _Json(file_dir: str, file_name: str, is_save: bool = False, data_dict: Dict | None = None) -> Dict:
        # directory check
        _file_dir = Directory._Divider_check(file_dir)
        assert Directory._Exist_check(_file_dir), f"Data save Directrory : {_file_dir} is NOT EXIST.\n check it"
        # file_name check
        _, file_name = File._Extension_check(file_name, ["json", ], True)

        _file = f"{_file_dir}{file_name}"

        # Dictionary data save to json file
        if is_save and data_dict is not None:
            with open(_file, "w") as _file:
                json.dump(data_dict, _file, indent="\t")
            return data_dict

        # Dictionary data load from json file
        else:
            assert File._Exist_check(_file), f"File '{_file}' not exist"
            _file = open(_file, "r")
            return json.load(_file)

    # @staticmethod
    # def _yaml(file_dir: str, file_name: str, is_save: bool = False, data_dict: Optional[Dict] = None):
    #     """
    #     Args:
    #         save_dir        :
    #         file_name       :
    #         data_dict       :
    #     Returns:
    #         return (dict)   :
    #     """
    #     raise NotImplementedError

    # @staticmethod
    # def _xml(file_dir, file_name, data_dict=None, is_save=False):
    #     pass

    # @staticmethod
    # def _del():
    #     raise NotImplementedError

    # @staticmethod
    # def _copy_to(dir, file):
    #     raise NotImplementedError
