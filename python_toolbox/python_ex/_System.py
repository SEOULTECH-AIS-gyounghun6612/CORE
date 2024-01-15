"""
File object
-----
    When write down python program, be used custom function.

Requirement
-----
    None
"""
from __future__ import annotations
from enum import Enum
from typing import Dict, List, Tuple, Literal, Type, Any
import json
import csv
# import yaml
import platform

import sys
from glob import glob
from os import path, system, getcwd, mkdir, makedirs


# -- DEFINE CONSTNAT -- #
# Data type for hint
TYPE_NUMBER = int | float

# System Constant
PYTHON_VERSION = sys.version_info


class OS():
    THIS_STYLE = platform.system()

    class Style(Enum):
        WINDOW = "Windows"
        LINUX = "Linux"

    @staticmethod
    def _Is_it_runing(like_this_os: OS.Style):
        """
        #### 제시된 OS와 프로그램이 돌아가는 OS를 비교하는 함수
        ---------------------------------------------------------------------------------------
        """
        return OS.THIS_STYLE == like_this_os


# -- Mation Function -- #
class Path():
    ABSOLUTE_HERE = getcwd()

    class Type(Enum):
        DIRECTORY = "dir"
        FILE = "file"

    @staticmethod
    def _Seperater_check(obj_path: str):
        """
        #### 경로 문자열에 존재하는 구분자 확인
        ---------------------------------------------------------------------------------------
        """
        _old_sep = "\\" if OS._Is_it_runing(OS.Style.LINUX) else "/"
        return obj_path.replace(_old_sep, path.sep)

    @staticmethod
    def _Join(obj_path: str | List[str], root_path: str | None = None):
        """
        #### 경로 문자열 연결
        ---------------------------------------------------------------------------------------
        """
        _path_comp = [] if root_path is None else [root_path]
        _path_comp += obj_path if isinstance(obj_path, list) else [obj_path]
        return path.join(*_path_comp)

    @staticmethod
    def _Devide(obj_path: str, level: int = -1):
        """
        #### 주어진 경로 문자열 분할
        ---------------------------------------------------------------------------------------
        """
        _path_comp = obj_path.split(path.sep)
        return Path._Join(_path_comp[:level]), Path._Join(_path_comp[level:])

    @staticmethod
    def _Get_here(just_name: bool = False) -> str:
        """
        #### 현재 작동중인 python script의 경로 문자열 생성
        ---------------------------------------------------------------------------------------
        """
        return Path._Devide(getcwd())[-1] if just_name else getcwd()

    @staticmethod
    def _Exist_check(obj_path: str, target: Path.Type, raise_error: bool = False):
        """
        #### 해당 경로의 존재 여부 확인
        ---------------------------------------------------------------------------------------
        #### Parameter

        """
        if path.isdir(obj_path):
            if target is Path.Type.DIRECTORY:
                return True
            elif raise_error:
                raise ValueError("this path is not a file.")
            else:
                return False
        elif path.isfile(obj_path):
            if target is Path.Type.FILE:
                return True
            elif raise_error:
                raise ValueError("this path is not a directory.")
            else:
                return False
        else:
            if raise_error:
                raise ValueError("this path is not exist.")
            else:
                return False

    @staticmethod
    def _Make_directory(obj_dir: str | List[str], root_dir: str | None = None, is_force: bool = True):
        if root_dir is not None:  # root directory check
            if not Path._Exist_check(root_dir, Path.Type.DIRECTORY) and is_force:
                _front, _back = Path._Devide(root_dir)
                Path._Make_directory(_back, _front, is_force)
            else:
                raise ValueError(f"The entered path '{root_dir}' does not exist. Check it.")

        else:  # use relative root directory (= cwd)
            root_dir = Path.ABSOLUTE_HERE

        _obj_dir = Path._Join(obj_dir, root_dir)
        mkdir(_obj_dir) if isinstance(obj_dir, str) else makedirs(_obj_dir, exist_ok=True)
        return _obj_dir

    @staticmethod
    def _Search(obj_path: str, target: Path.Type, keyword: str | None = None, ext_filter: str | List[str] | None = None):
        assert Path._Exist_check(obj_path, Path.Type.DIRECTORY)

        # make keyword
        _obj_keyword = "*" if keyword is None else f"*{keyword}*"

        # make ext list
        if isinstance(ext_filter, list):
            _ext_list = [_ext[1:] if _ext[0] == "." else _ext for _ext in ext_filter]
        elif isinstance(ext_filter, str):
            _ext_list = [ext_filter[1:] if ext_filter[0] == "." else ext_filter]
        else:
            _ext_list = [""]

        _ext_list = ext_filter if isinstance(ext_filter, list) else [ext_filter]
        _searched_list = []

        for _ext in _ext_list:
            _searched_list += sorted(glob(Path._Join(_obj_keyword if _ext == "" else f"{_obj_keyword}.{_ext}", obj_path)))

        return _searched_list


class File():
    class Basement():
        @classmethod
        def _Path_check(cls, file_name: str, file_dir: str, ext: str | None = None, raise_error: bool = False):
            # file root path check
            _file_dir = Path._Seperater_check(file_dir)
            if not Path._Exist_check(_file_dir, Path.Type.DIRECTORY):
                ValueError(f"Data save Directrory : {_file_dir} is NOT EXIST.\n check it")

            # make the file name
            if ext is not None:
                _file_name = f"{file_name if file_name.find('.') == -1 else file_name.split('.')[0]}.{ext}"
            else:
                _file_name = file_name

            # make file path
            _file_path = Path._Join(_file_name, _file_dir)
            _file_exist = Path._Exist_check(_file_path, Path.Type.FILE)

            if raise_error:  # file exist check for read stream
                assert _file_exist, f"File '{_file_name}' not exist in {_file_dir}. please check it"

            return _file_exist, _file_path

    class Json(Basement):
        KEYABLE = TYPE_NUMBER | bool | str
        VALUEABLE = KEYABLE | Tuple | List | Dict | None
        WRITEABLE = Dict[KEYABLE, VALUEABLE]

        @staticmethod
        def _Read(file_name: str, file_dir: str) -> Dict:
            # make file path
            _is_exist, _file = File.Json._Path_check(file_name, file_dir, "json")

            # read the file
            if _is_exist:
                with open(_file, "r") as _file:
                    _load_data = json.load(_file)
                return _load_data
            else:
                print(f"file {file_name} is not exist in {file_dir}")
                return {}

        @staticmethod
        def _Write(file_name: str, file_dir: str, data: WRITEABLE):
            # make file path
            _, _file = File.Json._Path_check(file_name, file_dir, "json")

            # dump to file
            with open(_file, "w") as _file:
                try:
                    json.dump(data, _file, indent="\t")
                except:
                    return False
            return True

    class CSV(Basement):
        @staticmethod
        def _Read(file_name: str, file_dir: str, fotmating: List[Type], delimiter: str = "|", encoding="UTF-8") -> List[Dict[str, Any]]:
            # make file path
            _is_exist, _file = File.Json._Path_check(file_name, file_dir, "csv")

            if _is_exist:
                # read the file
                with open(_file, "r", encoding=encoding) as file:
                    _read_data = [_data for _data in csv.DictReader(file, delimiter=delimiter)]
                return _read_data
            else:
                print(f"file {file_name} is not exist in {file_dir}")
                return []

        @staticmethod
        def _Write(file_name: str, file_dir: str, data: List[Dict], feildnames: List[str], delimiter: str = "|", mode: Literal['a', 'w'] = 'w', encoding="UTF-8"):
            # make file path
            _is_exist, _file = File.Json._Path_check(file_name, file_dir, "csv")

            # dump to file
            with open(_file, mode if not _is_exist else "w", encoding=encoding, newline="") as _file:
                try:
                    _dict_writer = csv.DictWriter(_file, fieldnames=feildnames, delimiter=delimiter)
                    _dict_writer.writeheader()
                    _dict_writer.writerows(data)
                except:
                    return False
            return True


class Server():
    IS_WINDOW = OS._Is_it_runing(OS.Style.WINDOW)

    class Connection_Porcess(Enum):
        CIFS = "cifs"

    def __init__(self, process: Connection_Porcess = Connection_Porcess.CIFS) -> None:
        self.process = process

    def _connect_to_Linux(self, host_name: str, mount_dir: str, mount_point: str, user_id: int, group_id: int, dir_mode: int, file_mode: int, credent_path: str):
        _command = f"sudo -S mount -t {self.process.value}"
        _command += f" -o uid={user_id},gid={group_id},dir_mode={dir_mode%1000:0>4d},dile_mode={file_mode%1000:0>4d},credentials={credent_path}"
        _command += f" //{host_name}/{mount_dir} {mount_point}"

        system(_command)

        return mount_point

    def _connect_to_Window(self, host_name: str, mount_dir: str, mount_point: str, user_name: str):
        system(f"NET USE {mount_point}: \\\\{host_name}\\{mount_dir} /user:{user_name}")
        return mount_point + ":"

    def _disconnect(self, mounted_dir: str):
        if self.IS_WINDOW:
            system(f"NET USE {mounted_dir}: /DELETE")
        else:
            system(f"fuser -ck {mounted_dir}")
            system(f"sudo umount {mounted_dir}")
