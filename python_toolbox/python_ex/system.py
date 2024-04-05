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
from typing import Dict, List, Tuple, Literal, Any, Union
import json
import csv
# import yaml
import platform

import sys
from glob import glob
from os import path, system, getcwd, makedirs


# -- DEFINE CONSTNAT -- #
# Data type for hint
TYPE_NUMBER = Union[int, float]

# System Constant
PYTHON_VERSION = sys.version_info


class OperatingSystem():
    THIS_STYLE = platform.system()

    class Name(Enum):
        WINDOW = "Windows"
        LINUX = "Linux"

    @staticmethod
    def Is_it_runing(like_this_os: OperatingSystem.Name):
        """
        #### 제시된 OS와 프로그램이 돌아가는 OS를 비교하는 함수
        ----------------------------------------------------------------
        """
        return OperatingSystem.THIS_STYLE == like_this_os


# -- Mation Function -- #
class Path():
    WORK_SPACE = getcwd()

    class Type(Enum):
        DIRECTORY = "dir"
        FILE = "file"

    @staticmethod
    def Seperater_check(obj_path: str):
        """
        #### 경로 문자열에 존재하는 구분자 확인
        ----------------------------------------------------------------
        """
        _is_linux = OperatingSystem.Is_it_runing(OperatingSystem.Name.LINUX)
        _old_sep = "\\" if _is_linux else "/"
        return obj_path.replace(_old_sep, path.sep)

    @staticmethod
    def Join(obj_path: str | List[str], root_path: str | None = None):
        """
        #### 경로 문자열 연결
        ----------------------------------------------------------------
        """
        _path_comp = [] if root_path is None else [root_path]
        _path_comp += obj_path if isinstance(obj_path, list) else [obj_path]
        return path.join(*_path_comp)

    @staticmethod
    def Devide(obj_path: str, level: int = -1):
        """
        #### 주어진 경로 문자열 분할
        ----------------------------------------------------------------
        """
        _path_comp = obj_path.split(path.sep)
        return Path.Join(_path_comp[:level]), Path.Join(_path_comp[level:])

    @staticmethod
    def Exist_check(obj_path: str, target: Path.Type | None = None):
        """
        #### 해당 경로의 존재 여부 확인
        ----------------------------------------------------------------
        #### Parameter

        """
        if target is Path.Type.DIRECTORY:
            return path.isdir(obj_path)
        elif target is Path.Type.FILE:
            return path.isfile(obj_path)
        else:
            return path.isdir(obj_path) or path.isfile(obj_path)

    @staticmethod
    def Make_directory(
        obj_dir: str | List[str],
        root_dir: str | None = None,
        is_force: bool = False
    ):
        if root_dir is not None:  # root directory check
            _exist = Path.Exist_check(root_dir, Path.Type.DIRECTORY)
            if _exist:
                pass
            elif is_force:
                _front, _back = Path.Devide(root_dir)
                Path.Make_directory(_back, _front)
            else:
                raise ValueError(
                    "\n".join((
                        f"!!! Root directory {root_dir} is NOT EXIST !!!",
                        f"{obj_dir} can't make in {root_dir}")
                    )
                )

        else:  # use relative root directory (= cwd)
            root_dir = Path.WORK_SPACE

        _obj_dir = Path.Join(obj_dir, root_dir)
        makedirs(_obj_dir, exist_ok=True)
        return _obj_dir

    @staticmethod
    def Get_file_directory(file_path: str | None):
        if file_path is None:
            _file_path = path.abspath(__file__)
            _exist = True
        else:
            _file_path = file_path
            _exist = Path.Exist_check(file_path, Path.Type.FILE)

        return _exist, Path.Devide(_file_path)

    @staticmethod
    def Search(
        obj_path: str,
        target: Path.Type | None = None,
        keyword: str | None = None,
        ext_filter: str | List[str] | None = None
    ) -> List[str]:
        assert Path.Exist_check(obj_path, Path.Type.DIRECTORY)

        # make keyword
        _obj_keyword = "*" if keyword is None else f"*{keyword}*"

        # make ext list
        if isinstance(ext_filter, list):
            _ext_list = [
                _ext[1:] if _ext[0] == "." else _ext for _ext in ext_filter
            ]
        elif isinstance(ext_filter, str):
            _ext_list = [
                ext_filter[1:] if ext_filter[0] == "." else ext_filter
            ]
        else:
            _ext_list = [""]

        _searched_list = []

        for _ext in _ext_list:
            _list = sorted(glob(
                Path.Join(
                    _obj_keyword if _ext == "" else f"{_obj_keyword}.{_ext}",
                    obj_path
                ))
            )
            _searched_list += [
                _file for _file in _list if Path.Exist_check(_file, target)
            ]
        return _searched_list


class File():
    class Json():
        KEYABLE = Union[TYPE_NUMBER, bool, str]
        VALUEABLE = Union[KEYABLE, Tuple, List, Dict, None]
        WRITEABLE = Dict[KEYABLE, VALUEABLE]

        @staticmethod
        def Read(
            file_name: str,
            file_dir: str,
            encoding_type: str = "UTF-8"
        ) -> Dict:
            # make file path
            if "." in file_name:
                _ext = file_name.split(".")[-1]
                if _ext != "json":
                    _file_name = file_name.replace(_ext, "json")
                else:
                    _file_name = file_name
            else:
                _file_name = f"{file_name}.json"

            _file = Path.Join(_file_name, file_dir)
            _is_exist = Path.Exist_check(_file, Path.Type.FILE)

            # read the file
            if _is_exist:
                with open(_file, "r", encoding=encoding_type) as _file:
                    _load_data = json.load(_file)
                return _load_data
            else:
                print(f"file {file_name} is not exist in {file_dir}")
                return {}

        @staticmethod
        def Write(
            file_name: str,
            file_dir: str,
            data: WRITEABLE,
            encoding_type: str = "UTF-8"
        ):
            # make file path
            if "." in file_name:
                _ext = file_name.split(".")[-1]
                if _ext != "json":
                    _file_name = file_name.replace(_ext, "json")
                else:
                    _file_name = file_name
            else:
                _file_name = f"{file_name}.json"

            _file = Path.Join(_file_name, file_dir)

            # dump to file
            with open(_file, "w", encoding=encoding_type) as _file:
                try:
                    json.dump(data, _file, indent="\t")
                except TypeError:
                    return False
            return True

    class CSV():
        @staticmethod
        def _Read(
            file_name: str,
            file_dir: str,
            delimiter: str = "|",
            encoding_type="UTF-8"
        ) -> List[Dict[str, Any]]:
            # make file path
            _file = Path.Join([file_name, "csv"], file_dir)
            _is_exist = Path.Exist_check(_file, Path.Type.FILE)

            if _is_exist:
                # read the file
                with open(_file, "r", encoding=encoding_type) as file:
                    _raw_data = csv.DictReader(file, delimiter=delimiter)
                    _read_data = [
                        dict((
                                _key.replace(" ", ""),
                                _value.replace(" ", "")
                            ) for _key, _value in _line_dict.items()
                        ) for _line_dict in _raw_data
                    ]
                return _read_data
            else:
                print(f"file {file_name} is not exist in {file_dir}")
                return []

        @staticmethod
        def _Write(
            file_name: str,
            file_dir: str,
            data: List[Dict],
            feildnames: List[str],
            delimiter: str = "|",
            mode: Literal['a', 'w'] = 'w',
            encoding_type="UTF-8"
        ):
            # make file path
            _file = Path.Join([file_name, "csv"], file_dir)
            _is_exist = Path.Exist_check(_file, Path.Type.FILE)

            # dump to file
            with open(
                _file,
                mode if not _is_exist else "w",
                encoding=encoding_type,
                newline=""
            ) as _file:
                try:
                    _dict_writer = csv.DictWriter(
                        _file, fieldnames=feildnames, delimiter=delimiter)
                    _dict_writer.writeheader()
                    _dict_writer.writerows(data)
                except TypeError:
                    return False
            return True


class Server():
    IS_WINDOW = OperatingSystem.Is_it_runing(OperatingSystem.Name.WINDOW)

    class Connection_Porcess(Enum):
        CIFS = "cifs"

    def __init__(
        self,
        process: Connection_Porcess = Connection_Porcess.CIFS
    ) -> None:
        self.process = process

    def _connect_to_Linux(
        self,
        host_name: str,
        mount_dir: str,
        mount_point: str,
        user_id: int,
        group_id: int,
        dir_mode: int,
        file_mode: int,
        credent_path: str
    ):
        _command = path.join(
            f"sudo -S mount -t {self.process.value}",
            " -o ",
            ",".join((
                f"uid={user_id}",
                f"gid={group_id}",
                f"dir_mode={dir_mode%1000:0>4d}",
                f"dile_mode={file_mode%1000:0>4d}",
                f"credentials={credent_path}",
            )),
            f" //{host_name}/{mount_dir} {mount_point}"
        )
        system(_command)

        return mount_point

    def _connect_to_Window(
        self,
        host_name: str,
        mount_dir: str,
        mount_point: str,
        user_name: str
    ):
        raise NotImplementedError

    def _disconnect(self, mounted_dir: str):
        if self.IS_WINDOW:
            system(f"NET USE {mounted_dir}: /DELETE")
        else:
            system(f"fuser -ck {mounted_dir}")
            system(f"sudo umount {mounted_dir}")
