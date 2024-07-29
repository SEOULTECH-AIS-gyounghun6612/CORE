""" ### Frequently used features for handling system-related tasks

------------------------------------------------------------------------
### Requirement
    None

### Structure
    OperatingSystem: ...
    Path: ...
    File: ...

"""
from __future__ import annotations
from enum import Enum
from typing import Tuple, Literal, Any
import json
import csv
# import yaml
import platform

import sys
from glob import glob
from os import path, system, getcwd, makedirs

from dataclasses import dataclass

from math import log10, floor

from datetime import datetime, date, time, timezone
from dateutil.relativedelta import relativedelta


# -- DEFINE CONSTNAT -- #
# Data type for hint
TYPE_NUMBER = int | float

# System Constant
PYTHON_VERSION = sys.version_info


class OperatingSystem():
    """### Frequently used features for handling OS-related tasks
    --------------------------------------------------------------------
    ### Requirement
        None

    ### Structure
        ...

    """

    THIS_STYLE = platform.system()

    class Name(Enum):
        """
        각 OS 별 처리 문자열
        ----------------------------------------------------------------
        """
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
    """
    경로와 관련하여 자주 사용하는 기능 모음
    --------------------------------------------------------------------
    """
    WORK_SPACE = getcwd()

    class Type(Enum):
        DIR = "dir"
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
    def Join(obj_path: str | list[str], root_path: str | None = None):
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
        if target is Path.Type.DIR:
            return path.isdir(obj_path)
        elif target is Path.Type.FILE:
            return path.isfile(obj_path)
        else:
            return path.isdir(obj_path) or path.isfile(obj_path)

    @staticmethod
    def Make_directory(
        obj_dir: str | list[str],
        root_dir: str | None = None,
        is_force: bool = False
    ):
        if root_dir is not None:  # root directory check
            _exist = Path.Exist_check(root_dir, Path.Type.DIR)
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
    def Get_file_directory(file_path: str):
        _file_path = file_path
        _exist = Path.Exist_check(file_path, Path.Type.FILE)

        return _exist, *Path.Devide(_file_path)

    @staticmethod
    def Search(
        obj_path: str,
        target: Path.Type | None = None,
        keyword: str | None = None,
        ext_filter: str | list[str] | None = None
    ) -> list[str]:
        assert Path.Exist_check(obj_path, Path.Type.DIR)

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


class String():
    @staticmethod
    def Count_auto_aligning(this_count: int, max_count: int):
        _string_ct = floor(log10(max_count)) + 1
        _this = f"{this_count}".rjust(_string_ct, "0")

        return f"{_this}/{max_count}"

    @staticmethod
    def Str_adjust(
        text: str,
        max_length: int,
        fill: str = " ",
        align: Literal["l", "c", "r"] = "r"
    ) -> Tuple[int, str]:
        for _str in text:
            max_length -= 1 if _str.encode().isalpha() ^ _str.isalpha() else 0

        if max_length < 0:
            return -max_length, text
        if align == "l":
            return 0, text.ljust(max_length, fill)
        if align == "c":
            return 0, text.center(max_length, fill)
        return 0, text.rjust(max_length, fill)

    @staticmethod
    def Str_adjust_with_key(
        key: str,
        value: str,
        max_length: int,
        fill: str = " ",
        align: Literal["l", "c", "r"] = "r"
    ):
        _k_l, _k = String.Str_adjust(key, max_length, fill, align)
        _v_l, _v = String.Str_adjust(value, max_length, fill, align)

        if not (_k_l or _v_l):
            return (_k, _v)

        _max_length = max_length - min(_k_l, _v_l)

        return (
            String.Str_adjust(key, _max_length, fill, align)[-1],
            String.Str_adjust(value, _max_length, fill, align)[-1]
        )

    @staticmethod
    def Convert_from_str(str_data: str):
        if "," in str_data:
            return [
                String.Convert_from_str(
                    _d
                ) for _d in str_data[1:-1].split(",")
            ]
        if str_data[0] != "-" and ("-" in str_data or ":" in str_data):
            is_timezone = "+" in str_data or "-" in str_data
            return Time.Make_time_from(
                str_data,
                use_timezone=is_timezone
            )
        if ";" in str_data:
            return Time.Relative(
                *[int(_v) for _v in str_data.split(";")]
            )
        try:
            if "." in str_data:
                return float(str_data)
            return int(str_data)
        except ValueError:
            return str_data

    @staticmethod
    def Convert_to_str(
        data: list | datetime | Time.Relative | float | int | str
    ) -> str:
        if isinstance(data, list):
            return ",".join(
                [String.Convert_to_str(_data) for _data in data]
            )
        if isinstance(data, (datetime, date)):
            return Time.Make_text_from(data)
        if isinstance(data, Time.Relative):
            return ";".join(list(data))
        return data if isinstance(data, str) else str(data)


class Time():
    @staticmethod
    def Stamp(set_timezone: timezone | None = None):
        """
        현재 시간 정보를 생성하는 함수

        ---------------------------------------------------------------------------------------
        ### Parameters
        - start_time : 시간 측정을 위한 시작점

        ### Return
        - this_time : start_time 이후 흐른 시간 (start_time이 없는 경우 현재 시간)
        """
        return datetime.now(set_timezone)

    @staticmethod
    def Get_term(
        standard_time: datetime,
        to_str: bool = True,
        set_timezone: timezone | None = None
    ):
        _term = Time.Stamp(set_timezone) - standard_time
        return str(_term) if to_str else _term

    @staticmethod
    def Make_text_from(
        time_source: datetime | date | time,
        date_format: str | None = None
    ):
        if date_format is None:
            return time_source.isoformat()
        else:
            return time_source.strftime(date_format)

    @staticmethod
    def Make_time_from(
        text_source: str,
        date_format: str | None = None,
        use_timezone: bool = False
    ):
        if date_format is not None:
            _date_format = date_format
        else:
            _date_format = "%Y-%m-%dT%H:%M:%S"
            _date_format += "%z" if use_timezone else ""

        _datetime = datetime.strptime(text_source, _date_format)
        return _datetime

    @dataclass
    class Relative():
        years: int = 0
        months: int = 0
        weeks: int = 0
        days: int = 0
        hours: int = 0
        minutes: int = 0
        seconds: int = 0
        microsecond: int = 0

        def __post_init__(self):
            self.__position__ = 0

        def Delta_to_order_dict(self, time_delta: relativedelta):
            for _key in self.__dict__:
                self.__dict__[_key] = time_delta.__dict__[_key]

        def Order_dict_to_delta(self):
            return relativedelta(
                years=self.years,
                months=self.months,
                weeks=self.weeks,
                days=self.days,
                hours=self.hours,
                minutes=self.minutes,
                seconds=self.seconds,
                microsecond=self.microsecond
            )

        def __iter__(self):
            self.__position__ = 0
            return self

        def __next__(self):
            _p = self.__position__
            if _p >= 8:
                raise StopIteration
            if _p == 1:
                _v = self.months
            elif _p == 2:
                _v = self.weeks
            elif _p == 3:
                _v = self.days
            elif _p == 4:
                _v = self.hours
            elif _p == 5:
                _v = self.minutes
            elif _p == 6:
                _v = self.seconds
            elif _p == 7:
                _v = self.microsecond
            else:
                _v = self.years
            self.__position__ += 1
            return str(_v)


class File():
    class Support_Format(Enum):
        JSON = "json"
        CSV = "csv"

    class Json():
        KEYABLE = TYPE_NUMBER | bool | str
        VALUEABLE = KEYABLE | Tuple | list | dict | None
        WRITEABLE = dict[KEYABLE, VALUEABLE]

        @staticmethod
        def Read(
            file_name: str,
            file_dir: str,
            encoding_type: str = "UTF-8"
        ) -> dict:
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
        def Read_from_file(
            file_name: str,
            file_dir: str,
            delimiter: str = "|",
            encoding_type="UTF-8"
        ) -> list[dict[str, Any]]:
            """
            """
            # make file path
            _file = Path.Join([file_name, "csv"], file_dir)
            _is_exist = Path.Exist_check(_file, Path.Type.FILE)

            if _is_exist:
                # read the file
                with open(_file, "r", encoding=encoding_type) as file:
                    _raw_data = csv.DictReader(file, delimiter=delimiter)
                    _read_data = [
                        dict(
                            (
                                _key.replace(" ", ""), _value.replace(" ", "")
                            ) for _key, _value in _line_dict.items()
                        ) for _line_dict in _raw_data
                    ]
                return _read_data
            else:
                print(f"file {file_name} is not exist in {file_dir}")
                return []

        @staticmethod
        def Write_to_file(
            file_name: str,
            file_dir: str,
            data: list[dict],
            feildnames: list[str],
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
