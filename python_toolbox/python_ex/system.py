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
from enum import Enum, auto
from typing import (Tuple, Literal, Any, TypeVar)

from dataclasses import dataclass

import sys
import platform
from os import get_terminal_size

from pathlib import Path

from datetime import datetime, date, time, timezone
from dateutil.relativedelta import relativedelta


# -- DEFINE CONSTNAT -- #
# Data type for hint
NUMBER = TypeVar("NUMBER", bound=int | float)

# System Constant
PYTHON_VERSION = sys.version_info


class String():
    """ ### 문자열 처리 함수
    ---------------------------------------------------------------------------
    """
    @staticmethod
    def Count_auto_align(
        value: int, max_count: int,
        is_right: bool = False, filler: str = "0"
    ):
        _str_len = len(str(max_count))
        _v = str(value).rjust(_str_len, filler) if (
            is_right
        ) else str(value).ljust(_str_len, filler)

        return f"{_v}/{max_count}"

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
    def Convert_from_str(str_data: str, empty_is_None: bool = False) -> Any:
        try:
            if "," in str_data:
                return [
                    String.Convert_from_str(
                        _d
                    ) for _d in str_data[1:-1].split(",")
                ]
            if str_data[0] != "-" and ("-" in str_data or ":" in str_data):
                _use_timezone = "T" in str_data and (
                    any(_txt in str_data.split("T")[-1] for _txt in "+-")
                )
                _use_microsec = "." in str_data
                return Time_Utils.Make_time_from(
                    str_data,
                    use_timezone=_use_timezone,
                    use_microsec=_use_microsec
                )
            if ";" in str_data:
                return Time_Utils.Relative(
                    *[int(_v) for _v in str_data.split(";")]
                )
            if "." in str_data:
                return float(str_data)
            return int(str_data)
        except ValueError:
            if str_data in ("True", "False"):
                return str_data == "True"
            return str_data
        except IndexError:
            return None if empty_is_None else ""

    @staticmethod
    def Convert_to_str(
        data: list | datetime | Time_Utils.Relative | float | int | str
    ) -> str:
        if isinstance(data, list):
            return ",".join(
                [String.Convert_to_str(_data) for _data in data]
            )
        if isinstance(data, (datetime, date)):
            return Time_Utils.Make_text_from(data)
        if isinstance(data, Time_Utils.Relative):
            return ";".join(list(data))
        return data if isinstance(data, str) else str(data)

    @staticmethod
    def Progress_bar(
        iteration: int, total: int,
        prefix: str = '', suffix: str = '',
        decimals: int = 1, fill: str = '█'
    ):
        """
        Call in a loop to create terminal progress bar

        Parameters
        --------------------
        iteration
            current iteration
        total
            total iterations (Int)
        prefix
            prefix string (Str)
        suffix
            suffix string (Str)
        decimals
            positive number of decimals in percent complete (Int)
        length
            character length of bar (Int)
        fill
            bar fill character (Str)
        """
        _percentage = iteration / float(total)
        _str_p = ("{0:." + str(decimals) + "f}").format(100 * _percentage)

        _bias = len(prefix + _str_p + suffix) + 6
        _bar_l = get_terminal_size().columns - _bias
        _fill_l = round(_bar_l * _percentage)
        _str_b = fill * _fill_l + '-' * (_bar_l - _fill_l)

        print(f'\r{prefix} |{_str_b}| {_str_p}% {suffix}', end="\r")
        # Print New Line on Complete
        if iteration == total:
            print()

    class String_Enum(str, Enum):
        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name

        def __repr__(self):
            return self.name.lower()

        def __str__(self):
            return self.name.lower()

        @classmethod
        def list(cls):
            return [str(c) for c in cls]


class OperatingSystem():
    """### Frequently used features for handling OS-related tasks
    --------------------------------------------------------------------
    ### Requirement
        None

    ### Structure
        ...

    """

    THIS_STYLE = platform.system().lower()

    class Name(String.String_Enum):
        """
        각 OS 별 처리 문자열
        ----------------------------------------------------------------
        """
        WINDOW = auto()
        LINUX = auto()

    @staticmethod
    def Is_it_runing(like_this_os: OperatingSystem.Name):
        """
        #### 제시된 OS와 프로그램이 돌아가는 OS를 비교하는 함수
        ----------------------------------------------------------------
        """
        return OperatingSystem.THIS_STYLE == like_this_os


class Path_utils():
    """ ### Pathlib을 사용하여 구현한 자주 사용되는 경로 관련 기능 구현

    ---------------------------------------------------------------------------
    ### Structure
    - `Join`: 경로 생성 함수
    - `Path_split`: 경로 분할 함수
    - `Get_file_name`: 파일 이름 추출 함수
    - `Make_directory`: 디렉토리 생성 함수
    """

    @staticmethod
    def Join(obj_path: str | list[str], root_path: str | None = None):
        """ ### 경로 객체 생성 함수
        문자열 데이터를 사용하여 경로 생성을 위한 기본 함수

        -----------------------------------------------------------------------
        ### Args
        - `obj_path`: 경로 생성을 위한 문자열 데이터
        - `root_path`: 생성되는 경로의 기본 경로. (기본값 = 작업 디렉토리)

        ### Returns
        - `Path`: 결합된 경로 객체

        """
        _path = Path().cwd() if root_path is None else Path(root_path)

        if isinstance(obj_path, str):
            return _path.joinpath(obj_path)
        return _path.joinpath(*obj_path)

    @staticmethod
    def Path_split(obj_path: Path):
        """ ### 경로 분할 함수
        주어진 경로를 나누어 현재 경로의 이름과 부모 경로 문자열 반환하는 함수

        -----------------------------------------------------------------------
        ### Args
        - `obj_path`: 대상 경로

        ### Returns
        - `tuple[str, str]`: 부모 경로 문자열, 현재 경로 문자열

        """
        return str(obj_path.parent), obj_path.name

    @staticmethod
    def Get_file_name(file_path: str | Path):
        """ ### 파일 이름 추출 함수
        파일 경로에서 파일 이름과 부모 경로 문자열 반환하는 함수

        -----------------------------------------------------------------------
        ### Args
        - `file_path`: 대상 경로

        ### Returns
        - `tuple[str, str]`: 부모 경로 문자열, 파일 이름 문자열

        ### Raises
        - ValueError: 주어진 경로가 파일 경로가 아닐 경우
        """

        _obj = file_path if isinstance(file_path, Path) else Path(file_path)

        if _obj.is_file():
            return Path_utils.Path_split(_obj)
        raise ValueError(f"Path {_obj} is not file path")

    @staticmethod
    def Make_directory(
        obj_path: str | list[str],
        root_path: str | None = None, mode: int = 511
    ):
        """ ### 디렉토리 생성 함수
        주어진 디렉토리를 생성하는 함수

        -----------------------------------------------------------------------
        ### Args
        - `obj_path`: 대상 경로
        - `root_path`: 생성되는 경로의 기본 경로. (기본값 = 작업 디렉토리)
        - `mode`: 생성된 경로 계정별 작업 설정

        ### Returns
        - `Path` : 생성된 디렉토리 경로 객체

        """
        _obj = Path_utils.Join(obj_path, root_path)
        _obj.mkdir(mode, parents=True, exist_ok=True)

        return _obj

    @staticmethod
    def Is_exists(obj_path: str | Path):
        _path = obj_path if isinstance(obj_path, Path) else Path(obj_path)
        return _path.exists(), _path

    @staticmethod
    def Search_in(obj_path: Path, keyword: str = "*", is_sorted: bool = True):
        _list = obj_path.glob(keyword)
        return sorted(_list) if is_sorted else _list

    @staticmethod
    def Path_check(path: str | Path):
        return path if isinstance(path, Path) else Path(path)

    class Server():
        ...

    # class Server():
    #     """ ### 네트워크 내 데이터 서버 관련 기능 모음

    #     -----------------------------------------------------------------------
    #     """
    #     IS_WINDOW = OperatingSystem.Is_it_runing(OperatingSystem.Name.WINDOW)

    #     class Connection_Porcess(String.String_Enum):
    #         CIFS = auto()

    #     def __init__(
    #         self,
    #         process: Connection_Porcess = Connection_Porcess.CIFS
    #     ) -> None:
    #         self.process = process

    #     def _connect_to_Linux(
    #         self,
    #         host_name: str,
    #         mount_dir: str,
    #         mount_point: str,
    #         user_id: int,
    #         group_id: int,
    #         dir_mode: int,
    #         file_mode: int,
    #         credent_path: str
    #     ):
    #         _command = path.join(
    #             f"sudo -S mount -t {self.process}",
    #             " -o ",
    #             ",".join((
    #                 f"uid={user_id}",
    #                 f"gid={group_id}",
    #                 f"dir_mode={dir_mode%1000:0>4d}",
    #                 f"dile_mode={file_mode%1000:0>4d}",
    #                 f"credentials={credent_path}",
    #             )),
    #             f" //{host_name}/{mount_dir} {mount_point}"
    #         )
    #         system(_command)

    #         return mount_point

    #     def _connect_to_Window(
    #         self,
    #         host_name: str,
    #         mount_dir: str,
    #         mount_point: str,
    #         user_name: str
    #     ):
    #         raise NotImplementedError

    #     def _disconnect(self, mounted_dir: str):
    #         if self.IS_WINDOW:
    #             system(f"NET USE {mounted_dir}: /DELETE")
    #         else:
    #             system(f"fuser -ck {mounted_dir}")
    #             system(f"sudo umount {mounted_dir}")


class Time_Utils():
    @staticmethod
    def Stamp(set_timezone: timezone | None = None):
        """ ### 현재 시간 정보를 생성하는 함수

        ---------------------------------------------------------------------------------------
        ### Parameters
        - `set_timezone` : 타임존 정보

        ### Return
        - `this_time` : 현재 시간 정보
        """
        return datetime.now(set_timezone)

    @staticmethod
    def Get_term(
        standard_time: datetime,
        set_timezone: timezone | None = None
    ):
        return Time_Utils.Stamp(set_timezone) - standard_time

    @staticmethod
    def Make_text_from(
        time_source: datetime | date | time | None = None,
        date_format: str | None = None
    ):
        _time = Time_Utils.Stamp() if time_source is None else time_source
        if date_format is None:
            return _time.isoformat()
        return _time.strftime(date_format)

    @staticmethod
    def Make_time_from(
        text_source: str,
        date_format: str | None = None,
        use_microsec: bool = False,
        use_timezone: bool = False
    ):
        if date_format is not None:
            _date_format = date_format
        else:  # iso
            _date_format = "%Y-%m-%dT%H:%M:%S"
            _date_format += ".%f" if use_microsec else ""
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
