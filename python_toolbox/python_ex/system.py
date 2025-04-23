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
from enum import StrEnum, auto
from typing import (Tuple, Literal, TypeVar)

from dataclasses import dataclass, field

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
    """ ### 문자열 처리와 변환 관련 유틸리티 함수 모음
    문자열 정렬, 변환, 출력 포맷 지정 등에 활용되는 기능 제공

    ---------------------------------------------------------------------------
    ### Structure
    - Count_auto_align: 숫자 정렬 문자열 생성
    - Str_adjust: 문자열 길이 조정
    - Progress_bar: 터미널 진행바 출력
    - String_Enum: 문자열로 출력되는 열거형 클래스
    """
    @staticmethod
    def Count_auto_align(
        value: int, max_count: int, is_right: bool = True, fill: str = "0"
    ):
        """ ### 숫자를 정렬하여 문자열 형태로 반환

        ------------------------------------------------------------------
        ### Args
        - value: 출력할 숫자 값
        - max_count: 최대 숫자 (전체 자리수 기준)
        - is_right: 오른쪽 정렬 여부
        - filler: 채울 문자

        ### Returns
        - str: 정렬된 문자열 (예: "003/100")
        """
        if is_right:
            return f"{str(value).rjust(len(str(max_count)), fill)}/{max_count}"
        return f"{str(value).ljust(len(str(max_count)), fill)}/{max_count}"

    @staticmethod
    def Str_adjust(
        text: str,
        max_length: int,
        fill: str = " ",
        align: Literal["l", "c", "r"] = "r"
    ) -> Tuple[int, str]:
        """ ### 문자열을 지정된 길이에 맞춰 정렬

        ------------------------------------------------------------------
        ### Args
        - text: 입력 문자열
        - max_length: 최대 길이
        - fill: 채움 문자
        - align: 정렬 방향 ("l", "c", "r")

        ### Returns
        - Tuple[int, str]: 초과 길이, 정렬된 문자열
        """
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
    def Progress_bar(
        iteration: int, total: int,
        prefix: str = '', suffix: str = '',
        decimals: int = 1, fill: str = '█'
    ):
        """ ### 반복문 내에서 사용할 터미널 진행 표시 바 출력

        ------------------------------------------------------------------
        ### Args
        - iteration: 현재 반복 횟수
        - total: 전체 반복 횟수
        - prefix: 진행바 앞에 붙는 문자열
        - suffix: 진행바 뒤에 붙는 문자열
        - decimals: 퍼센트 표시 소수점 자리수
        - fill: 진행바 채움 문자
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

    class String_Enum(StrEnum):
        """ ### 문자열로 출력되는 열거형 클래스

        ------------------------------------------------------------------
        """
        def __repr__(self) -> str:
            return self.name.lower()


class Operating_System():
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
    def Matches_os(like_this_os: Operating_System.Name | str = "window"):
        """
        #### 제시된 OS와 프로그램이 돌아가는 OS를 비교하는 함수
        ----------------------------------------------------------------
        """
        return Operating_System.THIS_STYLE == like_this_os


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
    def Is_exists(obj_path: str | Path):
        _path = obj_path if isinstance(obj_path, Path) else Path(obj_path)
        return _path.exists(), _path

    @staticmethod
    def Search_in(obj_path: Path, keyword: str = "*", is_sorted: bool = True):
        _list = list(obj_path.glob(keyword))
        return sorted(_list) if is_sorted else _list

    @staticmethod
    def Path_check(path: str | Path):
        return path if isinstance(path, Path) else Path(path)

    @dataclass
    class Server():
        is_window: bool = field(default_factory=Operating_System.Matches_os)

        @classmethod
        def Connect_to(cls):
            # TODO:
            raise NotImplementedError

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
