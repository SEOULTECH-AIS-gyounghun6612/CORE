from __future__ import annotations
from typing import (
    Any, Tuple, Literal, Type, TypeVar, Generic
)
from dataclasses import asdict, dataclass

from datetime import datetime, date, time, timezone
from dateutil.relativedelta import relativedelta
from math import log10, floor

from .system import Path, File


# -- DEFINE CONSTNAT -- #


# -- Mation Function -- #
@dataclass
class Config():
    """
    프로젝트에 사용되는 인자값을 관리하기 위한 객체(dataclass) 기본 구조

    --------------------------------------------------------------------
    """
    def Config_to_parameter(self) -> dict[str, Any]:
        """
        설정값을 사용가능한 인자값으로 변경하는 함수

        ----------------------------------------------------------------
        ### Parameters
        - None

        ### Return
        - dictionary : 파라메터로 활용하기 위하여 구성된 인자값

        ----------------------------------------------------------------
        """
        return asdict(self)

    def Write_to(self, file_name: str, file_dir: str):
        File.Json.Write(file_name, file_dir, asdict(self))  # type: ignore

    @staticmethod
    def Read_from(file_name: str, file_dir: str):
        return File.Json.Read(file_name, file_dir)


class Data_n_Block():
    @dataclass
    class Numbered_Data():
        id_num: int

        def _Str_adjust(
            self,
            key: str,
            value: str,
            data_size: dict[str, int] | None = None,
            align: Literal["l", "c", "r"] = "r",
        ):
            if data_size is not None and key in data_size:
                return (
                    Debuging.Str_adjust(key, data_size[key], mode=align)[-1],
                    Debuging.Str_adjust(value, data_size[key], mode=align)[-1]
                )
            return (key, value)

        def Convert_from_str(self, **kwarg: str):
            raise NotImplementedError

        def Convert_to_str(
            self,
            additional: dict[str, str] | None = None,
            data_size: dict[str, int] | None = None
        ) -> dict[str, str]:
            _data: dict[str, str] = dict((
                self._Str_adjust("id_num", str(self.id_num), data_size),
            ))

            if additional is not None:
                _data.update(additional)

            return _data

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                for _key, _value in self.__dict__.items():
                    if _key == "id_num":
                        continue

                    if _value == other.__dict__[_key]:
                        return False
                    return True
            return False

        def __ne__(self, other: Data_n_Block.Numbered_Data):
            return not self.__eq__(other)

    NUMBERED_DATA = TypeVar("NUMBERED_DATA", bound=Numbered_Data)

    class Block(Generic[NUMBERED_DATA]):
        def __init__(
            self,
            data_format: Type[Data_n_Block.NUMBERED_DATA],
            file_name: str = "data",
            file_dir: str = Path.WORK_SPACE,
        ) -> None:
            _file_path = Path.Join(file_name, file_dir)

            self.data_format = data_format

            if Path.Exist_check(_file_path):
                self.data_dict = self.Read_from_csv(
                    data_format, file_name, file_dir
                )
                self.next_id = max(self.data_dict) + 1
            else:
                self.data_dict: dict[int, Data_n_Block.NUMBERED_DATA] = {}
                self.next_id = 0

        def Read_from_csv(
            self,
            data_format: Type[Data_n_Block.NUMBERED_DATA],
            file_name: str,
            file_dir: str
        ) -> dict[int, Data_n_Block.NUMBERED_DATA]:
            _holder: dict[int, data_format] = {}

            for _data in File.CSV.Read_from_file(file_name, file_dir):
                _id_num = int(_data["id_num"])

                _comp: Data_n_Block.NUMBERED_DATA = data_format(
                    int(_data["id_num"]),)
                _comp.Convert_from_str(**_data)
                _holder[_id_num] = _comp

            return _holder

        def Write_to_csv(
            self,
            file_name: str,
            file_dir: str,
            data_socket_size: dict[str, int] | None = None
        ):
            _data_dict = self.data_dict

            if not _data_dict:
                return False

            return File.CSV.Write_to_file(
                file_name,
                file_dir,
                [
                    _data.Convert_to_str(
                        data_size=data_socket_size
                    ) for _data in _data_dict.values()
                ],
                list(self.data_format.__annotations__.__dict__)
            )

        def Set_data(
            self,
            new_data: Data_n_Block.NUMBERED_DATA,
            is_override: bool = False
        ) -> bool:

            if isinstance(new_data, self.data_format):
                _data_id = new_data.id_num
                if is_override:
                    if _data_id in self.data_dict:  # override
                        self.data_dict.update({_data_id: new_data})
                        return True
                elif new_data not in self.data_dict.values():  # add
                    _this_id = self.next_id
                    new_data.id_num = self.next_id
                    self.data_dict[_this_id] = new_data
                    self.next_id += 1
                    return True
            return False

        def Get_data_from(self, id_num: int, is_pop: bool = False):
            if id_num in self.data_dict:
                if is_pop:
                    return True, self.data_dict.pop(id_num)
                else:
                    return True, self.data_dict[id_num]
            return False, None

        def Clear_data(self):
            self.data_dict = {}
            self.next_id = 0


class Template():
    """ ### 프로젝트 구성을 위한 기본 구조

    ---------------------------------------------------------------------
    ### Args
    - Super
        - None
    - This
        - `project_name`: 프로젝트 이름
        - `category`: 프로젝트 구분
        - `result_dir`: 프로젝트 결과 저장 최상위 경로를 생성하기 위한 경로

    ### Attributes
    - `project_name`: 프로젝트 이름
    - `result_root`: 프로젝트 결과 저장 최상위 경로

    ### Structure
    - Make_save_root: 프로젝트 결과 저장 최상위 경로 생성 함수

    """
    def __init__(
        self,
        project_name: str,
        category: str | None = None,
        result_dir: str | None = None
    ):
        self.project_name = project_name
        self.Make_save_root(category, result_dir)

    def Make_save_root(
        self,
        description: str | None = None,
        result_root: str | None = None
    ):
        """ ### 프로젝트 결과 저장 최상위 경로 생성 함수

        ------------------------------------------------------------------
        ### Args
            - arg_name: Description of the input argument

        ### Returns or Yields
            - data_format: Description of the output argument

        ### Raises
            - None

        """
        _this_time = Debuging.Time.Stamp()
        _result_dir_dirs = Path.Join(
            [
                "result" if result_root is None else result_root,
                "default" if description is None else description,
                "default" if description is None else description,
                Debuging.Time.Make_text_from(_this_time, "%Y-%m-%d_%H:%M:%S")
            ]
        )
        self.result_root = Path.Make_directory(_result_dir_dirs)


class Debuging():
    """
    프로젝트 진행에 따른 실행 내역 및 결과와 같은 주요 내용을 생성, 출력, 기록하기 위한 모듈

    --------------------------------------------------------------------
    ### Module list
    - Time
    - Progress
    - Logging
    """

    @staticmethod
    def Str_adjust(
        text: str,
        max_length: int,
        fill: str = " ",
        mode: Literal["l", "c", "r"] = "r"
    ) -> Tuple[int, str]:
        for _str in text:
            max_length -= 1 if _str.encode().isalpha() ^ _str.isalpha() else 0

        if max_length < 0:
            return -max_length, text
        if mode == "l":
            return 0, text.ljust(max_length, fill)
        if mode == "c":
            return 0, text.center(max_length, fill)
        return 0, text.rjust(max_length, fill)

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
            _term = Debuging.Time.Stamp(set_timezone) - standard_time
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
        class Relative_Delta():
            years: int = 0
            months: int = 0
            weeks: int = 0
            days: int = 0
            hours: int = 0
            minutes: int = 0
            seconds: int = 0
            microsecond: int = 0

            def Delta_to_order_dict(self, time_delta: relativedelta):
                for _key in self.__dict__:
                    self.__dict__[_key] = time_delta.__dict__[_key]

            def Order_dict_to_delta(self):
                return relativedelta(**self.__dict__)

            def __iter__(self):
                self._position = 0
                return self

            def __next__(self):
                _p = self._position
                if _p >= 8:
                    raise StopIteration
                elif _p == 1:
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
                self._position += 1
                return str(_v)

    class Progress():
        @staticmethod
        def Count_auto_aligning(this_count: int, max_count: int):
            _string_ct = floor(log10(max_count)) + 1
            _this = f"{this_count}".rjust(_string_ct, "0")

            return f"{_this}/{max_count}"

        @staticmethod
        def Progress_bar(
            iteration: int,
            total: int,
            prefix: str = '',
            suffix: str = '',
            decimals: int = 1,
            length: int = 100,
            fill: str = '█'
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
            percent = (
                "{0:." + str(decimals) + "f}"
            ).format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end="\r")
            # Print New Line on Complete
            if iteration == total:
                print()
