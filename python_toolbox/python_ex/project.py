from typing import Dict, List, Any, Literal, Union, Type
from dataclasses import asdict, dataclass

from datetime import datetime, date, time, timezone
# from dateutil.relativedelta import relativedelta
from math import log10, floor
from numpy import ndarray

import cv2

from .system import Path, File


# -- DEFINE CONSTNAT -- #


# -- Mation Function -- #
@dataclass
class Config():
    """
    프로젝트에 사용되는 인자값을 관리하기 위한 객체(dataclass) 기본 구조

    --------------------------------------------------------------------
    """
    def Config_to_parameter(self) -> Dict[str, Any]:
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
        _this_time = Debuging.Time.Stemp()
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
    ) -> str:
        for _str in text:
            max_length -= 1 if _str.encode().isalpha() ^ _str.isalpha() else 0
        if mode == "l":
            return text.ljust(max_length, fill)
        elif mode == "c":
            return text.center(max_length, fill)
        else:
            return text.rjust(max_length, fill)

    class Time():
        @staticmethod
        def Stemp(set_timezone: timezone | None = None):
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
            _term = Debuging.Time.Stemp(set_timezone) - standard_time
            return str(_term) if to_str else _term

        @staticmethod
        def Make_text_from(
            time_source: Union[datetime, date, time],
            date_format: str | None = None
        ):
            if date_format is None:
                return time_source.isoformat()
            else:
                return time_source.strftime(date_format)

        @staticmethod
        def Make_time_from(
            text_source: str,
            time_type: Type[Union[datetime, date, time]],
            date_format: str | None = None,
            use_timezone: bool = False
        ):
            if date_format is not None:
                _date_format = date_format
            else:
                _date_format = "%Y-%m-%dT%H:%M:%S"
                _date_format += "%z" if use_timezone else ""

            _datetime = datetime.strptime(text_source, _date_format)
            if time_type is date:
                return _datetime.date()
            elif time_type is time:
                return _datetime.time()
            else:
                return _datetime

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

    class Visualize():
        @staticmethod
        def Put_text_to_img(
            draw_img: ndarray,
            location: List[int],
            text_list: List[str],
            padding: int
        ):
            # draw image check
            _shape = draw_img.shape

            if len(_shape) > 2:
                _h, _w = _shape[:2]
            else:
                _h, _w = _shape

            # decide text position in image
            _p_y, _p_x = location
            _txt_h = 0
            _txt_w = 0

            for _text in text_list:  # get text box size
                (_size_x, _size_y), _ = cv2.getTextSize(_text, 1, 1, 1)
                _txt_h += _size_y
                _txt_w = _size_x if _size_x > _txt_w else _txt_w

            _is_over_h = 2 * (_txt_h + padding) > _h
            _is_over_w = 2 * (_txt_w + padding) > _w

            if _is_over_h or _is_over_w:  # can't put text box in image
                return False, draw_img
            else:  # default => right under
                # set text box x position
                if (_p_x + padding + _txt_w) < _w:
                    _text_x = _p_x + padding
                else:
                    _text_x = _p_x - (_txt_w + padding)
                # set text box y position
                if (_p_y + _txt_h + padding) < _h:
                    _text_y = _p_y + _txt_h + padding
                else:
                    _text_y = _p_y - padding

                for _ct, _text in enumerate(reversed(text_list)):
                    cv2.putText(
                        draw_img,
                        _text,
                        [_text_x, _text_y - (_ct * _size_y)],
                        1,
                        fontScale=1,
                        color=(0, 255, 0)
                    )
                return True, draw_img

        @staticmethod
        def Image_Labeling():
            ...
