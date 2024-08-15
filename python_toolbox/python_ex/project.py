from __future__ import annotations
from typing import (
    Any
)
from dataclasses import asdict, dataclass

from .system import Path, File, Time


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
        project_name: str
    ):
        self.project_name = project_name
        self.result_root = Path.WORK_SPACE

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
        _this_time = Time.Stamp()
        _result_dir_dirs = Path.Join(
            [
                "result" if result_root is None else result_root,
                "default" if description is None else description,
                Time.Make_text_from(_this_time, "%Y-%m-%d_%H:%M:%S")
            ]
        )
        self.result_root = Path.Make_directory(_result_dir_dirs)


class Debuging():
    """
    프로젝트 진행에 따른 실행 내역 및 결과와 같은 주요 내용을 생성, 출력, 기록하기 위한 모듈

    --------------------------------------------------------------------
    """

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
