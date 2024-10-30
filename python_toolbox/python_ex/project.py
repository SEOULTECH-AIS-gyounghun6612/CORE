from __future__ import annotations
from typing import (
    Any
)
from dataclasses import asdict, dataclass

import argparse

from .system import Path, File, Time


@dataclass
class Config():
    """
    프로젝트에 사용되는 인자값을 관리하기 위한 객체(dataclass) 기본 구조

    --------------------------------------------------------------------
    """
    def Config_to_dict(self) -> dict[str, Any]:
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

    def Write_to(
        self,
        file_name: str,
        file_dir: str
    ):
        _ext = file_name.split(".")[-1].lower()
        _write_data = self.Config_to_dict()

        if _ext == "yaml":
            File.Yaml.Write(file_name, file_dir, _write_data)

        elif _ext == "json":
            File.Json.Write(file_name, file_dir, _write_data)

        else:
            _msg = f"Config {self.__class__.__name__} is not support"
            _msg += f" {_ext} file foramt for write to file.\n"
            _msg += "If you want do this, overide the function 'Write_to'."
            raise ValueError(_msg)

    def Read_from(self, file_name: str, file_dir: str):
        _ext = file_name.split(".")[-1].lower()

        if _ext == "yaml":
            _data = File.Yaml.Read(file_name, file_dir)
        elif _ext == "json":
            _data =  File.Json.Read(file_name, file_dir)
        else:
            _msg = f"Config {self.__class__.__name__} is not support"
            _msg += f" {_ext} file foramt for read from file.\n"
            _msg += "If you want do this, overide the function 'Read_from'."
            raise ValueError(_msg)

        self.Set_from(_data)

    def Set_from(self, source: dict[str, Any]):
        for _k, _v in source.items():
            if _k in self.__dict__:
                self.__dict__[_k] = _v

    def Get_summation(self) -> list[str]:
        raise NotImplementedError


@dataclass
class Project_Args():
    name: str


class Project_Template():
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
    project_args: Project_Args

    def __init__(self):
        self.project_args = self.project_args.__class__(
            **vars(self.Get_args_by_argparser().parse_args())
        )
        self.result_root = self.Make_save_root()

    def Get_args_by_argparser(self) -> argparse.ArgumentParser:
        _parser = argparse.ArgumentParser()
        _parser.add_argument(
            "-n", "--name", dest="name", type=str, help="name for this project"
        )
        return _parser

    def Make_save_root(
        self,
        obj_dir: str | list[str] | None = None,
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

        if obj_dir is None:
            _obj_dir = [
                "result",
                self.project_args.name,
                Time.Make_text_from(Time.Stamp(), "%Y-%m-%d_%H:%M:%S")
            ]
        else:
            _obj_dir = obj_dir

        return Path.Make_directory(
            _obj_dir, Path.WORK_SPACE if result_root else result_root)


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
