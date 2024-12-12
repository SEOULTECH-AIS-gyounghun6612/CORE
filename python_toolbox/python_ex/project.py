from __future__ import annotations
from typing import (
    Any, TypeVar, Generic
)
from dataclasses import asdict, dataclass

import argparse

from .system import Path, File, Time


class Config():
    @dataclass
    class Basement():
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

        def Set_from(self, source: dict[str, Any]):
            for _k, _v in source.items():
                if _k in self.__dict__:
                    self.__dict__[_k] = _v

        def Get_summation(self) -> list[str]:
            raise NotImplementedError

    @staticmethod
    def Read_from_file(file_name: str, file_dir: str):
        _ext = file_name.split(".")[-1].lower()

        if _ext == "yaml":
            return File.Yaml.Read(file_name, file_dir)
        if _ext == "json":
            return File.Json.Read(file_name, file_dir)

        _msg = f"file foramt '{_ext}' is not support for build config "
        _msg += "by using this code.\n"
        _msg += "If you want use do file foramt '{_ext}', "
        _msg += "overide the function 'Config.Read_from'."
        raise ValueError(_msg)


CFG = TypeVar("CFG", bound=Config.Basement)


class Project_Template(Generic[CFG]):
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
    def __init__(self, config_template: type[CFG]):
        self.project_CFG: CFG = self.Get_config(config_template)
        self.save_root = self.Make_save_root()

    def Get_args_by_arg_parser(self) -> argparse.ArgumentParser:
        _parser = argparse.ArgumentParser()
        _parser.add_argument(
            "--file_path", "-f",
            dest="file_path",
            type=str)
        return _parser

    def Get_config(self, config_template: type[CFG]) -> CFG:
        _arg_dict = vars(
            self.Get_args_by_arg_parser().parse_args())

        _file_path = _arg_dict["file_path"]
        if _file_path is not None:
            _f_dir, f_name = Path.Devide(_file_path)
            _arg_dict.update(Config.Read_from_file(f_name, _f_dir))

        _arg_dict.pop("file_path")
        return config_template(**_arg_dict)

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
