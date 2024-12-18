from __future__ import annotations
from typing import (
    Any, TypeVar, Generic
)
from dataclasses import asdict, dataclass

import argparse

from .system import Path_utils, Time_Utils
from .file import Read_from_file, Write_to_file


class Config():
    @dataclass
    class Basement():
        """ ### 프로젝트에 사용되는 설정을 관리하기 위한 객체(dataclass) 구조
        기존에 사용되는 dict 구조 대비 효율적으로 사용하고 관리하기 위한 객체\n
        field 함수와 __post_init__을 이용하여 자동화 값 자동화 구현

        -----------------------------------------------------------------------
        ### Structure
        - `Config_to_dict`: 객체를 저장 가능한 dict 데이터 구조로 변환는 함수
        - `Write_to`: 주어진 경로 정보에 객체 정보를 저장하는 함수
        - `Get_summation`: 객체를 구성하는 설정 값들 정보를 취합하는 함수

        """
        def Config_to_dict(self) -> dict[str, Any]:
            """ ### 객체를 저장 가능한 dict 데이터 구조로 변환하는 함수

            -------------------------------------------------------------------
            ### Return
            - `dict[str, Any]` : 저장 가능한 dict 데이터

            -------------------------------------------------------------------
            """
            return asdict(self)

        def Write_to(
            self, file_name: str, file_dir: str, encoding_type: str = "UTF-8"
        ):
            """ ### 주어진 경로 정보에 객체 정보를 저장하는 함수
            -------------------------------------------------------------------
            ### Args
            - `file_name`: 파일 이름
            - `obj_path`: 대상 경로
            - `encoding_type`: 저장 파일 문자열 포멧 (기본값 = UTF-8)

            -------------------------------------------------------------------
            """
            try:
                Write_to_file(
                    file_name, self.Config_to_dict(), file_dir, encoding_type)
            except ValueError:
                _cfg_name = self.__class__.__name__
                _msg = "If you want save this file,"
                _msg += f" override the function `Write_to` in {_cfg_name}."
                print(_msg)

    cfg_class = TypeVar("cfg_class", bound=Basement)

    @staticmethod
    def Read_from_file(
        config_obj: type[cfg_class], file_name: str,
        file_dir: str | None = None, encoding_type: str = "UTF-8"
    ):
        try:
            return config_obj(
                **Read_from_file(file_name, file_dir, encoding_type))

        except ValueError as _value_e:
            print("If you want do this, override the function 'Read_from'.")
            raise ValueError from _value_e


class Project_Template(Generic[Config.cfg_class]):
    """ ### 프로젝트 구성을 위한 기본 구조

    ---------------------------------------------------------------------
    ### Args
    - `name`: 프로젝트 이름
    - `config_type`: 프로젝트에 사용되는 config class 타입

    ### Attributes
    - `project_name`: 프로젝트 이름
    - `project_cfg`: 프로젝트 config 데이터
    - `save_root`: 프로젝트 결과 저장 최상위 경로

    ### Structure
    - `_Get_args_by_arg_parser`
    - `_Get_config`
    - `_Make_save_root`: 프로젝트 결과 저장 최상위 경로 생성 함수

    """
    def __init__(self, name: str, config_type: type[Config.cfg_class]):
        self.project_name = name
        self.project_cfg: Config.cfg_class = self._Get_config(config_type)
        self.save_root = self._Make_save_root()

    def _Get_args_by_arg_parser(self) -> argparse.ArgumentParser:
        _parser = argparse.ArgumentParser()
        _parser.add_argument("--cfg_file", dest="cfg_file", type=str)
        return _parser

    def _Get_config(
        self, config_template: type[Config.cfg_class]
    ) -> Config.cfg_class:
        _arg_dict = vars(
            self._Get_args_by_arg_parser().parse_args())

        _file_path = _arg_dict["cfg_file"]

        if _arg_dict["cfg_file"] is not None:
            _arg_dict.pop("cfg_file")
            _f_dir, _f_name = Path_utils.Get_file_name(_file_path)

            return Config.Read_from_file(config_template, _f_name, _f_dir)

        _cfg_dict = vars(config_template)
        return config_template(
            **dict((
                _k, _v
            ) for _k, _v in _arg_dict.items() if _k in _cfg_dict))

    def _Make_save_root(
        self, obj_dir: str | list[str] = "result", base_path: str | None = None
    ):
        """ ### 프로젝트 결과 저장 최상위 경로 생성 함수

        ------------------------------------------------------------------
        ### Args
            - `obj_dir`:

        ### Returns
            - `Path`: 저장 최상위 경로
        """
        _obj_dir = obj_dir if isinstance(obj_dir, list) else [obj_dir]
        _obj_dir += [Time_Utils.Make_text_from()]

        return Path_utils.Make_directory(_obj_dir, base_path)


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
