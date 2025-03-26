from __future__ import annotations
from typing import (
    Any, TypeVar, Generic
)
from dataclasses import asdict, dataclass
from pathlib import Path

import argparse

from .system import Path_utils
from .file import Json


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
            Json.Write_to(
                Path_utils.Join(file_name, file_dir),
                self.Config_to_dict(),
                encoding_type
            )

    cfg_class = TypeVar("cfg_class", bound=Basement)

    @staticmethod
    def Read_from_file(
        config_obj: type[cfg_class],
        file_path: Path,
        encoding_type: str = "UTF-8"
    ):
        _is_done, _meta_data = Json.Read_from(file_path, encoding_type)

        return _is_done, config_obj(**_meta_data)

    @staticmethod
    def Read_with_arg_parser(
        config_template: type[Config.cfg_class],
        parser: argparse.ArgumentParser | None = None
    ):
        if parser is None:
            _parser = argparse.ArgumentParser()
            _parser.add_argument("--cfg_file", dest="cfg_file", type=str)
            return Config.Read_from_file(
                config_template,
                Path(vars(_parser.parse_args())["cfg_file"]),
            )

        _arg_dict = vars(parser.parse_args())
        _cfg_dict = vars(config_template)
        return config_template(
            **dict((
                _k, _v
            ) for _k, _v in _arg_dict.items() if _k in _cfg_dict))


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
    def __init__(
        self, name: str, config: Config.cfg_class, result_dir: str
    ):
        self.project_name = name
        self.project_cfg = config
        self.result_path = Path(result_dir)


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
