from __future__ import annotations
from typing import (
    Any, TypeVar, Generic
)
from dataclasses import asdict, dataclass
from pathlib import Path

import argparse

from .file import Utils, Suffix_check


class Config():
    """ ### 프로젝트 설정 관리를 위한 유틸리티 클래스

    설정 정보 객체화를 통해 dict 기반 대비 가독성과 유지보수성 향상

    ---------------------------------------------------------------------------
    ### Structure
    - Basement: 설정 기본 구조를 정의한 dataclass
    - Read_from_file: 설정 파일에서 설정 정보를 불러오는 함수
    - Read_with_arg_parser: argparse 기반 설정 파싱 (미구현)
    """

    @dataclass
    class Basement():
        """ ### 프로젝트 설정 값을 담는 클래스 템플릿

        field 및 __post_init__ 기능을 통해 자동 처리 로직 적용 가능

        -----------------------------------------------------------------------
        ### Structure
        - Config_to_dict: 객체를 dict 형태로 변환
        - Write_to: 객체를 파일로 저장
        """
        def Config_to_dict(self) -> dict[str, Any]:
            """ ### 객체 정보를 dict 데이터로 변환

            ------------------------------------------------------------------
            ### Returns
            - dict[str, Any]: 저장 가능한 형태의 설정 정보
            """
            return asdict(self)

        def Write_to(
            self, name: str, save_dir: str, encoding_type: str = "UTF-8"
        ):
            """ ### 설정 정보를 주어진 경로에 파일로 저장

            ------------------------------------------------------------------
            ### Args
            - name: 저장할 파일 이름
            - save_dir: 저장할 디렉토리 경로
            - encoding_type: 인코딩 형식 (기본값: UTF-8)
            """
            Utils.Write_to(
                Path(save_dir) / name, self.Config_to_dict(), encoding_type
            )

    cfg_class = TypeVar("cfg_class", bound=Basement)

    @staticmethod
    def Read_from_file(
        cfg_obj: type[cfg_class], file_path: Path, encoding_type: str = "UTF-8"
    ) -> tuple[bool, cfg_class]:
        """ ### 파일에서 설정 정보를 로드하여 객체로 반환

        json 또는 yaml 포맷을 지원함

        ------------------------------------------------------------------
        ### Args
        - cfg_obj: 설정 클래스 타입
        - file_path: 읽어올 파일 경로
        - encoding_type: 파일 인코딩 (기본값: UTF-8)

        ### Returns
        - Tuple[bool, cfg_class]: 성공 여부, 설정 객체 인스턴스
        """
        _is_ok, _ = Suffix_check(file_path, [".json", ".yaml"], True)

        if _is_ok:
            _meta: tuple[bool, dict[str, Any]] = Utils.Read_from(
                file_path, encoding_type)
            return _meta[0], cfg_obj(**_meta[1])

        return False, cfg_obj()

    @staticmethod
    def Read_with_arg_parser(
        config_template: type[Config.cfg_class],
        parser: argparse.ArgumentParser | None = None
    ) -> tuple[bool, Config.cfg_class]:
        """ ### argparse 파서를 통해 설정 정보를 구성 (미구현)

        향후 커맨드라인 기반 설정 파싱을 위한 인터페이스

        ------------------------------------------------------------------
        ### Raises
        - NotImplementedError: 현재 구현되지 않음
        """
        # TODO:
        raise NotImplementedError("this function has problem")


class Project_Template(Generic[Config.cfg_class]):
    """ ### 프로젝트 구조를 정의하는 템플릿 클래스

    프로젝트명, 설정값, 결과 저장 경로 등을 관리

    ---------------------------------------------------------------------------
    ### Structure
    - __init__: 프로젝트 초기 설정
    - Get_result_path: 결과 경로 반환 메서드 (미구현)
    """
    def __init__(
        self, name: str, config: Config.cfg_class
    ):
        """ ### 프로젝트 템플릿 초기화

        ------------------------------------------------------------------
        ### Args
            - This
                - name: 프로젝트 이름
                - config: Config 클래스에서 파생된 설정 객체

        ### Attributes
        - project_name: 프로젝트 이름 문자열
        - result_path: 설정 기반으로 계산된 결과 저장 경로

        ### Structure
        - __init__: 템플릿 초기화 메서드
        - Get_result_path: 결과 저장 경로를 반환하는 함수 (미구현)
        """
        self.project_name = name
        self.result_path = self.Get_result_path(config)

    def Get_result_path(self, config: Config.cfg_class) -> Path:
        """ ### 결과 저장 경로를 반환하는 함수

        ------------------------------------------------------------------
        ### Raises
        - NotImplementedError: 각 프로젝트마다 별도 구현 필요
        """
        raise NotImplementedError
