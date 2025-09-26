"""파일 I/O 진입점이 부여된 Data_Schema 특화 모듈.

JSON/YAML 파일 또는 argparse.Namespace로부터의 객체 구성과 파일 저장을
지원하는 Base_Config 및 팩토리 함수를 제공함. python_toolbox.file 의존을
이 모듈로 격리하여 Data_Schema 코어를 stdlib만 의존하도록 유지함.

Requirement:
    - Python >= 3.10
    - argparse, pathlib
    - python_toolbox.data_schema, python_toolbox.file
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
import argparse

from python_toolbox.data_schema import Data_Schema
from python_toolbox.file import (
    Read_from as _File_Read_from,
    Write_to as _File_Write_to,
    Suffix_check,
)


@dataclass
class Base_Config(Data_Schema):
    """파일 I/O 진입점을 보유한 Data_Schema 특화 클래스.

    설정/구성 데이터를 JSON/YAML로 저장하거나, argparse 결과 또는
    설정 파일에서 직접 객체를 구성하는 용도에 적합함. 데이터 스키마로서의
    역할(Serialize/Extract)은 Data_Schema에서 상속받음.

    ## 사용 패턴
    - `cfg.Write_to(name, dir)` — 현재 상태를 파일로 저장.
    - `Build_from_args(MyConfig, ns)` — argparse.Namespace에서 객체 구성.
    - `Read_from_file(MyConfig, path)` — JSON/YAML에서 객체 구성.
    """

    def Write_to(
        self, name: str, save_dir: str | Path, encoding_type: str = "UTF-8"
    ) -> None:
        """현재 상태를 Serialize 결과로 파일에 저장함.

        Args:
            name: 저장 파일명 (확장자 포함, 예: "config.json").
            save_dir: 저장 디렉토리 경로.
            encoding_type: 파일 인코딩 (기본 UTF-8).
        """
        _File_Write_to(Path(save_dir) / name, self.Serialize(), encoding_type)


C_Type = TypeVar("C_Type", bound=Base_Config)


def Build_from_args(
    cfg_obj: type[C_Type], args: argparse.Namespace | dict[str, Any]
) -> C_Type:
    """인자 데이터로부터 Base_Config 객체를 구성함.

    모든 객체 생성의 단일 진입점(Single Source of Truth) 역할 수행.
    파일 로드 경로(Read_from_file) 또한 최종적으로 본 함수에 위임됨.

    Args:
        cfg_obj: 구성할 Base_Config 자식 클래스.
        args: argparse.Namespace 또는 dict 형태의 인자.

    Returns:
        구성된 Base_Config 인스턴스.

    Raises:
        ValueError: 시그니처 불일치 등으로 객체 생성 실패 시.
    """
    _arg_dict = vars(args) if isinstance(args, argparse.Namespace) else args

    try:
        return cfg_obj(**_arg_dict)
    except TypeError as e:
        raise ValueError(
            f"[ERROR] Failed to build config '{cfg_obj.__name__}': {e}"
        ) from e


def Read_from_file(
    cfg_obj: type[C_Type], file_path: Path, encoding_type: str = "UTF-8"
) -> C_Type:
    """JSON/YAML 파일에서 Base_Config 객체를 구성함.

    I/O 처리 및 포맷 검증만 담당하고 객체 생성은 Build_from_args에 위임함.

    Args:
        cfg_obj: 구성할 Base_Config 자식 클래스.
        file_path: 설정 파일 경로 (.json 또는 .yaml).
        encoding_type: 파일 인코딩 (기본 UTF-8).

    Returns:
        구성된 Base_Config 인스턴스.

    Raises:
        ValueError: 미지원 포맷, 파싱 실패, 또는 dict 형태가 아닌 경우.
    """
    _is_ok, _ = Suffix_check(file_path, [".json", ".yaml"], is_fix=False)
    if not _is_ok:
        raise ValueError(
            f"[ERROR] Unsupported config file format: {file_path.suffix}"
        )

    _is_read_ok, _data = _File_Read_from(file_path, enc=encoding_type)

    if not _is_read_ok or not isinstance(_data, dict):
        raise ValueError(
            f"[ERROR] Failed to read or parse config file: {file_path}"
        )

    return Build_from_args(cfg_obj, _data)
