"""파일 입출력 공통 기반.

ABC 베이스 클래스(File_Process), 예외 처리 데코레이터(Handle_exp), 에러 메시지
카탈로그, 확장자 검사 유틸(Suffix_check)을 한 모듈에 모음. 포맷별 모듈
(_text/_json/_yaml)과 디스패처(dispatch)가 공통으로 의존하는 진입점임.

Requirement:
    - Python >= 3.10
    - json, pathlib
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from functools import wraps
from typing import TypeVar, Callable, cast, Any
from pathlib import Path

import json


F = TypeVar("F", bound=Callable)


def Handle_exp(extra_exp: dict[type[Exception], str] | None = None):
    """예외 발생 시 사용자 정의 메시지로 처리하는 데코레이터 팩토리.

    함수 실행 중 예외를 잡아 등록 메시지를 출력 후 (False, None) 반환함.
    파일 입출력 함수에 wrapping하여 예외 메시지를 통합 관리하는 용도.

    Args:
        extra_exp: 예외 클래스 → 출력 메시지 매핑.

    Returns:
        Callable: 데코레이터.
    """
    extra_exp = extra_exp or {}

    def Checker(func: F) -> F:

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                _exp = type(e)
                _msg = extra_exp.get(_exp, "알 수 없는 파일 처리 오류 발생:")
                print(f"{_msg} -> {e}")
            return False, None
        return cast(F, wrapper)

    return Checker


BASIC_FILE_ERROR = {
    PermissionError: "파일 권한 부족"
}

JSON_FILE_READ_ERROR = {
    **BASIC_FILE_ERROR,
    json.JSONDecodeError: "JSON 파싱 오류 발생",
    TypeError: "JSON 변환 불가능한 데이터 포함"
}


def Suffix_check(
    path: Path, ext: list[str] | str, is_fix: bool = True
) -> tuple[bool, Path]:
    """파일 경로의 확장자를 확인하고 조건에 따라 수정.

    Args:
        path: 검사할 파일 경로.
        ext: 허용 확장자 (단일 문자열 또는 리스트).
        is_fix: 불일치 시 자동 수정 여부.

    Returns:
        (일치 여부, (수정된) 파일 경로).
    """
    _ext = ext if isinstance(ext, list) else [ext]
    if path.suffix in _ext:
        return True, path

    return False, path.with_suffix(_ext[0]) if is_fix else path


class File_Process(ABC):
    """파일 입출력 공통 베이스 클래스.

    파일 존재 보장과 일반화된 read/write 인터페이스를 정의함. 자식 클래스는
    포맷별 Read_from / Write_to를 구현해야 함.
    """

    @classmethod
    def Ensure_dir(cls, path: Path) -> None:
        """지정된 파일 경로의 상위 디렉토리가 없을 경우 생성.

        Args:
            path: 디렉토리 생성이 필요한 파일 경로.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    @abstractmethod
    def Read_from(
        cls, *args, file: Path, enc: str = "UTF-8", **kwarg
    ) -> tuple[bool, Any]:
        """파일로부터 데이터를 읽어오는 인터페이스 (미구현)."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def Write_to(
        cls, *args, file: Path, data: Any, enc: str = "UTF-8", **kwarg
    ) -> bool:
        """데이터를 파일로 저장하는 인터페이스 (미구현)."""
        raise NotImplementedError