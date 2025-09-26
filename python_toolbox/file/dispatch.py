"""확장자 기반 파일 입출력 디스패치.

신규 포맷 지원 추가 시 본 모듈의 _REGISTRY에 한 줄 등록만 하면 자동 노출됨.
포맷별 처리 클래스는 File_Process를 상속하여 Read_from / Write_to를 구현해야 함.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

from ._base import File_Process
from ._text import Text
from ._json import Json
from ._yaml import Yaml


# 확장자(소문자) → 처리 클래스 매핑. 신규 포맷 추가 시 본 dict에만 등록.
_REGISTRY: dict[str, type[File_Process]] = {
    ".txt": Text,
    ".json": Json,
    ".yaml": Yaml,
}


def Read_from(file: Path, enc: str = "UTF-8", **kwarg) -> tuple[bool, Any]:
    """파일 확장자에 따라 알맞은 읽기 메서드를 호출함.

    Args:
        file: 읽을 파일 경로.
        enc: 인코딩.
        kwarg: 포맷별 확장 인자.

    Returns:
        (성공 여부, 읽은 데이터).

    Raises:
        ValueError: 파일이 아니거나 미지원 확장자인 경우.
    """
    if not Path.is_file(file):
        raise ValueError(f"!!! This path {file} is not FILE !!!")

    _processor = _REGISTRY.get(file.suffix.lower())
    if _processor is None:
        raise ValueError(
            f"File extension '{file.suffix}' is not supported"
        )

    return _processor.Read_from(file, enc, **kwarg)


def Write_to(
    file: Path, data: Any, enc: str = "UTF-8", **kwarg
) -> None:
    """파일 확장자에 따라 알맞은 쓰기 메서드를 호출함.

    Args:
        file: 저장할 파일 경로.
        data: 저장할 데이터.
        enc: 인코딩.
        kwarg: 포맷별 확장 인자.

    Raises:
        ValueError: 미지원 확장자인 경우.
    """
    _processor = _REGISTRY.get(file.suffix.lower())
    if _processor is None:
        raise ValueError(
            f"File extension '{file.suffix}' is not supported for writing."
        )

    _processor.Write_to(file, data, enc, **kwarg)
