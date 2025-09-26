"""JSON(.json) 파일 입출력."""
from __future__ import annotations
from pathlib import Path
from typing import Any

import json

from ._base import File_Process, Handle_exp, Suffix_check, JSON_FILE_READ_ERROR


class Json(File_Process):
    """JSON 파일의 읽기/쓰기를 제공하는 클래스."""

    @classmethod
    @Handle_exp(extra_exp=JSON_FILE_READ_ERROR)
    def Read_from(
        cls, file: Path, enc: str = "UTF-8"
    ) -> tuple[bool, dict[str, Any]]:
        """JSON 파일을 읽어 딕셔너리로 반환.

        Args:
            file: JSON 파일 경로.
            enc: 인코딩.

        Returns:
            (성공 여부, 데이터 딕셔너리).
        """
        _, _file = Suffix_check(file, ".json")

        if not _file.exists():
            return False, {}

        with _file.open(encoding=enc) as _f:
            return True, json.load(_f)

    @classmethod
    @Handle_exp()
    def Write_to(
        cls, file: Path, data: Any, enc: str = "UTF-8", indent: int = 4
    ) -> bool:
        """딕셔너리 데이터를 JSON 파일로 저장.

        Args:
            file: 저장 파일 경로.
            data: 저장 데이터.
            enc: 인코딩.
            indent: 들여쓰기 수준.

        Returns:
            저장 성공 여부.
        """
        cls.Ensure_dir(file)
        _, _path = Suffix_check(file, ".json", True)

        with _path.open(mode="w", encoding=enc) as _f:
            json.dump(data, _f, indent=indent)
        return True