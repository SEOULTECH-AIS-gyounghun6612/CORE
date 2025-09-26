"""YAML(.yaml) 파일 입출력."""
from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml

from ._base import File_Process, Handle_exp, Suffix_check


class Yaml(File_Process):
    """YAML 파일의 읽기/쓰기를 제공하는 클래스."""

    loader = yaml.FullLoader

    @classmethod
    @Handle_exp()
    def Read_from(
        cls, file: Path, enc: str = "UTF-8", **kwarg
    ) -> tuple[bool, Any]:
        """YAML 파일을 읽어 객체로 반환.

        Args:
            file: YAML 파일 경로.
            enc: 인코딩.

        Returns:
            (성공 여부, 파싱된 객체).
        """
        _, _file = Suffix_check(file, ".yaml")

        if not _file.exists():
            return False, {}

        with _file.open(encoding=enc) as _f:
            return True, yaml.load(_f, cls.loader)

    @classmethod
    @Handle_exp()
    def Write_to(
        cls, file: Path, data: Any, enc: str = "UTF-8", indent: int = 4
    ) -> bool:
        """데이터를 YAML 형식으로 파일에 저장.

        Args:
            file: 저장 파일 경로.
            data: 저장 데이터.
            enc: 인코딩.
            indent: 들여쓰기 수준.

        Returns:
            저장 성공 여부.
        """
        cls.Ensure_dir(file)
        _, _path = Suffix_check(file, ".yaml", True)

        with _path.open(mode="w", encoding=enc) as _f:
            yaml.dump(data, _f, indent=indent, sort_keys=False)
        return True