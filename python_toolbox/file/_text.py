"""텍스트(.txt) 파일 입출력."""
from __future__ import annotations
from pathlib import Path

from ._base import File_Process, Handle_exp, Suffix_check


class Text(File_Process):
    """텍스트 파일(.txt) 읽기/쓰기를 위한 처리 클래스."""

    @classmethod
    @Handle_exp()
    def Read_from(
        cls, file: Path,
        enc: str = "UTF-8", start: int = 0, delim: str = "\n"
    ) -> tuple[bool, list[str]]:
        """텍스트 파일을 읽어 구분자 기준으로 나눈 리스트 반환.

        Args:
            file: 읽을 파일 경로.
            enc: 인코딩.
            start: 읽기 시작 인덱스.
            delim: 구분자 (기본 줄바꿈).

        Returns:
            (성공 여부, 문자열 리스트).
        """
        _, _file = Suffix_check(file, ".txt")
        if _file.exists():
            return True, _file.read_text(enc).split(delim)[start:]
        return False, []

    @classmethod
    @Handle_exp()
    def Write_to(
        cls, file: Path, data: str | list[str], enc: str = "UTF-8",
        anno: list[str] | str | None = None
    ) -> bool:
        """텍스트 데이터를 파일로 저장 (선택적으로 상단 주석 포함).

        Args:
            file: 저장 파일 경로.
            data: 저장할 문자열 또는 리스트.
            enc: 인코딩.
            anno: 상단에 추가할 주석/메타.

        Returns:
            저장 성공 여부.
        """
        cls.Ensure_dir(file)
        _, _path = Suffix_check(file, ".txt", True)

        _data = (
            [anno] if isinstance(anno, str) else anno
        ) if anno else []
        _data += [data] if isinstance(data, str) else data

        _path.write_text("\n".join(_data), enc)
        return True