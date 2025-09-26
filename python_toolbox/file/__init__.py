"""파일 처리 통합 패키지 — 텍스트/JSON/YAML 입출력 + 예외 처리.

확장자 기반 디스패치(Read_from/Write_to)로 포맷 무관 파일 처리가 가능하며,
포맷별 클래스(Text/Json/Yaml)로 세부 동작을 직접 호출할 수도 있음.

## 모듈 구성
- `_base`: File_Process ABC, Handle_exp 데코레이터, Suffix_check, 에러 카탈로그
- `_text` / `_json` / `_yaml`: 포맷별 처리 클래스
- `dispatch`: 확장자 기반 디스패치 함수 (Read_from / Write_to)
- `group`: 파일 그룹 디렉토리 생성 유틸 (Make_the_file_group)
"""
from ._base import (
    Handle_exp,
    File_Process,
    Suffix_check,
    BASIC_FILE_ERROR,
    JSON_FILE_READ_ERROR,
)
from ._text import Text
from ._json import Json
from ._yaml import Yaml
from .dispatch import Read_from, Write_to
from .group import Make_the_file_group


__all__ = [
    "Handle_exp",
    "File_Process",
    "Suffix_check",
    "BASIC_FILE_ERROR",
    "JSON_FILE_READ_ERROR",
    "Text",
    "Json",
    "Yaml",
    "Read_from",
    "Write_to",
    "Make_the_file_group",
]