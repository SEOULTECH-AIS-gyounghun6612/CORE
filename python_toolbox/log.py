"""Data_Schema 기반의 확장 가능한 구조적 로깅 모듈.

포맷팅(직렬화)은 Data_Schema를 상속받은 Record 객체가 자체적으로 수행하며,
Logger는 이를 전달받아 콘솔, 파일 등의 목적지로 출력하는 단일 역할을 수행함.
시스템 로그와 AI 학습 로그 등 서로 다른 목적의 로그를 Record 확장을 통해 지원함.
"""
from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass, field, fields
from typing import Any, TypeVar, Literal, cast

from .data_schema import Data_Schema
from .system import Time_Utils


class Log_Level(IntEnum):
    """로그 심각도 정의."""
    TRA = 0  # Trace: 함수 진입/루프 등 매우 상세한 흐름
    DBG = 1  # Debug: 디버깅용 변수 상태 및 크기
    INF = 2  # Info: 일반적인 시스템 정상 작동 상태
    WRN = 3  # Warning: 경고성 이벤트 (무시 가능하나 주의 필요)
    ERR = 4  # Error: 에러 발생 (기능 일부 동작 실패)
    CRI = 5  # Critical: 시스템 종료 수준의 치명적 오류


@dataclass
class Log_Line(Data_Schema):
    """모든 로그의 최상위 베이스 스키마.

    Data_Schema를 상속하여 Serialize/Extract 기능을 기본 제공하며,
    발생 시간을 자동으로 기록함.
    """
    level: Literal[0, 1, 2, 3, 4, 5] = 2
    timestamp: str = field(
        default_factory=lambda: Time_Utils.Make_text_from()
    )
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 비정상적인 레벨 값이 들어오는 것을 방지 (0~5 사이로 클램핑)
        self.level = cast(
            Literal[0, 1, 2, 3, 4, 5], max(0, min(self.level, 5))
        )

    def Format(self) -> str:
        """콘솔 출력을 위한 문자열 포맷팅 메서드.
        
        자식 클래스에서 이 메서드를 오버라이드하여 출력 형식을 커스텀할 수 있음.
        """
        _level_name = Log_Level(self.level).name
        _info_str = ", ".join(f"{k}={v}" for k, v in self.info.items())
        return f"[{self.timestamp}] {_level_name} - {_info_str}"


LINE_TYPE = type[TypeVar("LINE", bound=Log_Line)] | None

@dataclass
class Logger(Data_Schema):
    _default_type: type[Log_Line]
    book: list[Log_Line] = field(default_factory=list)

    def Print_to_console(self, line_num: int | list[int] | None = None):
        """저장된 로그 중 지정된 인덱스의 로그를 콘솔에 출력함.
        
        Args:
            line_num: 출력할 로그의 인덱스 (단일 int, int 리스트, None이면 전체 출력)
        """
        if not self.book:
            return

        _targets = []
        if line_num is None:
            _targets = self.book
        elif isinstance(line_num, int):
            _targets = [self.book[line_num]]
        else:
            _targets = [self.book[i] for i in line_num if i < len(self.book)]

        for _line in _targets:
            print(_line.Format())

    def _Logging(self, level: Log_Level, line_type: LINE_TYPE, **kwarg):
        _type = line_type or self._default_type

        # 로깅이 빈번하게 호출되므로, 리플렉션(fields) 연산을 클래스 타입별로 캐싱하여 속도 최적화
        if getattr(self, "_field_cache", None) is None:
            self._field_cache: dict[type, set[str]] = {}
            
        if _type not in self._field_cache:
            self._field_cache[_type] = {f.name for f in fields(_type)}
            
        _valid_fields = self._field_cache[_type]
        
        _init_args: dict[str, Any] = {"level": level.value}
        _info_dict: dict[str, Any] = {}

        # kwargs를 분석하여 명시된 필드면 직접 매핑, 아니면 info로 분류
        for k, v in kwarg.items():
            if k in _valid_fields:
                _init_args[k] = v
            else:
                _info_dict[k] = v
                
        _init_args["info"] = _info_dict

        self.book.append(_type(**_init_args))

    def Trace(self, line_type: LINE_TYPE = None, **kwargs: Any) -> None:
        self._Logging(Log_Level.TRA, line_type, **kwargs)

    def Debug(self, line_type: LINE_TYPE = None, **kwargs: Any) -> None:
        self._Logging(Log_Level.DBG, line_type, **kwargs)

    def Info(self, line_type: LINE_TYPE = None, **kwargs: Any) -> None:
        self._Logging(Log_Level.INF, line_type, **kwargs)

    def Warning(self, line_type: LINE_TYPE = None, **kwargs: Any) -> None:
        self._Logging(Log_Level.WRN, line_type, **kwargs)

    def Error(self, line_type: LINE_TYPE = None, **kwargs: Any) -> None:
        self._Logging(Log_Level.ERR, line_type, **kwargs)

    def Critical(self, line_type: LINE_TYPE = None, **kwargs: Any) -> None:
        self._Logging(Log_Level.CRI, line_type, **kwargs)