"""Log_Level, Log_Line, Logger 단위 테스트."""
from dataclasses import dataclass, field
from typing import Any

import pytest

from python_toolbox.log import Log_Level, Log_Line, Logger


# =============================================================================
# Log_Level
# =============================================================================

def test_log_level_values():
    """레벨 상수 순서 및 정수값 검증."""
    assert Log_Level.TRA < Log_Level.DBG < Log_Level.INF
    assert Log_Level.INF < Log_Level.WRN < Log_Level.ERR < Log_Level.CRI
    assert Log_Level.TRA == 0 and Log_Level.CRI == 5


# =============================================================================
# Log_Line
# =============================================================================

def test_log_line_defaults():
    """기본 생성 — level=INF, timestamp 자동, info 빈 dict."""
    line = Log_Line()
    assert line.level == Log_Level.INF
    assert isinstance(line.timestamp, str) and len(line.timestamp) > 0
    assert line.info == {}


def test_log_line_level_clamping():
    """level 범위 초과 → 0~5로 클램핑."""
    assert Log_Line(level=-1).level == 0
    assert Log_Line(level=99).level == 5
    assert Log_Line(level=3).level == 3


def test_log_line_format_contains_level_and_info():
    """Format() 결과에 레벨명과 info 키-값이 포함됨."""
    line = Log_Line(level=Log_Level.WRN, info={"loss": 0.5})
    text = line.Format()
    assert "WRN" in text
    assert "loss=0.5" in text


def test_log_line_format_empty_info():
    """info가 빈 경우 Format() 정상 반환."""
    text = Log_Line(level=Log_Level.ERR).Format()
    assert "ERR" in text


# =============================================================================
# Logger
# =============================================================================

@pytest.fixture
def logger():
    return Logger(Log_Line)


def test_logger_info_appends_line(logger):
    """Info 호출 → book에 Log_Line 추가, level=INF."""
    logger.Info(msg="start")
    assert len(logger.book) == 1
    assert logger.book[0].level == Log_Level.INF


def test_logger_all_levels(logger):
    """각 레벨 메서드 → 대응하는 level 저장."""
    logger.Trace()
    logger.Debug()
    logger.Info()
    logger.Warning()
    logger.Error()
    logger.Critical()
    levels = [line.level for line in logger.book]
    assert levels == [0, 1, 2, 3, 4, 5]


def test_logger_field_vs_info_dispatch(logger):
    """Log_Line 필드명 kwargs → 직접 매핑, 나머지 → info dict."""
    logger.Info(info={"explicit": True}, extra_key="goes_to_info")
    line = logger.book[0]
    assert line.info.get("explicit") is True
    assert line.info.get("extra_key") == "goes_to_info"


def test_logger_custom_line_type():
    """custom line_type 지정 시 해당 타입으로 로그 생성."""
    @dataclass
    class _CustomLine(Log_Line):
        tag: str = ""

    log = Logger(_CustomLine)
    log.Info(line_type=_CustomLine, tag="my_tag")
    assert isinstance(log.book[0], _CustomLine)
    assert log.book[0].tag == "my_tag"


def test_logger_print_to_console_all(logger, capsys):
    """line_num=None → 전체 출력."""
    logger.Info(info={"a": 1})
    logger.Warning(info={"b": 2})
    logger.Print_to_console()
    out = capsys.readouterr().out
    assert "INF" in out and "WRN" in out


def test_logger_print_to_console_single(logger, capsys):
    """단일 인덱스 → 해당 라인만 출력."""
    logger.Info()
    logger.Error()
    logger.Print_to_console(1)
    out = capsys.readouterr().out
    assert "ERR" in out
    assert "INF" not in out


def test_logger_print_to_console_empty(logger, capsys):
    """book이 비어 있으면 아무것도 출력하지 않음."""
    logger.Print_to_console()
    assert capsys.readouterr().out == ""


def test_logger_field_cache_populated(logger):
    """_Logging 이후 _field_cache에 타입이 캐싱됨."""
    logger.Info()
    assert hasattr(logger, "_field_cache")
    assert Log_Line in logger._field_cache
