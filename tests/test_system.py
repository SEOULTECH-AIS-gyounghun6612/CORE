"""String, Operating_System, Time_Utils 단위 테스트."""
import platform
from datetime import datetime, timezone

import pytest

from python_toolbox.system import String, Operating_System, Time_Utils


# =============================================================================
# String.Count_auto_align
# =============================================================================

def test_count_auto_align_right():
    """우측 정렬(기본) — 자리수 맞춤 및 슬래시 포함."""
    result = String.Count_auto_align(3, 100)
    assert result == "003/100"


def test_count_auto_align_left():
    """is_right=False → 좌측 정렬."""
    result = String.Count_auto_align(3, 100, is_right=False)
    assert result.endswith("/100")
    assert result.startswith("3")


def test_count_auto_align_fill_char():
    """커스텀 fill 문자 적용."""
    result = String.Count_auto_align(5, 100, fill=" ")
    assert result == "  5/100"


# =============================================================================
# String.Str_adjust
# =============================================================================

def test_str_adjust_right_align():
    """우측 정렬 — 패딩 추가."""
    overflow, result = String.Str_adjust("hi", 6, align="r")
    assert overflow == 0
    assert result.endswith("hi")
    assert len(result) == 6


def test_str_adjust_left_align():
    """좌측 정렬 — 우측에 패딩."""
    overflow, result = String.Str_adjust("hi", 6, align="l")
    assert overflow == 0
    assert result.startswith("hi")
    assert len(result) == 6


def test_str_adjust_center_align():
    """중앙 정렬 — 양쪽 패딩."""
    overflow, result = String.Str_adjust("hi", 6, align="c")
    assert overflow == 0
    assert "hi" in result
    assert len(result) == 6


def test_str_adjust_overflow():
    """텍스트가 max_length 초과 → overflow > 0."""
    overflow, result = String.Str_adjust("toolong", 3, align="r")
    assert overflow > 0
    assert result == "toolong"


# =============================================================================
# Operating_System.Matches_os
# =============================================================================

def test_matches_os_current_platform():
    """현재 OS와 일치하는 이름으로 True 반환."""
    current = platform.system().lower()
    assert Operating_System.Matches_os(current) is True


def test_matches_os_wrong_platform():
    """다른 OS 이름으로 False 반환."""
    assert Operating_System.Matches_os("__nonexistent_os__") is False


# =============================================================================
# Time_Utils.Stamp
# =============================================================================

def test_stamp_returns_datetime():
    """`Stamp()` → datetime 객체 반환."""
    t = Time_Utils.Stamp()
    assert isinstance(t, datetime)


def test_stamp_with_timezone():
    """timezone 지정 시 timezone-aware datetime."""
    t = Time_Utils.Stamp(timezone.utc)
    assert t.tzinfo is not None


# =============================================================================
# Time_Utils.Get_term
# =============================================================================

def test_get_term_positive():
    """과거 기준 시간 → 양의 timedelta."""
    past = Time_Utils.Stamp()
    term = Time_Utils.Get_term(past)
    assert term.total_seconds() >= 0


# =============================================================================
# Time_Utils.Make_text_from
# =============================================================================

def test_make_text_from_iso():
    """포맷 없이 호출 → ISO 8601 문자열."""
    text = Time_Utils.Make_text_from()
    assert "T" in text


def test_make_text_from_custom_format():
    """커스텀 포맷 → 해당 형식 문자열."""
    dt = datetime(2024, 1, 15, 10, 30, 0)
    text = Time_Utils.Make_text_from(dt, d_fmt="%Y/%m/%d")
    assert text == "2024/01/15"


# =============================================================================
# Time_Utils.Make_time_from
# =============================================================================

def test_make_time_from_iso():
    """ISO 문자열 → datetime 객체."""
    dt = Time_Utils.Make_time_from("2024-06-01T12:00:00")
    assert dt.year == 2024 and dt.month == 6 and dt.day == 1


def test_make_time_from_custom_format():
    """커스텀 포맷 문자열 → datetime."""
    dt = Time_Utils.Make_time_from("15/01/2024", d_fmt="%d/%m/%Y")
    assert dt.year == 2024 and dt.day == 15


def test_make_time_from_invalid_raises():
    """포맷 불일치 → ValueError."""
    with pytest.raises(ValueError):
        Time_Utils.Make_time_from("not-a-date", d_fmt="%Y-%m-%d")


# =============================================================================
# Time_Utils.Relative
# =============================================================================

def test_relative_to_delta_and_back():
    """Relative → relativedelta 변환 및 역변환."""
    rel = Time_Utils.Relative(years=1, months=2, days=3)
    delta = rel.to_delta()
    assert delta.years == 1 and delta.months == 2 and delta.days == 3

    restored = Time_Utils.Relative.from_delta(delta)
    assert restored.years == 1 and restored.months == 2 and restored.days == 3
