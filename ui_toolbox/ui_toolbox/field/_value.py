"""값 하나의 계약과 자료형 판별.

- 계약 - 값 하나를 받는 위젯이 만족할 것
- 판별 - 어느 자료형에 어느 위젯이 서나
"""

from __future__ import annotations

import types
from typing import Any, Union, get_args, get_origin

from PySide6.QtWidgets import QWidget

__all__ = ["Value", "list_pair", "list_str", "optional_float"]


class Value(QWidget):
    """값 하나를 받는 위젯의 계약.

    변경 신호는 각 위젯이 자기 자료형으로 선언한다 (`value_changed(float)` ·
    `value_changed(int)`). payload 자료형이 달라 여기서 하나로 안 묶는다.
    """

    def value(self) -> Any:
        """지금 값."""
        raise NotImplementedError(f"{type(self).__name__}.value")

    def set_value(self, value: Any) -> None:
        """값을 앉힌다. 신호는 안 낸다 - 복원과 사람의 편집을 가르기 위해."""
        raise NotImplementedError(f"{type(self).__name__}.set_value")


def _strip_optional(tp):
    """`X | None` 이면 `X`, 아니면 그대로."""
    _origin = get_origin(tp)
    _is_union = _origin is Union or (
        hasattr(types, "UnionType") and isinstance(tp, types.UnionType))
    if not _is_union:
        return tp
    # nullable 목록도 목록 위젯을 받게 한다 - 위젯이 안 생기면 그 값이 통째로 유실된다
    _rest = [_a for _a in get_args(tp) if _a is not type(None)]
    return _rest[0] if len(_rest) == 1 else tp


def list_str(tp) -> bool:
    """`list[str]` · `list[Any]` 인가 (Optional 포함)."""
    _tp = _strip_optional(tp)
    return get_origin(_tp) is list and get_args(_tp) in ((str,), (Any,))


def list_pair(tp) -> bool:
    """`list[tuple[str, str]]` 인가 (Optional 포함)."""
    _tp = _strip_optional(tp)
    return get_origin(_tp) is list and get_args(_tp) == (tuple[str, str],)


def optional_float(tp) -> bool:
    """`float | None` 인가."""
    _origin = get_origin(tp)
    _is_union = _origin is Union or (
        hasattr(types, "UnionType") and isinstance(tp, types.UnionType))
    return _is_union and set(get_args(tp)) == {float, type(None)}
