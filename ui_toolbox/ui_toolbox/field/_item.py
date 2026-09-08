"""조각 선언 - 무엇이 붙고 눌렸을 때 무엇을 내나."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Button"]


@dataclass(frozen=True)
class Button:
    """버튼 하나 선언.

    Attributes:
        text: 글리프 또는 짧은 라벨.
        tip: 툴팁. 글리프만으로는 무엇인지 안 보이므로 비우지 않음.
        value: 눌렸을 때 낼 값.
        enabled: 처음에 켜져 있나.
    """

    text:    str
    tip:     str = ""
    value:   int | None = None
    enabled: bool = True
