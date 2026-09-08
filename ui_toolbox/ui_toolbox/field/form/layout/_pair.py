"""key/value 쌍 목록 - layout 표현의 소비처.

칸이 둘로 고정이라 표로도 설 수 있지만, 행마다 파일 찾기 · ROI 그리기 버튼이 붙어 layout 표현
쪽이 맞다. 중복 key 와 순서를 허용하므로 dict 이 아니라 쌍의 목록으로 왕복한다.
"""

from __future__ import annotations

from PySide6.QtWidgets import QFileDialog, QWidget

from ..._item import Button
from ..widget import Button_bar
from ..._field import Field, Rows
from ._stack import Stack_view

KEY   = "key"
VALUE = "value"

_BROWSE = 0
_DRAW   = 1


class Pair_editor(Stack_view):
    """``(key, value)`` 쌍을 행 단위로 편집.

    Attributes:
        changed: 행 추가 · 삭제 · 편집 시 emit.
    """

    def __init__(self, title: str = "", kind: str | None = None, tip: str = "",
                 draw: bool = False, roi_provider=None,
                 parent: QWidget | None = None) -> None:
        """편집기를 구성.

        Args:
            title: 맨 위 라벨. 빈 문자열이면 생략.
            kind: ``'path'`` 면 값 칸에 파일 찾기 버튼을 붙임.
            tip: 위젯 툴팁.
            draw: True 이고 ``roi_provider`` 가 있으면 ROI 그리기 버튼을 붙임.
            roi_provider: key 를 받아 마스크 PNG 경로를 돌려주는 콜백.
            parent: 부모 위젯.
        """
        self._kind = kind
        self._draw = draw and roi_provider is not None
        self._roi_provider = roi_provider
        super().__init__(
            Rows([Field(KEY, KEY, width=140), Field(VALUE, VALUE)]),
            title=title, header=False, parent=parent)
        if tip:
            self.setToolTip(tip)

    def _extras(self, at: int) -> list[QWidget]:
        """행 끝 버튼 - 파일 찾기 · ROI 그리기. 선언한 것만 붙음."""
        _buttons = []
        if self._kind == "path":
            _buttons.append(Button("📁", "파일 찾기", value=_BROWSE))
        if self._draw:
            _buttons.append(Button("🖉", "ROI 그리기 -> 마스크 PNG (파일명 = key)", value=_DRAW))
        if not _buttons:
            return []
        _bar = Button_bar(_buttons)
        _bar.fired.connect(lambda _what, _a=at: self._on_button(_a, _what))
        return [_bar]

    def _on_button(self, at: int, what: int) -> None:
        """행 끝 버튼 - 고른 경로를 값 칸에 적음. 취소면 그대로 둠."""
        if what == _BROWSE:
            _path, _ = QFileDialog.getOpenFileName(self, "파일 선택")
        else:
            _path = self._roi_provider(str(self._data.get(at, KEY) or "").strip())
        if _path and self._data.set(at, VALUE, _path):
            self.refresh()
            self.changed.emit()

    # ── public API ────────────────────────────────────────────────────────────
    def pairs(self) -> list[tuple[str, str]]:
        """현재 행들을 ``(key, value)`` 목록으로. key 가 빈 행은 뺌."""
        return [(str(_r.get(KEY, "")), str(_r.get(VALUE, "")))
                for _r in self.rows() if str(_r.get(KEY, "")).strip()]

    def set_pairs(self, pairs: list[tuple[str, str]] | None) -> None:
        """쌍 목록으로 행을 다시 채움. 로드는 ``changed`` 를 내지 않음."""
        self.load([{KEY: str(_k), VALUE: str(_v)} for _k, _v in (pairs or [])])

    def append_pair(self, key: str, value: str) -> None:
        """행 하나를 붙이고 ``changed`` 를 냄."""
        self._data.append({KEY: key, VALUE: value})
        self.refresh()
        self.changed.emit()
