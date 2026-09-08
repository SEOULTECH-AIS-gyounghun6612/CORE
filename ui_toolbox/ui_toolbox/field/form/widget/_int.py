"""정수 입력 행 - 슬라이더와 스핀박스를 맞물림."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal

from ....style import LABEL, VALUE, Mark
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QSpinBox, QWidget


class Int_slider_row(QWidget):
    """슬라이더와 스핀박스를 동기화한 정수 입력 행.

    Attributes:
        value_changed: 값이 바뀔 때 새 정수값을 emit하는 시그널.
    """

    value_changed = Signal(int)

    def __init__(
        self,
        label: str,
        min_val: int,
        max_val: int,
        default: int,
        tooltip: str = "",
        parent: QWidget | None = None,
    ) -> None:
        """행을 구성한다.

        Args:
            label: 좌측 라벨 텍스트.
            min_val: 최솟값.
            max_val: 최댓값.
            default: 초기값.
            tooltip: 위젯 툴팁.
            parent: 부모 위젯.
        """
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl = Mark(QLabel(label), LABEL)
        if tooltip:
            self.setToolTip(tooltip)
        layout.addWidget(lbl)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(min_val, max_val)
        self._slider.setValue(default)
        layout.addWidget(self._slider, stretch=1)

        self._spin = QSpinBox()
        self._spin.setRange(min_val, max_val)
        self._spin.setValue(default)
        Mark(self._spin, VALUE)
        layout.addWidget(self._spin)

        self._slider.valueChanged.connect(self._spin.setValue)
        self._spin.valueChanged.connect(self._slider.setValue)
        self._slider.valueChanged.connect(self.value_changed)

    def value(self) -> int:
        """현재 값을 반환한다."""
        return self._spin.value()

    def set_value(self, v: int) -> None:
        """값을 설정한다 — **시그널 없이**(복원용). 슬라이더·스핀을 함께 맞춘다."""
        self._spin.blockSignals(True)
        self._slider.blockSignals(True)
        self._spin.setValue(int(v))
        self._slider.setValue(int(v))
        self._spin.blockSignals(False)
        self._slider.blockSignals(False)
